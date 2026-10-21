import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from lightning import LightningModule


class TimeSeqMoSEMERTModel(LightningModule):
    def __init__(
        self,
        mert,
        num_class=10,
        config=None,
    ):
        super().__init__()

        if "95M" in mert:
            mert_dim = 768
        elif "330M" in mert:
            mert_dim = 1024
        else:
            raise ValueError(f"Invalid MERT model: {mert}")

        self.loss_weights = config["loss_weights"]
        self.num_class = num_class
        self.use_lca_loss = True
        self.lca_layers = config["progressive_layers"]
        self.num_cf = config["mose_cf"]
        self.freeze_mert = True
        assert self.num_cf in (0, 1), "Number of counterfactual braches must be 0 or 1"

        # Model architecture
        self.feature_extract_layer = list(range(13))
        self.mert = AutoModel.from_pretrained(mert, trust_remote_code=True)
        if self.freeze_mert:
            for param in self.mert.parameters():
                param.requires_grad = False

        feature_num = len(self.feature_extract_layer)
        if feature_num > 1:
            self.aggregator = nn.Conv2d(
                in_channels=feature_num, out_channels=1, kernel_size=1
            )
        else:
            self.aggregator = nn.Identity()
            
        self.time_encode = nn.Sequential(
            SinusoidalPosEmb(mert_dim),
        )
        self.add_time = nn.Sequential(
            nn.Dropout(config["time_dropout"]),
        )
        
        self.z_dropout = nn.Dropout(config["main_dropout"])
        self.main_branch = AvgPoolAtt1D(random_att=False)
        if self.num_cf > 0:
            self.z_cf_dropout = nn.Dropout(config["cf_dropout"])
            self.cf_branch = AvgPoolAtt1D(random_att=False)

        self.hidden_layer = nn.Linear(mert_dim, config["representation_dim"])
        self.output_layer = nn.Linear(config["representation_dim"], num_class)
    
        self.intermediate_layer = AttIntermediate(hidden_size=mert_dim, activation=config["intermediate_act"])

        self.phase = "train"

    def set_phase(self, phase):
        assert phase in ("att", "rep", "test", "predict", "train", "val")
        self.phase = phase

    def forward(self, batch):
        if self.phase not in ("test", "rep", "att"):
            print("phase not set to test or rep, force set to test")
            self.phase = "test"
        return self._forward(batch)

    def _mert_encode(self, x):
        if self.freeze_mert:
            with torch.no_grad():
                outputs = self.mert(x, output_hidden_states=True)
        else:
            outputs = self.mert(x, output_hidden_states=True)
        # outputs (13 layer, batch_size, time_seq, 768 feature_dim)

        mert_feature = torch.stack(
            [outputs.hidden_states[i] for i in self.feature_extract_layer], dim=1
        ) # (batch_size, # layer, time_seq, 768 feature_dim)

        mert_feature = self.aggregator(mert_feature).squeeze(1) # (batch_size, time_seq, 768 feature_dim)
        return mert_feature

    def _forward_lca(self, x):
        
        x_reps, atts, logits = [], [], []
        x_cf_reps, att_cfs, logit_cfs,  = [], [], []
        
        for i in range(self.lca_layers):
            time_emb = self.time_encode(torch.tensor([i], dtype=x.dtype, device=x.device)).unsqueeze(1)
            mose_input = self.add_time(x + time_emb)
            
            att, _ = self.main_branch(mose_input) # (batch_size, time_seq, 1)
            z_main = torch.mul(x, x.shape[1] * att) # (batch_size, time_seq, 768 feature_dim)
            atts.append(att)
            
            z = torch.mean(z_main, dim=1, keepdim=False) # (batch_size, 768 feature_dim)
            z = self.z_dropout(z)

            x_rep = self.hidden_layer(z) # (batch, hidden_dim)
            x_reps.append(x_rep)

            logit = self.output_layer(x_rep)  # (batch, num_class)
            logits.append(logit)
        
            if self.num_cf > 0:
                att_cf, _ = self.cf_branch(mose_input)
                
                z_cf = torch.mul(x, x.shape[1] * att_cf)
                att_cfs.append(att_cf)
                
                z_cf = torch.mean(z_cf, dim=1, keepdim=False)
                z_cf = self.z_cf_dropout(z_cf)

                x_cf_rep = self.hidden_layer(z_cf) # (batch, hidden_dim)
                x_cf_reps.append(x_cf_rep)

                logit_cf = self.output_layer(x_cf_rep)  # (batch, num_class)
                logit_cfs.append(logit_cf)
            
            if i < self.lca_layers - 1:
                x = self.intermediate_layer(z_main)
            
        if self.num_cf > 0:
            return [
                (torch.stack(x_reps), torch.stack(logits), torch.stack(atts)),
                (torch.stack(x_cf_reps), torch.stack(logit_cfs), torch.stack(att_cfs)),
            ]
        return [
            (torch.stack(x_reps), torch.stack(logits), torch.stack(atts)),
            (None, None, None)
        ]

    def _forward(self, batch):
        x, y = batch
        # x (batch_size, time_seq)

        mert_feature = self._mert_encode(x)

        branch_out = self._forward_lca(x=mert_feature)
        
        if self.phase == "att":
            return [out[2] for out in branch_out]
        
        elif self.phase == "rep":
            return [out[0] for out in branch_out]
        
        elif self.phase in ("predict", "test"):
            if self.num_cf > 0:
                return [out[1] for out in branch_out]
            return branch_out[0][1], None
        
        return self.lca_loss(
            {
                "y_pred": branch_out[0][1], 
                "y_cf_pred": branch_out[1][1], 
                "att": branch_out[0][2],
                "att_cf": branch_out[1][2]
                }, 
            y
            )
        
    def lca_loss(self, lca_outputs, y):
        logit = lca_outputs["y_pred"]
        logit_cf = lca_outputs["y_cf_pred"]
        att = lca_outputs["att"]
        att_cf = lca_outputs["att_cf"]

        y_pred = F.softmax(logit, dim=-1)  # (batch, num_class)
        y_cf = None
        y_effect = None

        if self.num_cf > 0:
            y_cf = F.softmax(logit_cf, dim=-1)  # (batch, num_class)
            effect_logit = y_pred - y_cf
            y_effect = F.softmax(effect_logit, dim=-1)

        loss = 0
        for i in range(self.lca_layers):
            y_entropy = 0
            y_cf_loss = 0
            y_cf_entropy = 0
            y_effect_loss = 0
            attention_loss = 0
            
            y_loss = F.cross_entropy(logit[i], y)
            
            if self.num_cf > 0:
                y_cf_loss = F.cross_entropy(logit_cf[i], y)
                y_effect_loss = F.cross_entropy(effect_logit[i], y)
                y_entropy = -F.cross_entropy(logit[i], y_pred[i])
                y_cf_entropy = -F.cross_entropy(logit_cf[i], y_cf[i])
                attention_loss = -F.l1_loss(att[i], att_cf[i])
            
            loss += self.loss_weights["y_pred"] * y_loss

            if self.num_cf > 0 and self.use_lca_loss:
                loss += self.loss_weights["y_cf"] * y_cf_loss
                loss += self.loss_weights["y_effect"] * y_effect_loss
                loss += self.loss_weights["y_entropy"] * y_entropy
                loss += self.loss_weights["y_cf_entropy"] * y_cf_entropy
                loss += self.loss_weights["att_loss"] * attention_loss
        
        return loss, y_pred, y_cf, y_effect

    def on_load_checkpoint(self, checkpoint):
        return super().on_load_checkpoint(checkpoint)
    

class AvgPoolAtt1D(nn.Module):
    def __init__(self, random_att=False):
        super().__init__()
        self.random_att = random_att

        if not random_att:
            # tensor_input (batch_size, num_feature, d)
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            # (batch_size, num_feature, 1)

            self.attention = nn.Sequential( # (batch_size, 1, num_feature, 1)
                nn.Conv2d(1, 16, kernel_size=1, stride=1), # (batch_size, 16, num_feature, 1)
                nn.Tanh(),
                nn.Conv2d(16, 1, kernel_size=1, stride=1), # (batch_size, 1, num_feature, 1)
                nn.Flatten(start_dim=1, end_dim=2), # (batch_size, num_feature, 1)
            )
            self.att_act = nn.Softmax(dim=-2)

    def forward(self, x):
        # x (batch_size, num_feature, 768 feature_dim)
        if not self.random_att:
            feature = self.avg_pool(x)
            feature = torch.unsqueeze(feature, dim=1) # (batch_size, 1, num_feature, 1)

            c_out_att = self.attention(feature)
            att = self.att_act(c_out_att) # (batch_size, num_feature, 1)
            return att, c_out_att
        else:
            device = feature.get_device() if feature.get_device() >= 0 else None
            att = torch.rand(feature.shape[0], feature.shape[1], 1, device=device) * 2
        
            return att, att

class AttIntermediate(nn.Module):
    def __init__(self, hidden_size, activation):
        super().__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU() if activation == "gelu" else nn.ReLU())
        
    def forward(self, x):
        # x (batch_size, num_feature, 768 feature_dim)
        return self.dense1(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale = 1, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.scale = scale
        
    def forward(self, x: torch.Tensor):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = self.scale * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
