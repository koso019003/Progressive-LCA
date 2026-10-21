import os
import torch
import lightning
import numpy as np
from pytorch_lightning.utilities.model_summary import ModelSummary
from sklearn.metrics import confusion_matrix, classification_report
import rich
from rich.table import Table

from src.dataset import GTZANDataModule
from src.model import TimeSeqMoSEMERTModel
from src.utility import plot_confusion_matrix, predict_artist


def test_model(
    model_config,
    train_config,
    data_config,
    save_folder,
    model_weight,
):
    """
    Main function for training the model and testing
    """
    # Load data
    print("Loading dataset...")
    music_data = GTZANDataModule(
        train_config["pre_trained_model"],
        data_config=data_config,
        batch_size=train_config["batch_size"],
    )
    music_data.setup(stage="test")
    
    model:TimeSeqMoSEMERTModel = load_model(
        pretrained_mert=train_config["pre_trained_model"],
        config=model_config,
        model_weight=model_weight,
    )
    
    model.eval()
    model.set_phase("test")

    # Generate result
    ## Predict the test dataset
    print("Predicting test dataset result")
    trainer = lightning.Trainer(default_root_dir=save_folder, logger=False)
    pred = trainer.predict(
        model,
        dataloaders=music_data.test_dataloader(),
        return_predictions=True
    )
    y_pred, y_cfs = aggregate_pred(pred)
    
    ## Save and show results
    base_filename = os.path.join(save_folder, "testing_result")
    os.makedirs(base_filename, exist_ok=True)

    print(f"\nNumber of parameters: {ModelSummary(model).total_parameters / 10**6 :.2f} (M)")
    write_test_result(torch.softmax(y_pred, dim=-1), music_data, os.path.join(base_filename, "genreMERT_P-LCA"))

def aggregate_pred(list_pred):
    y_pred_ = []
    y_cf_ = []
    for batch_pred in list_pred:
        y_pred_.append(batch_pred[0])
        if batch_pred[1] is not None:
            y_cf_.append(batch_pred[1])
    if y_cf_:
        return torch.cat(y_pred_, dim=1), torch.cat(y_cf_, dim=1)
    return torch.cat(y_pred_, dim=1), None

def write_test_result(y_scores, data_module, base_filename):
    y_true = data_module.test_set.y
    song_list = data_module.test_set.s
    class_names_original = data_module.ids2class(np.arange(data_module.nb_classes))
    num_test = len(y_true)
    num_songs = len(np.unique(song_list))
    
    table = Table(title=f"{os.path.basename(base_filename)} result on {len(y_true)} testing data ")
    
    table.add_column("LCA Layer", justify="right", style="cyan", no_wrap=True)
    table.add_column("F-Accuracy", justify="right", style="green")
    table.add_column("F-F1-score", justify="right", style="green")
    table.add_column("S-Accuracy", justify="right", style="blue")
    table.add_column("S-F1-score", justify="right", style="blue")
    
    for i in range(y_scores.shape[0]):
        filename = base_filename + f"_Layer_{i+1}"
        y_score = y_scores[i]

        # Calculate and plot confusion matrix

        y_predict = torch.argmax(y_score, dim=-1)

        # Calculate and plot confusion matrix
        cm = confusion_matrix(y_true, y_predict)
        plot_confusion_matrix(
            cm,
            classes=class_names_original,
            filename=filename + "_frames_confusion_matrix",
            normalize=True,
        )

        scores = classification_report(
            y_true, y_predict, target_names=class_names_original, zero_division=1, digits=4
        )
        scores_dict = classification_report(
            y_true,
            y_predict,
            target_names=class_names_original,
            zero_division=1, 
            digits=4,
            output_dict=True,
        )

        # Predict artist using pooling methodology
        # (#branchs, #threshold, 1)
        multi_threshold_predictions, multi_threshold_passed_slices_num = predict_artist(
            y_pred=y_score,
            y_true=y_true,
            song_list=song_list,
            class_names=class_names_original,
        )
        pooling_max_acc, pooling_max_acc_idx = 0, 0
        for j in range(len(multi_threshold_predictions)):
            if multi_threshold_predictions[j][1]["accuracy"] > pooling_max_acc:
                pooling_max_acc = multi_threshold_predictions[j][1]["accuracy"]
                pooling_max_acc_idx = j
        
        # Print out metrics
        table.add_row(f"{i+1}", 
                      f'{scores_dict["accuracy"] :.4f}', 
                      f'{scores_dict["weighted avg"]["f1-score"] :.4f}',
                      f'{multi_threshold_predictions[pooling_max_acc_idx][1]["accuracy"] :.4f}',
                      f'{multi_threshold_predictions[pooling_max_acc_idx][1]["weighted avg"]["f1-score"] :.4f}',
                      )
        pooling_table = Table(title="Pooling results")
        pooling_table.add_column("Threshold:", justify="left")

        for i in range(10):
            tag = "**" if i == pooling_max_acc_idx else ""
            pooling_table.add_column(f"{tag}{i/10}", justify="right")
        pooling_table.add_row("Accuracy", *[f"{multi_threshold_predictions[j][1]['accuracy'] :.4f}" for j in range(10)])
        pooling_table.add_row("F1-score", *[f"{multi_threshold_predictions[j][1]['weighted avg']['f1-score'] :.4f}" for j in range(10)])
        pooling_table.add_row("#Passing", *[f"{sum(multi_threshold_passed_slices_num[j])}" for j in range(10)])
            

        with open(filename + ".txt", "w") as f:
            f.write(f"Testing on {num_test} slices\n")
            f.write(
                f"nb_classes: {data_module.nb_classes}\n"
                + f"slice_length: {data_module.slice_length}s\n"
            )
            f.write(f"Test accuracy: {scores_dict['accuracy'] :.4f}\n")
            f.write(f"Test f1 score: {scores_dict['weighted avg']['f1-score'] :.4f}\n")
            f.write("Test results on frame level:\n")
            f.write(str(scores))
            f.write(f"\nScores when pooling song slices in {num_songs} songs:\n")
            rich.print(pooling_table, file=f)
            f.write(f"Threshold: {pooling_max_acc_idx/10}\n")
            f.write(f"Test accuracy: {multi_threshold_predictions[pooling_max_acc_idx][1]['accuracy'] :.4f}\n")
            f.write(f"Test f1 score: {multi_threshold_predictions[pooling_max_acc_idx][1]['weighted avg']['f1-score'] :.4f}\n")
            f.write("Test results on song level:\n")
            f.write(str(multi_threshold_predictions[pooling_max_acc_idx][0]))
    with open(base_filename + "_result.txt", "w") as f:
        rich.print(table, file=f)

    print(f"Test accuracy of {os.path.basename(filename).replace('_', ' ')}:")
    print(f'Frame level: {scores_dict["accuracy"] :.2f}\tSong level: {pooling_max_acc :.2f}')


def load_model(pretrained_mert, config, model_weight):
    checkpoint = torch.load(model_weight, map_location=torch.device("cpu"))

    # Load pretrained MERT weights
    from transformers import AutoModel
    mert = AutoModel.from_pretrained(pretrained_mert, trust_remote_code=True)
    for key in mert.state_dict():
        checkpoint["state_dict"][f"mert.{key}"] = mert.state_dict()[key]

    model = TimeSeqMoSEMERTModel(mert=pretrained_mert, config=config)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    return model
    