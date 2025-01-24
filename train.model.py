import shutil
import sys

import pandas as pd
import torch
from picsellia.types.enums import LogType
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from src.dataset_manager.Picsellia import Picsellia
from src.file_reader.XmlTrainReader import XmlTrainReader
from src.yolo_training.YoloPrepareData import YoloPrepareData


def on_train_end(trainer:DetectionTrainer):
    metrics_csv = pd.read_csv(trainer.csv)
    for col in metrics_csv.columns:
        try:
            experiment.log(name=col, data=list(metrics_csv[col]), type=LogType.LINE)
        except Exception as e:
            print(f'Error: {e}')

    pics.upload_model_version(xml_reader.project_name, f'{trainer.save_dir}/weights')
    experiment.log_parameters(config)

if __name__ == '__main__':

    xml_reader = XmlTrainReader("config.train.xml")
    yolo_config_file_name = "yolo_config.yaml"
    dataset_path = xml_reader.dataset_path

    yolo_config_file = f'{dataset_path}/{yolo_config_file_name}'

    pics = Picsellia(
        api_token=xml_reader.api_token,
        organization_name=xml_reader.organization_name)

    experiment = pics.set_experiment(
        project_name=xml_reader.project_name,
        experiment_name=xml_reader.experiment_name,
        force_create=True
    )

    dataset = pics.set_dataset(xml_reader.dataset_version_id)

    pics.attach_current_dataset()

    if len(sys.argv) >= 2 and sys.argv[1] == "-clear":
        shutil.rmtree(dataset_path, ignore_errors=True)
        yolo_config_file = YoloPrepareData(dataset).prepare_new_dataset(dataset_path)


    if torch.backends.mps.is_available():
        device_type = "mps"
    elif torch.cuda.is_available():
        device_type = "cuda"
    else:
        device_type = "cpu"

    config = {
        "data" : yolo_config_file,
        "batch_size" : 16,
        "imgsz" : 640,
        "device" : device_type,
        "workers" : 8,
        "optimizer" : "auto",
        "lr0" : 0.01,
        "patience" : 100,
        "epochs" : 2,
    }

    print(f'Device type: {device_type}')
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    model.add_callback("on_train_end", on_train_end)

    # Train the model
    results = model.train(
        data=config["data"],
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        close_mosaic=0,
        device=config["device"],
        batch=config["batch_size"],
        workers=config["workers"],
        patience=config["patience"],
        lr0=config["lr0"],
        optimizer=config["optimizer"]
    )

    #eval = model.val()