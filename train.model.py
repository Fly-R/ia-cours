import shutil
import sys

import pandas as pd
import torch
from picsellia.types.enums import LogType
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from src.dataset_manager.Picsellia import Picsellia
from src.file_reader.XmlPicselliaReader import XmlPicselliaReader
from src.file_reader.XmlTrainReader import XmlTrainReader
from src.yolo_training.YoloPrepareData import YoloPrepareData

def on_train_end(trainer:DetectionTrainer):
    metrics_csv = pd.read_csv(trainer.csv)
    for col in metrics_csv.columns:
        try:
            experiment.log(name=col, data=list(metrics_csv[col]), type=LogType.LINE)
        except Exception as e:
            print(f'Error: {e}')

    pics.upload_model_version(xml_picsellia_config.project_name, trainer.best)
    experiment.log_parameters({
        "batch_size" : trainer.args.batch,
        "imgsz" : trainer.args.imgsz,
        "device" : trainer.args.device,
        "workers" : trainer.args.workers,
        "optimize" : trainer.args.optimize,
        "optimizer" : trainer.args.optimizer,
        "lr0" : trainer.args.lr0,
        "lrf" : trainer.args.lrf,
        "patience" : trainer.args.patience,
        "epochs" : trainer.args.epochs,
        "seed" : trainer.args.seed,
    })
    pics.upload_artifact("confusion_matrix", f'{trainer.save_dir}/confusion_matrix.png')
    pics.upload_artifact("F1_curve", f'{trainer.save_dir}/F1_curve.png')
    pics.upload_artifact("labels", f'{trainer.save_dir}/labels.jpg')

if __name__ == '__main__':

    xml_picsellia_config = XmlPicselliaReader("picsellia.config.xml")
    xml_train_config = XmlTrainReader("config.train.xml")
    yolo_config_file_name = "yolo_config.yaml"
    dataset_path = xml_train_config.dataset_path

    yolo_config_file = f'{dataset_path}/{yolo_config_file_name}'

    pics = Picsellia(
        api_token=xml_picsellia_config.api_token,
        organization_name=xml_picsellia_config.organization_name)

    experiment = pics.set_experiment(
        project_name=xml_picsellia_config.project_name,
        experiment_name=xml_train_config.experiment_name,
        force_create=True
    )

    dataset = pics.set_dataset(xml_train_config.dataset_version_id)

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

    print(f'Device type: {device_type}')
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    model.add_callback("on_train_end", on_train_end)

    # Train the model
    results = model.train(
        data=yolo_config_file,
        epochs=2,
        imgsz=640,
        close_mosaic=0,
        device=device_type,
        batch=16,
        patience=100,
    )

    #eval = model.val()