import os
import shutil
import sys

import torch
from ultralytics import YOLO

from src.dataset_manager.Picsellia import Picsellia
from src.file_reader.XmlTrainReader import XmlTrainReader
from src.yolo_training.YoloPrepareData import YoloPrepareData

if __name__ == '__main__':
    print(torch.backends.mps.is_available())
    print(torch.cuda.is_available())

    xml_reader = XmlTrainReader("config.train.xml")
    yolo_config_file_name = "yolo_config.yaml"
    dataset_path = xml_reader.dataset_path

    yolo_config_file = f'{dataset_path}/{yolo_config_file_name}'

    if len(sys.argv) >= 2 and sys.argv[1] == "-clear":
        pics = Picsellia(
            api_token=xml_reader.api_token,
            organization_name=xml_reader.organization_name)

        exp = pics.get_experiment(
            project_name=xml_reader.project_name,
            experiment_name=xml_reader.experiment_name,
            force_create=True
        )


        dataset = pics.get_dataset(xml_reader.dataset_id)

        Picsellia.attach_dataset(experiment=exp, dataset=dataset)

        shutil.rmtree(dataset_path, ignore_errors=True)
        # TODO : clear le dataset si exist
        yolo_config_file = YoloPrepareData(dataset).prepare_new_dataset(dataset_path)

        #https://documentation.picsellia.com/docs/evaluate-your-model-performances
        #https://documentation.picsellia.com/docs/experiment-experiment-tracking

    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    if torch.backends.mps.is_available():
        device_type = "mps"
    elif torch.cuda.is_available():
        device_type = "cuda"
    else:
        device_type = "cpu"

    # Train the model
    results = model.train(data=yolo_config_file, epochs=10, imgsz=640, close_mosaic=0, device=device_type)

    eval = model.val()