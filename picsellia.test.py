import sys

from ultralytics import YOLO

from src.dataset_manager.Picsellia import Picsellia
from src.file_reader.XmlReader import XmlReader
from src.yolo_training.YoloPrepareData import YoloPrepareData


if len(sys.argv) >= 2:
    if sys.argv[1] == "clear":
        xml_reader = XmlReader("./config.xml")
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

        yolo_config_file = YoloPrepareData(dataset).prepare_new_dataset("./dataset/v4")
        # yolo_config_file = "./dataset/v1/yolo_config.yaml"
        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data=yolo_config_file, epochs=1, imgsz=640, close_mosaic=0)

        eval = model.val()

