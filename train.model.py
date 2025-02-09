import os.path

from ultralytics import YOLO

from src.dataset_manager.Picsellia import Picsellia
from src.file_reader.XmlPicselliaReader import XmlPicselliaReader
from src.file_reader.XmlTrainReader import XmlTrainReader
from src.yolo_training.DeviceDetector import DeviceDetector
from src.yolo_training.YoloConfig import YoloConfig
from src.yolo_training.YoloPredictCallbacks import YoloPredictCallbacks
from src.yolo_training.YoloPrepareData import YoloPrepareData
from src.yolo_training.YoloTrainingCallback import YoloTrainingCallback


if __name__ == '__main__':

    xml_picsellia_config = XmlPicselliaReader("config/picsellia.config.xml")
    xml_train_config = XmlTrainReader("config/config.train.xml")
    dataset_path = xml_train_config.dataset_path

    yolo_config = YoloConfig(dataset_path)

    seed = 42

    pics = Picsellia(
        api_token=xml_picsellia_config.api_token,
        organization_name=xml_picsellia_config.organization_name,
        project_name=xml_picsellia_config.project_name,
        experiment_name=xml_train_config.experiment_name,
        dataset_id=xml_train_config.dataset_version_id
    )

    pics.attach_current_dataset()

    if os.path.exists(dataset_path) is False:
        yolo_config = YoloPrepareData(pics.dataset).prepare_new_dataset(dataset_path, seed=seed)

    device_type = DeviceDetector.get_device_type()
    print(f'Device type: {device_type}')

    model = YOLO("yolo11m.pt")

    YoloTrainingCallback(pics, xml_picsellia_config).apply_callbacks(model, send_metrics_on_epoch_end=xml_train_config.send_metrics_on_epoch_end)
    YoloPredictCallbacks(pics).apply_callbacks(model)

    model.train(
        data=yolo_config.file_path,
        epochs=250,
        device=device_type,
        close_mosaic=0,
        seed=seed,
        patience= 20,
        lr0 = 0.00179,
        lrf = 0.01518,
        translate=0.1,
        mosaic = 0.1,
        scale = 0.5,
        shear = 10,
        flipud = 0.5
    )

    model.predict(yolo_config.images_test, device=device_type)