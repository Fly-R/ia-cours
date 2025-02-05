import os.path

import torch
from picsellia.types.enums import LogType
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionPredictor

from src.dataset_manager.Picsellia import Picsellia
from src.file_reader.XmlPicselliaReader import XmlPicselliaReader
from src.file_reader.XmlTrainReader import XmlTrainReader
from src.yolo_training.YoloPrepareData import YoloPrepareData

def on_train_epoch_end(trainer:DetectionTrainer):
    experiment.log(name="precision", data=trainer.metrics["metrics/precision(B)"], type=LogType.LINE)
    experiment.log(name="recall", data=trainer.metrics["metrics/recall(B)"], type=LogType.LINE)
    experiment.log(name="mAP50", data=trainer.metrics["metrics/mAP50(B)"], type=LogType.LINE)
    experiment.log(name="mAP50-95", data=trainer.metrics["metrics/mAP50-95(B)"], type=LogType.LINE)

    experiment.log(name="lr-pg0", data=trainer.lr["lr/pg0"].item(), type=LogType.LINE)
    experiment.log(name="lr-pg1", data=trainer.lr["lr/pg1"].item(), type=LogType.LINE)
    experiment.log(name="lr-pg2", data=trainer.lr["lr/pg2"].item(), type=LogType.LINE)

def on_train_end(trainer:DetectionTrainer):
    pics.upload_model_version(xml_picsellia_config.project_name, trainer.best)
    experiment.log_parameters(trainer.args.__dict__)
    pics.upload_artifact("confusion_matrix", f'{trainer.save_dir}/confusion_matrix.png')
    pics.upload_artifact("confusion_matrix_normalized", f'{trainer.save_dir}/confusion_matrix_normalized.png')
    pics.upload_artifact("F1_curve", f'{trainer.save_dir}/F1_curve.png')
    pics.upload_artifact("labels", f'{trainer.save_dir}/labels.jpg')

def on_predict_batch_end(predictor:DetectionPredictor):
    for item in predictor.results:
        detected_class_item_count = item.boxes.cls.shape[0]
        boxes = []
        for class_index in range(detected_class_item_count):
            class_id = int(item.boxes.cls[class_index])
            box = [int(i) for i in item.boxes.xywh[class_index].tolist()]
            box[0] = box[0] - box[2] // 2
            box[1] = box[1] - box[3] // 2
            label = dataset.get_label(item.names[class_id])
            conf = float(item.boxes.conf[class_index])
            box.append(label)
            box.append(conf)
            boxes.append(tuple(box))
        img = os.path.splitext(os.path.basename(item.path))[0]
        asset = dataset.find_asset(id=img)
        experiment.add_evaluation(asset, rectangles=boxes)


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

    if os.path.exists(dataset_path) is False:
        yolo_config_file = YoloPrepareData(dataset).prepare_new_dataset(dataset_path)

    if torch.backends.mps.is_available():
        device_type = "mps"
    elif torch.cuda.is_available():
        device_type = "cuda"
    else:
        device_type = "cpu"

    print(f'Device type: {device_type}')
    # Load a model
    model = YOLO("yolo11n.pt")

    model.add_callback("on_train_end", on_train_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_predict_batch_end", on_predict_batch_end)

    model.train(data=yolo_config_file,epochs=1, device=device_type)

    metrics = model.predict(f'{dataset_path}/images/test', device=device_type)