import pandas as pd
from picsellia.types.enums import LogType
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

from src.dataset_manager.Picsellia import Picsellia
from src.file_reader.XmlPicselliaReader import XmlPicselliaReader


class YoloTrainingCallback:

    def __init__(self, pics:Picsellia, xml_config:XmlPicselliaReader):
        self.__pics = pics
        self.__xml_config = xml_config

    def apply_callbacks(self, model:YOLO, send_metrics_on_epoch_end=True):
        if send_metrics_on_epoch_end:
            model.add_callback("on_train_epoch_end", self.__send_metrics_on_train_epoch_end)
            model.add_callback("on_val_end", self.__send_metrics_on_val_end)
        else:
            model.add_callback("on_train_end", self.__send_metrics_on_train_end)

        model.add_callback("on_train_end", self.__on_train_end)

    def __on_train_end(self, trainer:DetectionTrainer):
        self.__pics.upload_model_version(self.__xml_config.project_name, trainer.best)
        self.__pics.experiment.log_parameters(trainer.args.__dict__)
        self.__pics.upload_artifact("confusion_matrix", f'{trainer.save_dir}/confusion_matrix.png')
        self.__pics.upload_artifact("confusion_matrix_normalized", f'{trainer.save_dir}/confusion_matrix_normalized.png')
        self.__pics.upload_artifact("F1_curve", f'{trainer.save_dir}/F1_curve.png')
        self.__pics.upload_artifact("labels", f'{trainer.save_dir}/labels.jpg')

    def __send_metrics_on_train_end(self, trainer:DetectionTrainer):
        experiment = self.__pics.experiment
        metrics_csv = pd.read_csv(trainer.csv)

        experiment.log(name="precision", data=metrics_csv["metrics/precision(B)"].tolist(), type=LogType.LINE)
        experiment.log(name="precision", data=metrics_csv["metrics/recall(B)"].tolist(), type=LogType.LINE)
        experiment.log(name="precision", data=metrics_csv["metrics/mAP50(B)"].tolist(), type=LogType.LINE)
        experiment.log(name="precision", data=metrics_csv["metrics/mAP50-95(B)"].tolist(), type=LogType.LINE)

        box_loss_data = {
            'train': metrics_csv["train/box_loss"].tolist(),
            'val': metrics_csv["val/box_loss"].tolist(),
        }
        experiment.log(name="box_loss", data=box_loss_data, type=LogType.LINE)

        cls_loss_data = {
            'train': metrics_csv["train/cls_loss"].tolist(),
            'val': metrics_csv["val/cls_loss"].tolist(),
        }
        experiment.log(name="cls_loss", data=cls_loss_data, type=LogType.LINE)

        dfl_loss_data = {
            'train': metrics_csv["train/dfl_loss"].tolist(),
            'val': metrics_csv["val/dfl_loss"].tolist(),
        }
        experiment.log(name="dfl_loss", data=dfl_loss_data, type=LogType.LINE)


    def __send_metrics_on_train_epoch_end(self, trainer:DetectionTrainer):
        experiment = self.__pics.experiment
        experiment.log(name="precision", data=float(trainer.metrics["metrics/precision(B)"]), type=LogType.LINE)
        experiment.log(name="recall", data=float(trainer.metrics["metrics/recall(B)"]), type=LogType.LINE)
        experiment.log(name="mAP50", data=float(trainer.metrics["metrics/mAP50(B)"]), type=LogType.LINE)
        experiment.log(name="mAP50-95", data=float(trainer.metrics["metrics/mAP50-95(B)"]), type=LogType.LINE)

        experiment.log(name="box_loss", data={'train':[float(trainer.loss_items[0].item())]}, type=LogType.LINE)
        experiment.log(name="cls_loss", data={'train':[float(trainer.loss_items[1].item())]}, type=LogType.LINE)
        experiment.log(name="dfl_loss", data={'train':[float(trainer.loss_items[2].item())]}, type=LogType.LINE)

    def __send_metrics_on_val_end(self, trainer:DetectionValidator):
        experiment = self.__pics.experiment
        experiment.log(name="box_loss", data={'val': [float(trainer.loss[0].item())]}, type=LogType.LINE)
        experiment.log(name="cls_loss", data={'val': [float(trainer.loss[1].item())]}, type=LogType.LINE)
        experiment.log(name="dfl_loss", data={'val': [float(trainer.loss[2].item())]}, type=LogType.LINE)