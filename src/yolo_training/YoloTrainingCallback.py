import pandas as pd
from picsellia.types.enums import LogType
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

from src.dataset_manager.Picsellia import Picsellia
from src.file_reader.XmlPicselliaReader import XmlPicselliaReader


class YoloTrainingCallback:

    __PRECISION = "precision"
    __RECALL = "recall"
    __MAP50 = "mAP50"
    __MAP50_95 = "mAP50-95"
    __BOX_LOSS = "box_loss"
    __CLS_LOSS = "cls_loss"
    __DFL_LOSS = "dfl_loss"

    __PRECISION_METRIC = "metrics/precision(B)"
    __RECALL_METRIC = "metrics/recall(B)"
    __MAP50_METRIC = "metrics/mAP50(B)"
    __MAP50_95_METRIC = "metrics/mAP50-95(B)"


    def __init__(self, pics:Picsellia, xml_config:XmlPicselliaReader):
        self.__pics = pics
        self.__xml_config = xml_config

    def apply_callbacks(self, model:YOLO, send_metrics_on_epoch_end:bool=True) -> None:
        """
        Add callbacks for the following events:
            - 'on_train_end' : Upload training artifacts to Picsellia
            - 'on_train_epoch_end' and 'on_val_end' : Upload training metrics to Picsellia

        If 'send_metrics_on_epoch_end' is set to True, metrics will be sent at the end of each epoch and validation.
        If set to False, all metrics will be sent at the end of the training process

        :param model: YOLO model to add the callback to
        :param send_metrics_on_epoch_end: Determines whether metrics are sent at each epoch end
        :return:
        """
        if send_metrics_on_epoch_end:
            model.add_callback("on_train_epoch_end", self.__send_metrics_on_train_epoch_end)
            model.add_callback("on_val_end", self.__send_metrics_on_val_end)
        else:
            model.add_callback("on_train_end", self.__send_metrics_on_train_end)

        model.add_callback("on_train_end", self.__send_artifacts_on_train_end)

    def __send_artifacts_on_train_end(self, trainer:DetectionTrainer) -> None:
        self.__pics.upload_model_version(self.__xml_config.project_name, trainer.best)
        self.__pics.experiment.log_parameters(trainer.args.__dict__)
        self.__pics.upload_artifact("confusion_matrix", f'{trainer.save_dir}/confusion_matrix.png')
        self.__pics.upload_artifact("confusion_matrix_normalized", f'{trainer.save_dir}/confusion_matrix_normalized.png')
        self.__pics.upload_artifact("F1_curve", f'{trainer.save_dir}/F1_curve.png')
        self.__pics.upload_artifact("labels", f'{trainer.save_dir}/labels.jpg')

    def __send_metrics_on_train_end(self, trainer:DetectionTrainer) -> None:
        experiment = self.__pics.experiment
        metrics_csv = pd.read_csv(trainer.csv)

        experiment.log(name=self.__PRECISION, data=metrics_csv[self.__PRECISION_METRIC].tolist(), type=LogType.LINE)
        experiment.log(name=self.__RECALL, data=metrics_csv[self.__RECALL_METRIC].tolist(), type=LogType.LINE)
        experiment.log(name=self.__MAP50, data=metrics_csv[self.__MAP50_METRIC].tolist(), type=LogType.LINE)
        experiment.log(name=self.__MAP50_95, data=metrics_csv[self.__MAP50_95_METRIC].tolist(), type=LogType.LINE)

        box_loss_data = {
            'train': metrics_csv["train/box_loss"].tolist(),
            'val': metrics_csv["val/box_loss"].tolist(),
        }
        experiment.log(name=self.__BOX_LOSS, data=box_loss_data, type=LogType.LINE)

        cls_loss_data = {
            'train': metrics_csv["train/cls_loss"].tolist(),
            'val': metrics_csv["val/cls_loss"].tolist(),
        }
        experiment.log(name=self.__CLS_LOSS, data=cls_loss_data, type=LogType.LINE)

        dfl_loss_data = {
            'train': metrics_csv["train/dfl_loss"].tolist(),
            'val': metrics_csv["val/dfl_loss"].tolist(),
        }
        experiment.log(name=self.__DFL_LOSS, data=dfl_loss_data, type=LogType.LINE)


    def __send_metrics_on_train_epoch_end(self, trainer:DetectionTrainer) -> None:
        experiment = self.__pics.experiment
        experiment.log(name=self.__PRECISION, data=float(trainer.metrics[self.__PRECISION_METRIC]), type=LogType.LINE)
        experiment.log(name=self.__RECALL, data=float(trainer.metrics[self.__RECALL_METRIC]), type=LogType.LINE)
        experiment.log(name=self.__MAP50, data=float(trainer.metrics[self.__MAP50_METRIC]), type=LogType.LINE)
        experiment.log(name=self.__MAP50_95, data=float(trainer.metrics[self.__MAP50_95_METRIC]), type=LogType.LINE)

        experiment.log(name=self.__BOX_LOSS, data={'train':[float(trainer.loss_items[0].item())]}, type=LogType.LINE)
        experiment.log(name=self.__CLS_LOSS, data={'train':[float(trainer.loss_items[1].item())]}, type=LogType.LINE)
        experiment.log(name=self.__DFL_LOSS, data={'train':[float(trainer.loss_items[2].item())]}, type=LogType.LINE)

    def __send_metrics_on_val_end(self, trainer:DetectionValidator) -> None:
        experiment = self.__pics.experiment
        experiment.log(name=self.__BOX_LOSS, data={'val': [float(trainer.loss[0].item())]}, type=LogType.LINE)
        experiment.log(name=self.__CLS_LOSS, data={'val': [float(trainer.loss[1].item())]}, type=LogType.LINE)
        experiment.log(name=self.__DFL_LOSS, data={'val': [float(trainer.loss[2].item())]}, type=LogType.LINE)