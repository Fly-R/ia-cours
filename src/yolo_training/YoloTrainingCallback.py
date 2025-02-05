from picsellia import Experiment
from picsellia.types.enums import LogType
from ultralytics.models.yolo.detect import DetectionTrainer

from src.dataset_manager.Picsellia import Picsellia
from src.file_reader.XmlPicselliaReader import XmlPicselliaReader


class YoloTrainingCallback:

    def __init__(self, pics:Picsellia, xml_config:XmlPicselliaReader):
        self.__pics = pics
        self.__xml_config = xml_config

    def on_train_end(self, trainer:DetectionTrainer):
        self.__pics.upload_model_version(self.__xml_config.project_name, trainer.best)
        self.__pics.experiment.log_parameters(trainer.args.__dict__)
        self.__pics.upload_artifact("confusion_matrix", f'{trainer.save_dir}/confusion_matrix.png')
        self.__pics.upload_artifact("confusion_matrix_normalized", f'{trainer.save_dir}/confusion_matrix_normalized.png')
        self.__pics.upload_artifact("F1_curve", f'{trainer.save_dir}/F1_curve.png')
        self.__pics.upload_artifact("labels", f'{trainer.save_dir}/labels.jpg')

    def on_train_epoch_end(self, trainer:DetectionTrainer):
        experiment = self.__pics.experiment
        experiment.log(name="precision", data=trainer.metrics["metrics/precision(B)"], type=LogType.LINE)
        experiment.log(name="recall", data=trainer.metrics["metrics/recall(B)"], type=LogType.LINE)
        experiment.log(name="mAP50", data=trainer.metrics["metrics/mAP50(B)"], type=LogType.LINE)
        experiment.log(name="mAP50-95", data=trainer.metrics["metrics/mAP50-95(B)"], type=LogType.LINE)

        experiment.log(name="lr-pg0", data=trainer.lr["lr/pg0"].item(), type=LogType.LINE)
        experiment.log(name="lr-pg1", data=trainer.lr["lr/pg1"].item(), type=LogType.LINE)
        experiment.log(name="lr-pg2", data=trainer.lr["lr/pg2"].item(), type=LogType.LINE)
