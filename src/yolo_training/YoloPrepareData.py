import os
import shutil

from picsellia import DatasetVersion
from picsellia.types.enums import AnnotationFileType

from src.yolo_training.YoloConfig import YoloConfig


class YoloPrepareData:

    def __init__(self, dataset: DatasetVersion, use_image_id=True):
        self.__dataset = dataset
        self.__use_image_id = use_image_id


    def prepare_new_dataset(self, dataset_path: str, ratios:list[float]=None, seed:int=42) -> YoloConfig:
        """
        Create 3 splits of data and annotations and download them according to the yolo folder structure.
        Generate the yolo config file
        :param dataset_path: Path to final location of the dataset
        :param ratios: Ratio of each split (train, val, test)
        :param seed: Seed use to split the initial dataset
        :return: Yolo config
        """
        if ratios is None:
            ratios = [0.6, 0.2, 0.2]

        yolo_config = YoloConfig(dataset_path)

        train_assets, test_assets, val_assets, count_train, count_test, count_val, labels = (
            self.__dataset.train_test_val_split(ratios=ratios, random_seed=seed))

        train_assets.download(yolo_config.images_train, use_id=self.__use_image_id)
        val_assets.download(yolo_config.images_val, use_id=self.__use_image_id)
        test_assets.download(yolo_config.images_test, use_id=self.__use_image_id)

        annotations_path = "./annotations"
        self.__download_annotations(annotations_path)

        YoloPrepareData.__move_annotations(os.listdir(yolo_config.images_train), annotations_path, yolo_config.labels_train)
        YoloPrepareData.__move_annotations(os.listdir(yolo_config.images_val), annotations_path, yolo_config.labels_val)
        YoloPrepareData.__move_annotations(os.listdir(yolo_config.images_test), annotations_path, yolo_config.labels_test)

        shutil.rmtree(annotations_path)

        yolo_config.generate_file(labels)

        return yolo_config


    @staticmethod
    def __move_annotations(images:list[str], annotations_src:str, annotations_dest:str) -> None:
        """
        Move annotation files to the destination folder according to image names
        :param images: List of images used to select which annotations to move
        :param annotations_src: Source folder of annotations
        :param annotations_dest: Destination folder for annotations
        :return:
        """
        os.makedirs(annotations_dest, exist_ok=True)
        for image in images:
            image_name = os.path.splitext(image)[0]
            annotation_path = f'{annotations_src}/{image_name}.txt'
            if os.path.exists(annotation_path):
                shutil.move(annotation_path, f'{annotations_dest}/{image_name}.txt')

    def __download_annotations(self, annotations_path:str) -> None:
        """
        Download annotations to the specified path and unzip them
        :param annotations_path: Final location for the annotations
        :return:
        """
        zip_path = self.__dataset.export_annotation_file(AnnotationFileType.YOLO, annotations_path, use_id=self.__use_image_id)
        shutil.unpack_archive(zip_path, annotations_path)