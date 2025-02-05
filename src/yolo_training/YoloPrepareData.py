import os
import shutil

from picsellia import DatasetVersion
from picsellia.types.enums import AnnotationFileType

from src.yolo_training.YoloConfig import YoloConfig


class YoloPrepareData:

    def __init__(self, dataset: DatasetVersion, use_image_id=True):
        self.__dataset = dataset
        self.__use_image_id = use_image_id


    def prepare_new_dataset(self, final_dataset_path: str, ratios:list[float]=None, seed:int=42) -> YoloConfig:

        if ratios is None:
            ratios = [0.6, 0.2, 0.2]

        train_assets, test_assets, val_assets, count_train, count_test, count_val, labels = (
            self.__dataset.train_test_val_split(ratios=ratios, random_seed=seed))

        yolo_config = YoloConfig(final_dataset_path)

        train_assets.download(yolo_config.images_train, use_id=self.__use_image_id)
        val_assets.download(yolo_config.images_val, use_id=self.__use_image_id)
        test_assets.download(yolo_config.images_test, use_id=self.__use_image_id)

        annotations_path = "./annotations"
        self.__download_annotations(annotations_path)

        YoloPrepareData.__move_annotations(os.listdir(yolo_config.images_train), annotations_path, yolo_config.labels_train)
        YoloPrepareData.__move_annotations(os.listdir(yolo_config.images_val), annotations_path, yolo_config.labels_val)
        YoloPrepareData.__move_annotations(os.listdir(yolo_config.images_test), annotations_path, yolo_config.labels_test)

        yolo_config.generate_file(final_dataset_path, labels)

        return yolo_config


    @staticmethod
    def __move_annotations(images, src, dest):
        os.makedirs(dest, exist_ok=True)
        for image in images:
            image_name = os.path.splitext(image)[0]
            annotation_path = f'{src}/{image_name}.txt'
            if os.path.exists(annotation_path):
                shutil.move(annotation_path, f'{dest}/{image_name}.txt')

    def __download_annotations(self, annotations_path:str) -> None:
        self.__dataset.export_annotation_file(AnnotationFileType.YOLO, annotations_path, use_id=self.__use_image_id)
        zip_path = f'{annotations_path}/0192f6db-86b6-784c-80e6-163debb242d5/annotations/{self.__dataset.id}_annotations.zip'
        shutil.unpack_archive(zip_path, annotations_path)