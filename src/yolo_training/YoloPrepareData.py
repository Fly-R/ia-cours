import os
import shutil
import yaml

from picsellia import DatasetVersion
from picsellia.types.enums import AnnotationFileType


class YoloPrepareData:

    def __init__(self, dataset: DatasetVersion, use_image_id=True):
        self.dataset = dataset
        self.__use_image_id = use_image_id


    def prepare_new_dataset(self, final_dataset_path: str, ratios:list[float]=None, seed:int=42) -> str:

        if ratios is None:
            ratios = [0.6, 0.2, 0.2]

        train_assets, test_assets, val_assets, count_train, count_test, count_val, labels = (
            self.dataset.train_test_val_split(ratios=ratios, random_seed=seed))

        training_split = ['train', 'val', 'test']

        train_assets.download(f"{final_dataset_path}/images/{training_split[0]}", use_id=self.__use_image_id)
        val_assets.download(f"{final_dataset_path}/images/{training_split[1]}", use_id=self.__use_image_id)
        test_assets.download(f"{final_dataset_path}/images/{training_split[2]}", use_id=self.__use_image_id)

        annotations_path = "./annotations"
        self.__download_annotations(annotations_path)

        for split in training_split:
            os.makedirs(f'{final_dataset_path}/labels/{split}', exist_ok=True)
            YoloPrepareData.__move_annotations(os.listdir(f"{final_dataset_path}/images/{split}"), annotations_path, final_dataset_path, split)

        yaml_config = {
            "path": final_dataset_path,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {}
        }

        index = 0
        for label in labels:
            yaml_config["names"][index] = label.name
            index += 1

        yaml_config_path = f'{final_dataset_path}/yolo_config.yaml'
        with open(yaml_config_path, 'w') as file:
            yaml.dump(yaml_config, file)

        return yaml_config_path


    @staticmethod
    def __move_annotations(images, src, dest, split):
        for image in images:
            image_name = os.path.splitext(image)[0]
            annotation_path = f'{src}/{image_name}.txt'
            if os.path.exists(annotation_path):
                shutil.move(annotation_path, f'{dest}/labels/{split}/{image_name}.txt')

    def __download_annotations(self, annotations_path:str) -> None:
        self.dataset.export_annotation_file(AnnotationFileType.YOLO, annotations_path, use_id=self.__use_image_id)
        zip_path = f'{annotations_path}/0192f6db-86b6-784c-80e6-163debb242d5/annotations/{self.dataset.id}_annotations.zip'
        shutil.unpack_archive(zip_path, annotations_path)