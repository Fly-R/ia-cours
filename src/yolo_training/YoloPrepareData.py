import os
import shutil
import zipfile

import yaml
from fsspec.compression import unzip
from picsellia import DatasetVersion, Project
from picsellia.types.enums import AnnotationFileType

from src.image_processor.ImageProcessor import ImageProcessor


class YoloPrepareData:

    @staticmethod
    def prepare_dataset(dataset: DatasetVersion, final_dataset_path: str) -> str:

        temp_path = "./temp_dataset"
        dataset.download("./temp")
        ImageProcessor("./temp", temp_path).process_folder()

        images = os.listdir(temp_path)
        item_per_dataset = (len(images) / 2) // 3

        train_images = images[:int(item_per_dataset)]
        valid_images = images[int(item_per_dataset):int(item_per_dataset*2)]
        test_images = images[int(item_per_dataset*2):]

        for split in ['train', 'val', 'test']:
            os.makedirs(f'{final_dataset_path}/images/{split}', exist_ok=True)
            os.makedirs(f'{final_dataset_path}/labels/{split}', exist_ok=True)

        YoloPrepareData.move_images(train_images, temp_path, final_dataset_path, "train")
        YoloPrepareData.move_images(valid_images, temp_path, final_dataset_path, "val")
        YoloPrepareData.move_images(test_images, temp_path, final_dataset_path, "test")


        dataset.export_annotation_file(AnnotationFileType.YOLO, temp_path)
        zip_path = f'{temp_path}/0192f6db-86b6-784c-80e6-163debb242d5/annotations/{dataset.id}_annotations.zip'
        shutil.unpack_archive(zip_path, temp_path)

        YoloPrepareData.move_annotations(train_images, temp_path, final_dataset_path, "train")
        YoloPrepareData.move_annotations(valid_images, temp_path, final_dataset_path, "val")
        YoloPrepareData.move_annotations(test_images, temp_path, final_dataset_path, "test")

        yaml_config = yaml.load(open(f'{temp_path}/data.yaml'), Loader=yaml.FullLoader)

        label_index=0
        names = ""
        for label in yaml_config['names']:
            names += f'     {label_index}:{label}\n'
            label_index += 1

        with open("./yolo_config.yaml", "w") as f:
            yaml_content = f"""
path : ./dataset
train: /images/train
val: images/val
test: images/test
names: \n{names}
"""
            f.write(yaml_content)

        shutil.copy(f'{temp_path}/data.yaml', f'{final_dataset_path}/data.yaml')

        return "./yolo_config.yml"

    @staticmethod
    def move_images(images, src, dest, split):
        for image in images:
            shutil.move(f'{src}/{image}', f'{dest}/images/{split}/{image}')

    @staticmethod
    def move_annotations(images, src, dest, split):
        for image in images:
            image_name = os.path.splitext(image)[0]
            annotation_path = f'{src}/{image_name}.txt'
            if os.path.exists(annotation_path):
                shutil.move(annotation_path, f'{dest}/labels/{split}/{image_name}.txt')
