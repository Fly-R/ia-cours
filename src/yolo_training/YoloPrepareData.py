import shutil

from fsspec.compression import unzip
from picsellia import DatasetVersion, Project
from picsellia.types.enums import AnnotationFileType


class YoloPrepareData:

    @staticmethod
    def prepare_dataset(project:Project, dataset:DatasetVersion, final_dataset_path:str) -> str:

        temp_path = "./temp_dataset"
        dataset.download(temp_path)
        dataset.export_annotation_file(AnnotationFileType.YOLO, temp_path, use_id=True)

        shutil.unpack_archive(f'{temp_path}/{project.id}/annotations/{dataset.id}.zip', temp_path)

        return "./yolo_config.yml"
