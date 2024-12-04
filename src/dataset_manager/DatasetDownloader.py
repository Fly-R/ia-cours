from picsellia import Client
import os

from picsellia.types.enums import AnnotationFileType


class DatasetDownloader:

    def __init__(self, api_token:str, organization_name:str):
        self.__client = Client(
            api_token=api_token,
            organization_name=organization_name
        )

    def download(self, dataset_path:str, dataset_id:str, clear:bool=False):
        if clear and os.path.exists(dataset_path):
            os.rmdir(dataset_path)

        dataset = self.__client.get_dataset_version_by_id(dataset_id)
        dataset.download(dataset_path)
        #dataset.list_assets().download(dataset_path)
        dataset.export_annotation_file(AnnotationFileType.YOLO, dataset_path)