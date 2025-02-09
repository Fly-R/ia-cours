import xml.etree.ElementTree as ET


class XmlTrainReader:

    def __init__(self, path: str) -> None:
        self.__path: str = path
        self.__root: ET.Element = ET.parse(path).getroot()

    @property
    def dataset_path(self) -> str:
        return self.__root.find("dataset_path").text

    @property
    def dataset_version_id(self) -> str:
        return self.__root.find("dataset_version_id").text

    @property
    def experiment_name(self) -> str:
        return self.__root.find("experiment_name").text

    @property
    def send_metrics_on_epoch_end(self) -> bool:
        return self.__root.find("send_metrics_on_epoch_end").text == "True"
