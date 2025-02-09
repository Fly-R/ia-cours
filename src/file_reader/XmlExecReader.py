import xml.etree.ElementTree as ET


class XmlExecReader:

    def __init__(self, path: str) -> None:
        self.__path: str = path
        self.__root: ET.Element = ET.parse(path).getroot()

    @property
    def experiment_name(self) -> str:
        return self.__root.find("experiment_name").text

    @property
    def source(self) -> str:
        return self.__root.find("source").text

    @property
    def inference_type(self) -> str:
        return self.__root.find("inference_type").text
