import xml.etree.ElementTree as ET

class XmlExecReader:

    def __init__(self, path:str):
        self.__path = path
        self.__root = ET.parse(path).getroot()

    @property
    def model_path(self):
        return self.__root.find("model_path").text

    @property
    def images_folder(self):
        return self.__root.find("image_folder").text