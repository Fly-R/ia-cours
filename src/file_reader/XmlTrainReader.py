import xml.etree.ElementTree as ET

class XmlTrainReader:

    def __init__(self, path:str):
        self.__path = path
        self.__root = ET.parse(path).getroot()

    @property
    def api_token(self):
        return self.__root.find("api_token").text

    @property
    def organization_name(self):
        return self.__root.find("organization_name").text

    @property
    def dataset_path(self):
        return self.__root.find("dataset_path").text

    @property
    def dataset_version_id(self):
        return  self.__root.find("dataset_version_id").text

    @property
    def project_name(self):
        return self.__root.find("project_name").text

    @property
    def experiment_name(self):
        return self.__root.find("experiment_name").text