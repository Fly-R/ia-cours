import xml.etree.ElementTree as ET


class XmlPicselliaReader:

    def __init__(self, path: str):
        self.__path = path
        self.__root = ET.parse(path).getroot()

    @property
    def api_token(self) -> str:
        return self.__root.find("api_token").text

    @property
    def organization_name(self) -> str:
        return self.__root.find("organization_name").text

    @property
    def project_name(self) -> str:
        return self.__root.find("project_name").text
