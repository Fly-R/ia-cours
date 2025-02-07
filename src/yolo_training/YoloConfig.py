import yaml
from picsellia import Label


class YoloConfig:

    def __init__(self, dataset_path:str):
        self.__path = dataset_path
        self.__save_path = f'{dataset_path}/yolo_config.yaml'

    @property
    def images_train(self) -> str:
        return f'{self.__path}/images/train'

    @property
    def images_val(self) -> str:
        return f'{self.__path}/images/val'

    @property
    def images_test(self) -> str:
        return f'{self.__path}/images/test'

    @property
    def labels_train(self) -> str:
        return f'{self.__path}/labels/train'

    @property
    def labels_val(self) -> str:
        return f'{self.__path}/labels/val'

    @property
    def labels_test(self) -> str:
        return f'{self.__path}/labels/test'

    @property
    def file_path(self) -> str:
        return self.__save_path

    def generate_file(self, labels:list[Label]) -> None:
        """
        Generate a yolo config file with the given labels and dataset path.
        Save it to the dataset path
        :param labels: List of labels to use
        :return:
        """
        yaml_config = {
            "path": self.__path,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {}
        }

        index = 0
        for label in labels:
            yaml_config["names"][index] = label.name
            index += 1

        with open(self.__save_path, 'w') as file:
            yaml.dump(yaml_config, file)