from ultralytics import YOLO

class YoloTrainer:

    def __init__(self, config_file:str):
        self.__model = YOLO(config_file)


    def train(self):
        pass
