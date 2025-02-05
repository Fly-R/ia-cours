import os

from ultralytics.models.yolo.detect import DetectionPredictor

from src.dataset_manager.Picsellia import Picsellia


class YoloPredictCallbacks:

    def __init__(self, pics: Picsellia):
        self.__pics = pics

    def on_predict_batch_end(self, predictor: DetectionPredictor):
        dataset = self.__pics.dataset
        experiment = self.__pics.experiment
        for item in predictor.results:
            detected_class_item_count = item.boxes.cls.shape[0]
            boxes = []
            for class_index in range(detected_class_item_count):
                class_id = int(item.boxes.cls[class_index])
                box = [int(i) for i in item.boxes.xywh[class_index].tolist()]
                box[0] = box[0] - box[2] // 2
                box[1] = box[1] - box[3] // 2
                label = dataset.get_label(item.names[class_id])
                conf = float(item.boxes.conf[class_index])
                box.append(label)
                box.append(conf)
                boxes.append(tuple(box))
            img = os.path.splitext(os.path.basename(item.path))[0]
            asset = dataset.find_asset(id=img)
            experiment.add_evaluation(asset, rectangles=boxes)

            print(f'Evaluated asset {img} uploaded')