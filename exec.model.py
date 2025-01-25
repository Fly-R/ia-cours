import os

from ultralytics import YOLO

from src.dataset_manager.Picsellia import Picsellia
from src.file_reader.XmlExecReader import XmlExecReader
from src.file_reader.XmlPicselliaReader import XmlPicselliaReader

if __name__ == '__main__':

    xml_picsellia_config = XmlPicselliaReader("picsellia.config.xml")
    xml_exec_reader = XmlExecReader("config.exec.xml")

    pics = Picsellia(
        api_token=xml_picsellia_config.api_token,
        organization_name=xml_picsellia_config.organization_name)

    experiment = pics.set_experiment(xml_picsellia_config.project_name, xml_exec_reader.experiment_name)

    model_path = f'./models/{xml_exec_reader.experiment_name}'

    experiment.get_base_model_version().get_file("model-latest").download(target_path=model_path, force_replace=True)

    model = YOLO(f'{model_path}/best.pt')

    # Run batched inference on a list of images
    images = []
    for image in os.listdir(xml_exec_reader.images_folder):
        images.append(f'{xml_exec_reader.images_folder}/{image}')

    results = model(images)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen                to disk