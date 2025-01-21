import os

from ultralytics import YOLO

from src.file_reader.XmlExecReader import XmlExecReader

if __name__ == '__main__':
    xml_reader = XmlExecReader("config.exec.xml")

    model = YOLO(xml_reader.model_path)

    # Run batched inference on a list of images
    images = []
    for image in os.listdir(xml_reader.images_folder):
        images.append(f'{xml_reader.images_folder}/{image}')

    results = model(images)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen                to disk