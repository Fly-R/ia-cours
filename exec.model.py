import os

from ultralytics import YOLO

from src.dataset_manager.Picsellia import Picsellia
from src.file_reader.XmlExecReader import XmlExecReader
from src.file_reader.XmlPicselliaReader import XmlPicselliaReader

if __name__ == '__main__':

    xml_picsellia_config = XmlPicselliaReader("config/picsellia.config.xml")
    xml_exec_reader = XmlExecReader("config/config.exec.xml")

    models_path = f'./models/{xml_exec_reader.experiment_name}'

    model_file_path = f'{models_path}/best.pt'

    if os.path.exists(model_file_path) is False:
        pics = Picsellia(
            api_token=xml_picsellia_config.api_token,
            organization_name=xml_picsellia_config.organization_name,
            project_name=xml_picsellia_config.project_name,
            experiment_name=xml_exec_reader.experiment_name
        )

        pics.experiment.get_base_model_version().get_file("model-latest").download(target_path=models_path,
                                                                              force_replace=True)

    model = YOLO(model_file_path)

    if xml_exec_reader.inference_type == "video":
        print("start processing video")
        model(xml_exec_reader.source, show=True)
        print("end processing video")

    elif xml_exec_reader.inference_type == "images":
        print("start processing images")
        save_images_path = f'./exec/{xml_exec_reader.experiment_name}'
        model(source=xml_exec_reader.source, save=True,project=save_images_path)
        print("end processing images")

    else:
        model(0, show=True)





