from ultralytics import YOLO

from src.dataset_manager.Picsellia import Picsellia
from src.file_reader.XmlExecReader import XmlExecReader
from src.file_reader.XmlPicselliaReader import XmlPicselliaReader
from src.yolo_training.DeviceDetector import DeviceDetector


def main() -> None:

    xml_picsellia_config: XmlPicselliaReader = XmlPicselliaReader(
        "config/picsellia.config.xml"
    )
    xml_exec_reader: XmlExecReader = XmlExecReader("config/config.exec.xml")

    models_path: str = f"./models/{xml_exec_reader.experiment_name}"

    pics: Picsellia = Picsellia(
        api_token=xml_picsellia_config.api_token,
        organization_name=xml_picsellia_config.organization_name,
        project_name=xml_picsellia_config.project_name,
        experiment_name=xml_exec_reader.experiment_name,
    )
    model_file_path: str = pics.download_model_version(models_path)

    device_type: str = DeviceDetector.get_device_type()
    print(f"Device type: {device_type}")

    model: YOLO = YOLO(model_file_path)

    if xml_exec_reader.inference_type == "video":
        print("start processing video")
        model(xml_exec_reader.source, device=device_type, show=True)
        print("end processing video")

    elif xml_exec_reader.inference_type == "images":
        print("start processing images")
        save_images_path: str = f"./exec/{xml_exec_reader.experiment_name}"
        model(
            source=xml_exec_reader.source,
            device=device_type,
            save=True,
            project=save_images_path,
        )
        print("end processing images")

    else:
        model(0, device=device_type, show=True)


if __name__ == "__main__":
    main()
