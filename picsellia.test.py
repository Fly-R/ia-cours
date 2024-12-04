import sys

from picsellia import ModelVersion

from src.dataset_manager.Picsellia import Picsellia
from src.file_reader.XmlReader import XmlReader
from src.yolo_training.YoloPrepareData import YoloPrepareData

''' DatasetDownloader(
            api_token=xml_reader.api_token,
            organization_name=xml_reader.organization_name
        ).download(
            dataset_path=xml_reader.dataset_path,
            dataset_id=xml_reader.dataset_id,
            clear=True
        )'''

if len(sys.argv) >= 2:
    if sys.argv[1] == "clear":
        xml_reader = XmlReader("./config.xml")
        pics = Picsellia(
            api_token=xml_reader.api_token,
            organization_name=xml_reader.organization_name)
        exp = pics.get_experiment(
            project_name=xml_reader.project_name,
            experiment_name=xml_reader.experiment_name,
            force_create=True
        )

        dataset = pics.get_dataset(xml_reader.dataset_id)

        Picsellia.attach_dataset(experiment=exp, dataset=dataset)

        yolo_config_file = YoloPrepareData.prepare_dataset(dataset, "./dataset")

