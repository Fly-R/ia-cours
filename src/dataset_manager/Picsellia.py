from typing import Optional

from picsellia import Client, Experiment, DatasetVersion
from picsellia.types.enums import InferenceType, Framework


class Picsellia:

    def __init__(
        self,
        api_token: str,
        organization_name: str,
        project_name: str,
        experiment_name: str,
        dataset_id: str = None,
    ) -> None:
        self.__client: Client = Client(
            api_token=api_token, organization_name=organization_name
        )

        self.__experiment: Optional[Experiment] = None
        self.__dataset: Optional[DatasetVersion] = None

        project = self.__client.get_project(project_name)
        try:
            self.__experiment = project.get_experiment(experiment_name)
        except Exception:
            self.__experiment = project.create_experiment(experiment_name)

        if dataset_id is not None:
            self.__dataset = self.__client.get_dataset_version_by_id(dataset_id)

    @property
    def experiment(self) -> Experiment:
        """
        Returns the current experiment.
        Raises an exception if the experiment is not set.
        """
        if self.__experiment is None:
            raise Exception("Experiment not set")
        return self.__experiment

    @property
    def dataset(self) -> Optional[DatasetVersion]:
        """
        Returns the current dataset, or None if it hasn't been set.
        """
        return self.__dataset

    def upload_model_version(self, project_name: str, model_path: str) -> None:
        """
        Upload trained model to Picsellia and attach it to the current experiment as an object detection model using
        Pytorch named 'model-latest'
        :param project_name: Name of the global project
        :param model_path: Path to the model to upload
        :return:
        """
        model = self.__client.get_model(name=project_name)
        if self.__experiment is None:
            raise Exception("Experiment not set")
        export = self.__experiment.export_in_existing_model(model)
        self.__experiment.attach_model_version(export)
        export.update(type=InferenceType.OBJECT_DETECTION)
        export.update(framework=Framework.PYTORCH)
        try:
            export.store(name="model-latest", path=model_path)
        except Exception as e:
            print(e)

    def download_model_version(self, models_path: str) -> str:
        """
        Download the latest model version from the current experiment
        :param models_path: Path to save the model
        :return: Path to the downloaded model
        """
        if self.__experiment is None:
            raise Exception("Experiment not set")
        self.__experiment.get_base_model_version().get_file("model-latest").download(
            target_path=models_path, force_replace=True
        )
        return f"{models_path}/best.pt"

    def upload_artifact(self, artifact_name: str, artifact_path: str) -> None:
        """
        Upload file as an artifact to the current experiment
        :param artifact_name: Name of the artifact in the experiment
        :param artifact_path: Path to the file to upload
        :return:
        """
        if self.__experiment is None:
            raise Exception("Experiment not set")
        self.__experiment.store(artifact_name, artifact_path)

    def attach_current_dataset(self) -> None:
        """
        Attach the current dataset to the experiment if it isn't already attached.
        :return:
        """
        if self.__dataset is None or self.__experiment is None:
            raise Exception("dataset or experiment is null")

        attached_datasets = self.__experiment.list_attached_dataset_versions()
        dataset_already_attached: bool = False
        for dataset in attached_datasets:
            if dataset.id == self.__dataset.id:
                dataset_already_attached = True
                break

        if dataset_already_attached is False:
            self.__experiment.attach_dataset(
                name="dataset", dataset_version=self.__dataset
            )
