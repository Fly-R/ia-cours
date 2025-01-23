
from picsellia import Client, Experiment, Project, DatasetVersion
from picsellia.types.enums import InferenceType, Framework


class Picsellia:

    def __init__(self, api_token:str, organization_name:str):
        self.__client = Client(
            api_token=api_token,
            organization_name=organization_name
        )

    def get_project(self, project_name:str) -> Project:
        return self.__client.get_project(project_name)

    def get_experiment(self, project_name:str, experiment_name:str, force_create:bool=False) -> Experiment:
        project = self.get_project(project_name)
        try:
            return project.get_experiment(experiment_name)
        except:
            if force_create:
                return project.create_experiment(experiment_name)

    def get_dataset(self, dataset_id:str) -> DatasetVersion:
        return self.__client.get_dataset_version_by_id(dataset_id)

    def upload_model_version(self, model_name:str, model_weights_path:str) -> None:
        try:
            model = self.__client.get_model(name=f'{model_name}_model')
        except Exception as e:
            model = self.__client.create_model(
                name=f'{model_name}_model',
                type=InferenceType.OBJECT_DETECTION,
                framework=Framework.PYTORCH
            )

        model_version = model.create_version(
            type=InferenceType.OBJECT_DETECTION,
            framework=Framework.PYTORCH
        )
        try:
            model_version.store(name="model-latest", path=f'{model_weights_path}/best.pt')
        except Exception as e:
            print(e)

    @staticmethod
    def attach_dataset(experiment:Experiment, dataset:DatasetVersion) -> None:
        attached_datasets = experiment.list_attached_dataset_versions()
        dataset_already_attached = False
        for dataset in attached_datasets:
            if dataset.id == dataset.id:
                dataset_already_attached = True
                break

        if dataset_already_attached is False:
            experiment.attach_dataset(dataset)


