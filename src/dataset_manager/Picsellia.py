
from picsellia import Client, Experiment, Project, DatasetVersion
from picsellia.types.enums import InferenceType, Framework


class Picsellia:

    def __init__(self, api_token:str, organization_name:str):
        self.__client = Client(
            api_token=api_token,
            organization_name=organization_name
        )

        self.__project = None
        self.__experiment = None
        self.__dataset = None

    def set_project(self, project_name:str) -> Project:
        if self.__project is None :
            self.__project = self.__client.get_project(project_name)
        return self.__project

    def set_experiment(self, project_name:str, experiment_name:str, force_create:bool=False) -> Experiment:
        if self.__experiment is None:
            project = self.set_project(project_name)
            try:
                self.__experiment = project.get_experiment(experiment_name)
            except Exception as e:
                if force_create:
                    self.__experiment = project.create_experiment(experiment_name)

        return self.__experiment

    def set_dataset(self, dataset_id:str) -> DatasetVersion:
        if self.__dataset is None:
            self.__dataset = self.__client.get_dataset_version_by_id(dataset_id)
        return self.__dataset

    def upload_model_version(self, model_name:str, model_weights_path:str) -> None:
        model = self.__client.get_model(name=model_name)
        export = self.__experiment.export_in_existing_model(model)
        export.update(type=InferenceType.OBJECT_DETECTION)
        export.update(framework=Framework.PYTORCH)
        try:
            export.store(name="model-latest", path=f'{model_weights_path}/best.pt')
        except Exception as e:
            print(e)

    def attach_current_dataset(self) -> None:
        if self.__dataset is None or self.__experiment is None:
            raise Exception("dataset or experiment is null")

        attached_datasets = self.__experiment.list_attached_dataset_versions()
        dataset_already_attached = False
        for dataset in attached_datasets:
            if dataset.id == self.__dataset.id:
                dataset_already_attached = True
                break

        if dataset_already_attached is False:
            self.__experiment.attach_dataset(self.__dataset)


