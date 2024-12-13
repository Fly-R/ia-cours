import os
from src.image_processor.Image import Image


class ImageProcessor:

    def __init__(self, image_to_process_path: str, dataset_folder_path: str=None):
        self.__image_path = image_to_process_path

        if dataset_folder_path is None:
            self.__dataset_folder_path = image_to_process_path
        else:
            self.__dataset_folder_path = dataset_folder_path

        self.__resized_size = 640

    def process_folder(self) -> None:
        os.makedirs(self.__dataset_folder_path, exist_ok=True)

        for image_file in os.listdir(self.__image_path):
            image_path = os.path.join(self.__image_path, image_file)

            image = Image(image_path)
            image.resize(self.__resized_size)
            image.add_padding()

            image.save(os.path.join(self.__dataset_folder_path, image_file))

            print(f"Processed image: {image_file}")
