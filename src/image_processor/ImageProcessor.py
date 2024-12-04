from datetime import datetime
import os

from src.image_processor.Image import Image


class ImageProcessor:

    def __init__(self, image_to_process_path: str, dataset_folder_path: str):
        self.__image_path = image_to_process_path
        self.__dataset_folder_path = dataset_folder_path

        self.__resized_size = 640

    def process_folder(self) -> None:
        os.mkdir(self.__dataset_folder_path)
        for image_file in os.listdir(self.__image_path):
            image_path = os.path.join(self.__image_path, image_file)

            image = Image(image_path)
            image.resize(self.__resized_size)
            image.add_padding()

            processed_image_path = os.path.join(
                self.__dataset_folder_path, image_file
            )
            image.save(processed_image_path)

            print(f"Processed image: {image_file}")
