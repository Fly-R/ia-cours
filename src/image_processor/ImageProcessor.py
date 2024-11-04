from datetime import datetime
import os

from src.image_processor.Image import Image


class ImageProcessor:

    def __init__(self, image_to_process_path: str, dataset_folder_path: str):
        self.__image_path = image_to_process_path
        self.__dataset_folder_path = dataset_folder_path

        self.__resized_size = 640

    def process_folder(self) -> None:
        process_output_folder = os.path.join(
            self.__dataset_folder_path, datetime.now().strftime("%Y%m%d%H%M%S")
        )
        os.mkdir(process_output_folder)
        for image_file in os.listdir(self.__image_path):
            image_path = os.path.join(self.__image_path, image_file)

            image = Image(image_path)
            image.resize(self.__resized_size)
            image.add_padding()

            processed_image_path = os.path.join(
                process_output_folder, image_file
            )
            image.save(processed_image_path)

            print(f"Processed image: {image_file}")
