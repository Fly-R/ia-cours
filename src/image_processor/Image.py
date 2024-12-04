import numpy as np
from PIL import Image as PILImage


class Image:

    def __init__(self, image_path: str):
        self.__image_path = image_path
        self.__array = np.array(PILImage.open(image_path))

        self.__padding_color = (114, 114, 144)

    def resize(self, new_size: int) -> None:
        """
        Resize the image to a new size while keeping the aspect ratio

        Args:
            new_size: The new size to resize width and height to.
        """
        width = self.width
        height = self.height
        if width > height:
            height = int(height // (width / new_size))
            width = new_size
        else:
            width = int(width // (height / new_size))
            height = new_size

        np.resize(self.__array, (height, width))

    def add_padding(self) -> None:
        """
        Add padding to the image to make it square.
        Bottom padding if the width > height
        Right padding if the height > width
        """
        width = self.width
        height = self.height
        if width == height:
            return

        add_padding_to_bottom = width > height
        if add_padding_to_bottom:
            padding = ((0, width - height), (0, 0), (0, 0))
        else:
            padding = ((0, 0), (0, height - width), (0, 0))

        if self.is_grayscale:
            padding_values = self.__padding_color[0]
            padding = (padding[0], padding[1])
        else:
            padding_values = (
                (self.__padding_color[0], self.__padding_color[0]),
                (self.__padding_color[1], self.__padding_color[1]),
                (self.__padding_color[2], self.__padding_color[2]),
            )

        self.__array = np.pad(
            self.__array, padding, "constant", constant_values=padding_values
        )

    def save(self, image_path: str) -> None:
        PILImage.fromarray(self.__array).save(image_path)

    @property
    def width(self) -> int:
        return self.__array.shape[1]

    @property
    def height(self) -> int:
        return self.__array.shape[0]

    @property
    def is_grayscale(self) -> bool:
        return len(self.__array.shape) == 2
