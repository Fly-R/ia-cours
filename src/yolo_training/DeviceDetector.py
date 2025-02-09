import torch


class DeviceDetector:

    @staticmethod
    def get_device_type() -> str:
        """
        Detects the available device type for PyTorch operations.

        Returns:
            A string representing the device type:
                - "mps" if Apple's Metal Performance Shaders are available,
                - "cuda" if an NVIDIA GPU with CUDA is available,
                - "cpu" otherwise.
        """
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
