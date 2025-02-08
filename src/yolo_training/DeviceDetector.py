import torch

class DeviceDetector:

    @staticmethod
    def get_device_type() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"