import sys

from src.image_processor.ImageProcessor import ImageProcessor

if len(sys.argv) < 2:
    print("Usage: main.py <image_folder>")
    sys.exit(1)

image_to_process_path = sys.argv[1]

ImageProcessor(image_to_process_path, "./datasets").process_folder()
