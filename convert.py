import os
from sahi.utils.coco import Coco

def convert_sliced_coco_to_yolo(sliced_coco_file, sliced_images_dir, output_dir):
    """
    Converts sliced COCO annotations to YOLO format.

    Args:
        sliced_coco_file (str): Path to the sliced COCO annotations file.
        sliced_images_dir (str): Directory containing the sliced images.
        output_dir (str): Directory to save YOLO format annotations and images.
    """
    # Load the sliced COCO annotations
    coco = Coco.from_coco_dict_or_path(sliced_coco_file)

    # Set the `image_dir` attribute to the sliced images directory
    coco.image_dir = sliced_images_dir

    # Export converted YOLO format dataset
    coco.export_as_yolov5(
        output_dir=output_dir,
        train_split_rate=0.85,  # 85% train, 15% val split
        disable_symlink=True
    )
    print(f"YOLO formatted dataset saved in: {output_dir}")

# Paths
sliced_coco_file = "./sliced_dataset/images/sliced_coco_coco.json"
sliced_images_dir = "./sliced_dataset/images"
yolo_output_dir = "./yolo_sliced_dataset/"

# Convert sliced COCO to YOLO format
convert_sliced_coco_to_yolo(sliced_coco_file, sliced_images_dir, yolo_output_dir)
