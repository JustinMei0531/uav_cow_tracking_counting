from sahi.slicing import slice_coco

# Define paths
input_coco_file = "./output_coco.json"
image_dir = "./Cows-uav"
output_sliced_dir = "./sliced_dataset/images"

# Define sliced image size
sliced_image_width = 640
sliced_image_height = 640

slice_coco(
    coco_annotation_file_path=input_coco_file,
    image_dir=image_dir,
    output_coco_annotation_file_name="sliced_coco",
    output_dir=output_sliced_dir,
    slice_height=sliced_image_height,
    slice_width=sliced_image_width,
    overlap_width_ratio=0.2,
    overlap_height_ratio=0.2, 
    min_area_ratio=0.0,
    ignore_negative_samples=False
)


print(f"Sliced images and annotations saved to: {output_sliced_dir}")