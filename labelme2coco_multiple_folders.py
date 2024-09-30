import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image

# Define COCO JSON structure
coco_format = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Category mapping (update this list based on your classes in LabelMe)
category_mapping = {
    "cow": 1
}

# Category index for COCO format
category_set = {value: key for key, value in category_mapping.items()}

def create_coco_image_entry(image_id, file_name, width, height):
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    }

def create_coco_annotation_entry(annotation_id, image_id, category_id, bbox, area, segmentation):
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": area,
        "segmentation": segmentation,
        "iscrowd": 0
    }

def labelme_to_coco_multiple_folders(parent_dir, output_file, category_mapping):
    """
    Converts multiple folders containing LabelMe JSON files into a single COCO format JSON file.

    Args:
        parent_dir (str): Parent directory containing multiple folders of LabelMe annotations.
        output_file (str): Output file name for the COCO JSON file.
        category_mapping (dict): Dictionary to map LabelMe labels to COCO category IDs.
    """
    image_id = 1
    annotation_id = 1

    # Create categories based on the mapping
    for label, category_id in category_mapping.items():
        coco_format["categories"].append({
            "id": category_id,
            "name": label,
            "supercategory": "none"
        })

    # Traverse each subfolder in the parent directory
    for root, dirs, files in os.walk(parent_dir):
        for file_name in tqdm(files, desc=f"Processing LabelMe files in {root}"):
            if not file_name.endswith(".json"):
                continue

            json_path = os.path.join(root, file_name)

            # Load LabelMe JSON data
            with open(json_path, 'r') as f:
                labelme_data = json.load(f)

            # Extract image details
            image_file_name = labelme_data['imagePath']
            image_path = os.path.join(root, image_file_name)
            if not os.path.exists(image_path):
                print(f"Image file '{image_file_name}' not found in '{root}'. Skipping.")
                continue

            # Get image size
            image = Image.open(image_path)
            width, height = image.size

            # Create COCO image entry
            coco_image = create_coco_image_entry(image_id, image_file_name, width, height)
            coco_format["images"].append(coco_image)

            # Process each shape (polygon/rectangle) in the LabelMe JSON
            for shape in labelme_data['shapes']:
                label = shape['label']
                if label not in category_mapping:
                    print(f"Label '{label}' not in category mapping. Skipping.")
                    continue

                # Get bounding box coordinates
                points = np.array(shape['points'])
                xmin = np.min(points[:, 0])
                ymin = np.min(points[:, 1])
                xmax = np.max(points[:, 0])
                ymax = np.max(points[:, 1])

                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]  # COCO format: [x, y, width, height]
                area = bbox[2] * bbox[3]  # width * height

                # Create segmentation (polygon) if the shape is a polygon
                segmentation = []
                if shape['shape_type'] == "polygon":
                    segmentation = [list(np.array(points).flatten())]

                # Create COCO annotation entry
                category_id = category_mapping[label]
                coco_annotation = create_coco_annotation_entry(annotation_id, image_id, category_id, bbox, area, segmentation)
                coco_format["annotations"].append(coco_annotation)

                # Increment annotation ID
                annotation_id += 1

            # Increment image ID
            image_id += 1

    # Save COCO formatted data to output file
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)
    print(f"Successfully created COCO JSON: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert multiple folders of LabelMe annotations to COCO format.")
    parser.add_argument('--parent_dir', type=str, required=True, help="Parent directory containing multiple folders of LabelMe JSON files and images.")
    parser.add_argument('--output_file', type=str, required=True, help="Output path for COCO JSON file.")
    args = parser.parse_args()

    # Update the category mapping based on your LabelMe labels
    label_to_category = {
        "cow": 1  # Example: if 'cow' is a label in your LabelMe annotations
    }

    # Run conversion for multiple folders
    labelme_to_coco_multiple_folders(args.parent_dir, args.output_file, label_to_category)
