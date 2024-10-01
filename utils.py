import os
import json
import cv2
import random
from tqdm import tqdm


def convert_annotation_format(json_file, output_path, image_width, image_height):
    with open(json_file, 'r') as f:
        data = json.load(f)

    yolo_labels = []
    
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        
        # Get bounding box
        xmin = min([p[0] for p in points])
        xmax = max([p[0] for p in points])
        ymin = min([p[1] for p in points])
        ymax = max([p[1] for p in points])

        # YOLO format: [class_id, x_center, y_center, width, height]
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        class_id = 0
        
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
        # Save YOLO labels
        yolo_file = os.path.join(output_path, os.path.splitext(os.path.basename(json_file))[0] + '.txt')
        with open(yolo_file, 'w') as f:
            f.write("\n".join(yolo_labels))
    
    # Save YOLO labels
    yolo_file = os.path.join(output_path, os.path.splitext(os.path.basename(json_file))[0] + '.txt')
    with open(yolo_file, 'w') as f:
        f.write("\n".join(yolo_labels))


def process_dataset(input_folder, output_folder, validation_ratio=0.1):
    os.makedirs(output_folder, exist_ok=True)
    
    # Create output folders for images and labels
    images_folder = os.path.join(output_folder, 'images')
    labels_folder = os.path.join(output_folder, 'labels')
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # Collect all JSON files and corresponding images
    all_files = [(f, os.path.splitext(f)[0] + '.jpg') for f in os.listdir(input_folder) if f.endswith('.json')]

    # Shuffle the dataset and split into train and validation sets
    random.shuffle(all_files)
    split_index = int(len(all_files) * (1 - validation_ratio))
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    def copy_files(file_list, dataset_type):
        dataset_image_folder = os.path.join(images_folder, dataset_type)
        dataset_label_folder = os.path.join(labels_folder, dataset_type)
        os.makedirs(dataset_image_folder, exist_ok=True)
        os.makedirs(dataset_label_folder, exist_ok=True)
        
        # Process each file in the dataset
        for json_file, img_file in tqdm(file_list, desc=f"Processing {dataset_type} files"):
            img_path = os.path.join(input_folder, img_file)
            json_path = os.path.join(input_folder, json_file)
            
            # Load the image to get its dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"Image {img_file} not found, skipping.")
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Copy image to the respective folder
            output_img_path = os.path.join(dataset_image_folder, img_file)
            cv2.imwrite(output_img_path, img)
            
            # Convert the LabelMe JSON to YOLO format and save it to the respective labels folder
            convert_annotation_format(json_path, dataset_label_folder, img_width, img_height)
    
    # Process train and validation datasets
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    
    # Create data.yaml file with correct paths
    yaml_content = f"""
    train: {images_folder}/train
    val: {images_folder}/val
    
    nc: 1  # Only one class (cow)
    names: ['cow']
    """

    if not os.path.exists(os.path.join(output_folder, "data.yaml")):
        with open(os.path.join(output_folder, 'data.yaml'), 'w') as f:
            f.write(yaml_content)


def slice_image(image_path, slice_width=800, slice_height=600, overlap=0):
    """
    Slice a large image into smaller patches and save them in the same directory with a modified name.

    Args:
        image_path (str): Path to the large image file.
        slice_width (int): Width of each slice.
        slice_height (int): Height of each slice.
        overlap (int): Number of pixels to overlap between slices.
    """
    # Get the directory and file name from the image path
    image_dir, image_filename = os.path.split(image_path)
    base_filename, file_ext = os.path.splitext(image_filename)

    # Read the large image
    img = cv2.imread(image_path)
    img_height, img_width, _ = img.shape

    # Slice the image into smaller patches
    patch_id = 0
    for y in range(0, img_height, slice_height - overlap):
        for x in range(0, img_width, slice_width - overlap):
            # Calculate the dimensions of the patch
            x_end = min(x + slice_width, img_width)
            y_end = min(y + slice_height, img_height)
            patch = img[y:y_end, x:x_end]

            # Name the patch based on the original filename and coordinates
            patch_filename = f"{base_filename}_patch_{x}_{y}{file_ext}"
            patch_path = os.path.join(image_dir, patch_filename)
            cv2.imwrite(patch_path, patch)

            print(f"Saved: {patch_path}")
            patch_id += 1

    print(f"Total {patch_id} patches saved in {image_dir}.")

def slice_multiple_images(input_dir, slice_width=800, slice_height=600, overlap=0):
    """
    Slice multiple large images in a directory into smaller patches and save them in the same directory.

    Args:
        input_dir (str): Directory containing the large images.
        slice_width (int): Width of each slice (default: 800).
        slice_height (int): Height of each slice (default: 600).
        overlap (int): Number of pixels to overlap between slices (default: 0).
    """
    # Iterate over all image files in the input directory
    for image_filename in os.listdir(input_dir):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_filename)

            print(f"Processing image: {image_filename}")
            # Slice the image into patches
            slice_image(image_path, slice_width, slice_height, overlap)


if __name__ == "__main__":

    # input_folder = "./Cows-uav/Flight 1"
    output_folder = './dataset'

    # List all images folders

    input_folders = os.listdir("./Cows-uav")
    
    for input_folder in input_folders:
        input_folder = os.path.join("./Cows-uav", input_folder)
        print("Processing folder {}".format(input_folder))
        process_dataset(input_folder, output_folder, 0.1)

    # images_folder = "./Cows-uav/Flight 2"

    # slice_multiple_images(images_folder)