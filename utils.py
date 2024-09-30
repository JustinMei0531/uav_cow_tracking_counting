# 1. 把yolov5格式转换成coco格式标签；
# 2. 切片图片和coco标签；
# 3. 把切片出来的coco标签转换回yolov5标签格式

import os
import numpy as np
import cv2
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.slicing import slice_coco
from sahi.utils.file import save_json
from tqdm import tqdm
import random
import json


def convert_annotation_format(json_file, output_path, image_width, image_height):
    with open(json_file, 'r') as f:
        data = json.load(f)

    yolo_labels = []

    # Read json file
    
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




def convert2coco(img_path,h,w,yololabel):

    coco = Coco()
    maps = {
        0 : "cow"
    }
    coco.add_category(CocoCategory(id=0, name='cow')) # 两个类别

    coco_image = CocoImage(file_name=img_path, height=h, width=w)
    
    for label in yololabel:

        coco_image.add_annotation(
        CocoAnnotation(
            bbox=[label[1], label[2], label[3], label[4]],
            category_id=int(label[0]),
            category_name=maps[label[0]]
        )
        )
    
    coco.add_image(coco_image)
    coco_json = coco.json
    save_json(coco_json, "coco_dataset.json")
    return coco_json

def convert2xywh(l, h, w):
    """
    Converts YOLO format (class_id, x_center, y_center, width, height) to
    (class_id, top_left_x, top_left_y, width, height).
    
    Args:
        l (numpy.ndarray): Array of YOLO labels (class_id, x_center, y_center, width, height).
        h (int): Image height.
        w (int): Image width.
    
    Returns:
        numpy.ndarray: Converted labels in the format (class_id, top_left_x, top_left_y, width, height).
    """
    # Ensure `l` is at least a 2D array
    if len(l.shape) == 1:
        l = l.reshape(-1, 5)  # Reshape to (1, 5) if it's a single row

    # Handle empty arrays
    if l.shape[0] == 0:
        return np.array([])

    # Create a new array to store converted labels
    new_l = np.zeros_like(l)

    # Convert YOLO format to top-left corner based bounding box (left_x, top_y, width, height)
    l[:, 1] = l[:, 1] * w  # x_center * image width
    l[:, 3] = l[:, 3] * w  # width * image width
    l[:, 2] = l[:, 2] * h  # y_center * image height
    l[:, 4] = l[:, 4] * h  # height * image height

    # Assign converted values to the new array
    new_l[:, 0] = l[:, 0]  # class_id
    new_l[:, 1] = l[:, 1] - l[:, 3] / 2  # top-left x = x_center - (width / 2)
    new_l[:, 2] = l[:, 2] - l[:, 4] / 2  # top-left y = y_center - (height / 2)
    new_l[:, 3] = l[:, 3]  # width
    new_l[:, 4] = l[:, 4]  # height

    return new_l

def slice_img(save_img_dir):
    
    coco_dict, coco_path = slice_coco(
                coco_annotation_file_path="coco_dataset.json",
                image_dir='',
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                output_dir = save_img_dir,
                output_coco_annotation_file_name = 'sliced',
                min_area_ratio = 0.2,
                ignore_negative_samples = True
            )
    return  

def convert2yolov5(coco_dir, save_img_dir, save_label_dir):
    """
    Converts the sliced COCO annotations back to YOLO format.
    
    Args:
        coco_dir (str): Path to the sliced COCO annotations file.
        save_img_dir (str): Path to the directory containing the sliced images.
        save_label_dir (str): Directory where YOLO labels will be saved.
    """
    # Load COCO object
    coco = Coco.from_coco_dict_or_path(coco_dir, save_img_dir)
    
    # Validate image paths before exporting
    for coco_image in coco.images:
        coco_image_path = coco_image.file_name
        full_image_path = os.path.join(save_img_dir, coco_image_path)
        if not os.path.exists(full_image_path):
            print(f"Warning: File not found -> {full_image_path}")
    
    # Export YOLOv5 formatted dataset
    coco.export_as_yolov5(
        output_dir=save_label_dir,
        disable_symlink=True
    )

if __name__ == '__main__':
    img_dir = f'dataset/images/train/'
    anno_dir = f'dataset/labels/train/'
    save_img_dir = 'dataset/sliced_images/' 
    save_label_dir = 'dataset/sliced_labels/'
    os.makedirs(save_img_dir,exist_ok=True)
    os.makedirs(save_label_dir,exist_ok=True)
    labels = os.listdir(anno_dir)
    for label in labels:
        if 'old' not in label:
            try:
                os.remove('coco_dataset.json') # 删除中间文件
                os.remove(save_img_dir+'sliced_coco.json')
            except:
                pass
            l = np.loadtxt(anno_dir+label,delimiter=' ') # class cx xy w h
            img_path = img_dir+label.replace('txt','jpg')
            img = cv2.imread(img_path)
            h,w,_ = img.shape
            new_l = convert2xywh(l,h,w)
            coco_json = convert2coco(img_path,h,w,new_l)
            slice_img(save_img_dir)  # 切分图片并保存
            convert2yolov5(save_img_dir+'sliced_coco.json', save_img_dir, save_label_dir) # 把切分完的coco标签转换回yolo格式并保存
            
# if __name__ == "__main__":
#     # input_folder = "./Cows-uav/Flight 1"
#     output_folder = './dataset'

#     # List all images folders

#     input_folders = os.listdir("./Cows-uav")
    
#     for input_folder in input_folders:
#         input_folder = os.path.join("./Cows-uav", input_folder)
#         print("Processing folder {}".format(input_folder))
#         process_dataset(input_folder, output_folder, 0.1)

    
            