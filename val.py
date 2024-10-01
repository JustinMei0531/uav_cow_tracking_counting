from ultralytics import YOLO
import os
import cv2
import random

'''
Take a prediction on a static image.
'''


# Path to the trained model
trained_model_path = "./models/cows-yolov10.pt"

images_path = "./dataset/images/val/"

# Load the trained model
model = YOLO(trained_model_path)

image_list = os.listdir(images_path)

# Pick up a rondom image
# image = cv2.imread(image_list[random.randint(0, len(image_list))], cv2.IMREAD_COLOR)


# Perform prediction on the image
results = model.predict(source="F:\\Python\\projects\\YOLO-cow\\dataset\\images\\val\\DJI_0007_patch_1600_1200.jpg", save=False, conf=0.4)

model.val("F:\\Python\\projects\\YOLO-cow\\dataset\\data.yaml")

# Show results
results[0].show()
