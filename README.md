# uav_cow_tracking_counting

## Project Preview
This project is based on YOLOv10 and ByteTrack, trying to track and count cattle on Australian farms from drone-captured videos. Supported by high-resolution cameras, UAVs can fly over groups of cattle in order to capture images, making it possible to create computer vision systems to support the management of cattles and some other animals.

## Materials and methods

### Materials
The dataset used in the project comes from a DJI drone. The drone collected several wide-angle images (resolution 4000 * 3000) and videos at three different altitudes. Some samples are shown below:
![show01](./images/show01.jpg)
![show02](./images/show02.jpg)
![show03](./images/show03.jpg)
The three samples correspond to the images collected by the drone at different altitudes.

### Dataset creation
Generally speaking, it is more straightforward to use the YOLOmark tool to create a YOLO format dataset, but considering that YOLOmark is not very friendly to the annotation of small objects, another annotation tool is used here: Labelme (https://github.com/wkentaro/labelme). This tool was originally used to create semantic segmentation datasets, but it supports the annotation of small objects. The following is a sample image labeled by the labelme tool.
![show04](./images/show04.png)

After getting the annotated image, you also need to convert the generated annotation file (JSON format) into YOLO format. I implemented the relevant functions in the utils.py file.


## Research process and problems encountered

### 1. Object detection
I used two models, YOLOv10n and YOLOv10s, with the same training parameters, and trained them on GPU for 100 epochs. Some training parameters are as follows:
 **Parameter**   | **Default Value** |
|-----------------|------------------|
| `batch-size`    | 16               |
| `epochs`        | 100              |
| `img-size`      | 640              |
| `learning-rate` | 0.0001           |
| `momentum`      | 0.937            |
| `weight-decay`  | 0.0005           |
| `optimizer`     | Adam             |                                                           

Here are some prediction results: