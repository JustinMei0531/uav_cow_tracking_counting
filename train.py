from ultralytics import YOLO

if __name__ == "__main__":
    model_path = "./models/yolov10s.pt"
    print("Loading YOLO model...")
    model = YOLO(model_path)

    print("Starting training...")
    model.train(data="F:\\Python\\uav_cow_tracking_counting\\yolo_sliced_dataset\\data.yml", epochs=50, device="cuda", imgsz=640, batch=32)

    print("Training completed. Saving the model...")
    model.save("./models/cows-yolov10.pt")
    print("Model saved successfully.")