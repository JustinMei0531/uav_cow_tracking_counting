import cv2
from ultralytics import YOLO

# Video path
video_path = "./Cows-uav/Flight 2/DJI_0004.mp4"
# Model path
model_path = "./models/cows-yolov10s.pt"

# Load the YOLO model
model = YOLO(model_path)

# Open the video file
videoCapture = cv2.VideoCapture(video_path)

# Set confidence threshold
confidence_threshold = 0.4

# Initialize a set to store unique cow IDs
unique_cows = set()

# Start processing each frame in the video
while True:
    ret, frame = videoCapture.read()

    # Check if the frame is read correctly
    if not ret:
        break

    # Resize frame for uniform display size
    frame = cv2.resize(frame, (1400, 800))
    h, w, c = frame.shape


    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=confidence_threshold)



    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("Cows Tracking", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
videoCapture.release()
cv2.destroyAllWindows()

# Print the total number of unique cows counted
print(f"Total unique cows detected: {len(unique_cows)}")
