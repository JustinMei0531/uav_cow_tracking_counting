import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# # Video path
# video_path = "./Cows-uav/Flight 3/DJI_0938.mp4"
# # Model path
# model_path = "./models/cows-yolov10.pt"

# # Load the YOLO model
# model = YOLO(model_path)

# # Open the video file
# videoCapture = cv2.VideoCapture(video_path)

# # Set confidence threshold
# confidence_threshold = 0.3

# # Initialize a set to store unique cow IDs
# unique_cows = set()

# # Start processing each frame in the video
# while True:
#     ret, frame = videoCapture.read()

#     # Check if the frame is read correctly
#     if not ret:
#         break

#     # Resize frame for uniform display size
#     frame = cv2.resize(frame, (1400, 800))
#     h, w, c = frame.shape

#     # # Use YOLO's track method for object detection and tracking
#     # results = model.track(source=frame, conf=confidence_threshold, persist=True, stream=True, tracker="bytetrack.yaml")

#     # # Create a blank frame to display only bounding boxes
#     # bbox_frame = frame.copy()

#     # # Iterate through the results to draw bounding boxes and track IDs
#     # for result in results:
#     #     if result.boxes is None:
#     #         continue

#     #     for box in result.boxes:
#     #         x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
#     #         track_id = box.id[0].item() if box.id is not None else None  # Get tracking ID

#     #         # Draw bounding box
#     #         cv2.rectangle(bbox_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     #         # Display track ID if available
#     #         if track_id is not None:
#     #             cv2.putText(bbox_frame, f"ID: {int(track_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     #             unique_cows.add(track_id)

#     # # Add the number of unique cows detected to the frame
#     # cv2.putText(bbox_frame, f"Total Unique Cows: {len(unique_cows)}", (10, h - 20), 
#     #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     # # Show the frame with bounding boxes and cow count
#     # cv2.imshow("Cows detection and tracking", bbox_frame)

#     results = model.track(frame, persist=True, conf=0.3, save=False, show=False)

#     # Visualize the results on the frame
#     annotated_frame = results[0].plot()

#     # Display the annotated frame
#     cv2.imshow("Cows tracking and counting", annotated_frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close windows
# videoCapture.release()
# cv2.destroyAllWindows()

# # Print the total number of unique cows counted
# print(f"Total unique cows detected: {len(unique_cows)}")


# Initialize DeepSORT with Re-ID
tracker = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    nn_budget=100,
)

# Model path
model_path = "./models/cows-yolov10.pt"

# Load the YOLO model
model = YOLO(model_path)

# Function to run detection and tracking on video
def detect_and_track(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    out = None

    if output_path:
        # Set up video writer to save output video
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    # Store unique cow IDs
    unique_cows = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect cows using YOLO
        results = model(frame, conf=0.3)  # Use confidence threshold of 0.5
        detections = []

        # Convert YOLO detections to DeepSORT format: [x1, y1, x2, y2, confidence]
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])
            if class_id == 0 and confidence > 0.5:  # Assuming '0' is the class ID for 'cow'
                detections.append([x1, y1, x2, y2, confidence])

        # Update tracker with detections
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw bounding boxes and track IDs on the frame
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Cow ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Add unique cow IDs to the set
            unique_cows.add(track_id)

        # Display total number of unique cows tracked
        cv2.putText(frame, f"Total Unique Cows: {len(unique_cows)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show frame
        cv2.imshow("Cow Detection and Tracking", frame)

        # Save the frame if output path is specified
        if out:
            out.write(frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

detect_and_track("./Cows-uav/Flight 3/DJI_0938.mp4")