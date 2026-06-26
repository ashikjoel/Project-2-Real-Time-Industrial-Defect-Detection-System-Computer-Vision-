import cv2
import time
from ultralytics import YOLO

# Load TensorRT model
model = YOLO(
    r"E:\Industrial_Defect _Detection_System\best_yolov8n.onnx"
)

# Webcam
cap = cv2.VideoCapture(0)

#for video file
#cap = cv2.VideoCapture(
 #   "/content/sample_video.mp4"
#)

prev_time = time.time()

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Inference
    results = model.predict(
        frame,
        verbose=False
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Industrial Defect Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()