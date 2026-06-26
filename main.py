from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
    Histogram,
    Counter,
    Gauge
)
import cv2
import numpy as np
import onnxruntime as ort
import time

app = FastAPI()

# Load ONNX model once at startup
model_path = "best_yolov8n.onnx"
session = ort.InferenceSession(model_path)

# NEW: Prometheus Histogram to track inference latency
INFERENCE_LATENCY = Histogram(
    "model_inference_seconds",
    "Time taken by the ONNX model for inference"
)

# Counts total inference requests
INFERENCE_REQUESTS = Counter(
    "inference_requests_total",
    "Total number of inference requests"
)

CAMERA_UP = Gauge(
    "camera_up",
    "Camera or image source status"
)

@app.get("/")
def home():
    return {"message": "Defect Detection API Running"}


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    # Read uploaded image
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
       CAMERA_UP.set(0)
       return {"error": "Invalid image"}

    CAMERA_UP.set(1)

    # Preprocess
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    img_input = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)

    # -----------------------------
    # NEW: Start timer
    # -----------------------------
    INFERENCE_REQUESTS.inc()
    start_time = time.time()

    # Inference
    outputs = session.run(
        None,
        {"images": img_input}
    )

    # -----------------------------
    # NEW: Stop timer
    # -----------------------------
    end_time = time.time()

    latency = end_time - start_time

    # Store latency in Prometheus
    INFERENCE_LATENCY.observe(latency)

    print(f"\nInference Time: {latency:.4f} seconds")

    output = outputs[0][0]

    # Find highest confidence detection
    max_conf = 0
    best_idx = -1

    for i in range(output.shape[1]):

        scores = output[4:, i]
        conf = np.max(scores)

        if conf > max_conf:
            max_conf = conf
            best_idx = i

    if best_idx == -1:
        return {"message": "No defect detected"}

    best_detection = output[:, best_idx]

    x_center = float(best_detection[0])
    y_center = float(best_detection[1])
    width = float(best_detection[2])
    height = float(best_detection[3])

    x1 = max(0, x_center - width / 2)
    y1 = max(0, y_center - height / 2)
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    class_id = int(np.argmax(best_detection[4:]))

    # Simulated PLC broadcast
    print("\n=== PLC DATA ===")
    print(f"Register 0 (x1): {round(x1, 2)}")
    print(f"Register 1 (y1): {round(y1, 2)}")
    print(f"Register 2 (x2): {round(x2, 2)}")
    print(f"Register 3 (y2): {round(y2, 2)}")
    print("================")

    return {
        "class_id": class_id,
        "confidence": round(float(max_conf), 4),
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2)
    }