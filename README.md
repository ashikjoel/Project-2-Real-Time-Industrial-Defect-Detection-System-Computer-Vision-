# Real-Time Industrial Defect Detection

A real-time computer vision system for detecting surface defects on metal sheets using **YOLOv8**. The project focuses on achieving high detection accuracy while optimizing inference speed for edge deployment using **ONNX**, **TensorRT**, **FastAPI**, and **Docker**.

---

## 📌 Overview

Manual inspection of industrial products is time-consuming and prone to human error. This project automates the inspection process by detecting defects on metal surfaces using a deep learning-based object detection model.

The solution includes the complete machine learning pipeline—from dataset preparation and model training to model optimization, API deployment, monitoring, and containerization.

---

## ✨ Features

* Real-time metal surface defect detection
* YOLOv8-based object detection model
* Data augmentation using Albumentations
* ONNX and TensorRT model optimization
* FastAPI inference service
* Prometheus metrics for monitoring
* Dockerized deployment
* Ready for edge deployment

---

## 📂 Dataset

**Dataset:** NEU Metal Surface Defect Database

### Defect Classes

* Crazing
* Inclusion
* Patches
* Pitted Surface
* Rolled-in Scale
* Scratches

The dataset was converted into YOLO format and split into training and validation sets.

---

## 🛠️ Technology Stack

| Category             | Technologies                  |
| -------------------- | ----------------------------- |
| Programming Language | Python                        |
| Deep Learning        | PyTorch, YOLOv8 (Ultralytics) |
| Image Processing     | OpenCV                        |
| Data Augmentation    | Albumentations                |
| Model Optimization   | ONNX, TensorRT                |
| API Framework        | FastAPI                       |
| Monitoring           | Prometheus                    |
| Deployment           | Docker                        |

---

## 📁 Project Structure

```text
Industrial_Defect_Detection/
│
├── app/
│   ├── main.py
│   ├── routes/
│   └── services/
│
├── models/
│   ├── best.pt
│   ├── best.onnx
│   └── best.engine
│
├── dataset/
│   ├── train/
│   ├── valid/
│   └── data.yaml
│
├── reports/
│
├── Dockerfile
├── requirements.txt
├── README.md
└── app.py
```

---

## 🚀 Workflow

1. Prepare the NEU dataset.
2. Apply data augmentation using Albumentations.
3. Train the YOLOv8 model.
4. Evaluate model performance using mAP.
5. Convert the trained model to ONNX.
6. Optimize the model with TensorRT.
7. Build a FastAPI inference service.
8. Monitor performance with Prometheus.
9. Deploy using Docker.

---

## 📊 Model Performance

| Metric       | Value |
| ------------ | ----: |
| mAP@0.5      |  0.75 |
| mAP@0.5:0.95 |  0.39 |
| Precision    |  0.73 |
| Recall       |  0.70 |

> Update these values with your final evaluation results if they differ.

---

## ⚡ Performance Comparison

| Model    | Inference Time |    FPS |
| -------- | -------------: | -----: |
| PyTorch  |      204.82 ms |   4.88 |
| ONNX     |      171.02 ms |   5.85 |
| TensorRT |        4.64 ms | 215.49 |

---

## 📡 API

The project exposes a FastAPI endpoint for inference.

### Sample Request

```http
POST /predict
```

Upload an image to receive detected defect classes, confidence scores, and bounding box coordinates.

### Sample Response

```json
{
  "class": "Scratches",
  "confidence": 0.91,
  "bbox": [378, 0, 505, 632]
}
```

---

## 📈 Monitoring

Prometheus metrics include:

* API request count
* Inference latency
* Processing time
* Camera uptime

---

## 🐳 Docker Deployment

Build the Docker image:

```bash
docker build -t industrial-defect-detector .
```

Run the container:

```bash
docker run --rm -p 8000:8000 industrial-defect-detector
```

---

## 💡 Future Improvements

* Support additional industrial datasets
* Multi-camera inspection
* NVIDIA Jetson deployment
* Web-based dashboard
* Defect analytics and reporting
* Continuous model retraining

---

## 📚 Learning Outcomes

This project provided hands-on experience in:

* Object detection using YOLOv8
* Data augmentation techniques
* Model evaluation using mAP
* Model optimization with ONNX and TensorRT
* FastAPI backend development
* Docker containerization
* Monitoring with Prometheus
* End-to-end AI model deployment

---

## 🤝 Contributing

Contributions, suggestions, and feedback are welcome. Feel free to fork the repository, open an issue, or submit a pull request.

---


