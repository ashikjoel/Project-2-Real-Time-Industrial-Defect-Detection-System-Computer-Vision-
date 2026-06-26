import cv2
import numpy as np
import onnxruntime as ort

# Load model
model_path = r"E:\Industrial_Defect _Detection_System\best_yolov8n.onnx"
session = ort.InferenceSession(model_path)

# Load image
image_path = r"C:\Users\ashik\Downloads\inclusion_256.jpg"

img = cv2.imread(image_path)

# Save original dimensions
orig_h, orig_w = img.shape[:2]

# Preprocess
img_resized = cv2.resize(img, (640, 640))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

img_input = img_rgb.astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2, 0, 1))
img_input = np.expand_dims(img_input, axis=0)

# Run inference
outputs = session.run(
    None,
    {"images": img_input}
)

output = outputs[0][0]  # shape: (10, 8400)

print("Output Shape:", output.shape)

# Find highest confidence detection
max_conf = 0
best_idx = -1

for i in range(output.shape[1]):

    scores = output[4:, i]

    conf = np.max(scores)

    if conf > max_conf:
        max_conf = conf
        best_idx = i

print("\nBest Detection Index:", best_idx)
print("Highest Confidence:", max_conf)

best_detection = output[:, best_idx]

print("\nBest Detection Values:")
print(best_detection)

# Extract bounding box
x_center = best_detection[0]
y_center = best_detection[1]
width = best_detection[2]
height = best_detection[3]

# Convert YOLO format to corner coordinates
x1 = x_center - width / 2
y1 = y_center - height / 2
x2 = x_center + width / 2
y2 = y_center + height / 2

print("\nBounding Box:")
print(f"x1={x1:.2f}")
print(f"y1={y1:.2f}")
print(f"x2={x2:.2f}")
print(f"y2={y2:.2f}")

# Draw box on image
scale_x = orig_w / 640
scale_y = orig_h / 640

x1 = int(x1 * scale_x)
y1 = int(y1 * scale_y)
x2 = int(x2 * scale_x)
y2 = int(y2 * scale_y)

cv2.rectangle(
    img,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    2
)

cv2.imshow("Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()