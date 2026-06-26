import onnxruntime as ort

model_path = r"E:\Industrial_Defect _Detection_System\best_yolov8n.onnx"

session = ort.InferenceSession(model_path)

print("Model loaded successfully!")

print("Inputs:")
for inp in session.get_inputs():
    print(inp.name, inp.shape)

print("\nOutputs:")
for out in session.get_outputs():
    print(out.name, out.shape)