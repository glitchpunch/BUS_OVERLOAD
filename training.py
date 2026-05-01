from ultralytics import YOLO
import torch

# ── Auto detect device ──
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# ── Train model ──
def train(model_name, data_yaml, run_name):
    model = YOLO(model_name)
    
    model.train(
        data=data_yaml,
        epochs=50,
        batch=16,
        imgsz=640,
        device=device,
        name=run_name,
        project="runs/train",
        plots=True
    )
    
    return f"runs/train/{run_name}/weights/best.pt"


# ── Validate model ──
def validate(weights, data_yaml):
    model = YOLO(weights)
    metrics = model.val(data=data_yaml)
    
    print("mAP50:", metrics.box.map50)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)


# ── Export to ONNX ──
def export(weights):
    model = YOLO(weights)
    model.export(format="onnx")
    print("Exported to ONNX")


# ── Run pipeline ──
if __name__ == "__main__":
    
    DATA = "data.yaml"   # your dataset file
    
    # Train YOLOv8n
    w1 = train("yolov8n.pt", DATA, "bus_n")
    validate(w1, DATA)
    export(w1)

    # Train YOLOv8s (optional)
    w2 = train("yolov8s.pt", DATA, "bus_s")
    validate(w2, DATA)
    export(w2)
