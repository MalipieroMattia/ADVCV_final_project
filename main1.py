from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="/work/yolo_dataset/dataset.yaml",
    epochs=5,
    imgsz=640,
    batch=8
)