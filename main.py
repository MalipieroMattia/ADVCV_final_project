from ultralytics import YOLO
from utils.data_loader import split_dataset

split_dataset(
    images_dir="raw_data/Data_YOLO",
    labels_dir="raw_data/Data_YOLO",
    output_dir="data/processed",
    test_size=0.2,
    val_size=0.2,
    is_already_split=True,
)


model = YOLO("yolov8n.yaml")

model.train(data="config.yaml", epochs=100)
