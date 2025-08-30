from ultralytics import YOLO



model = YOLO("yolov8s.pt")

# 6. Train Model
model.train(
    data="tooth_data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=2,
    project="tooth_yolo_train",
    name="yolov8s_tooth"
)



metrics = model.val()
print(metrics)

# Inference on Sample Images
results = model.predict(
    source=f"{DATASET_ROOT}/test/images",
    conf=0.25,
    save=True,
    save_txt=True,
    project="tooth_yolo_predictions",
    name="yolov8s_tooth"
)
print("Predictions saved!")