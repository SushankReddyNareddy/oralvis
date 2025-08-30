# FDI Numbering with YOLOv8

This project trains a YOLOv8s model to detect and classify teeth using the FDI numbering system (32 classes).

---



## ⚙️ Environment Setup
- Python 3.10+
- Install requirements:
  ```bash
  pip install ultralytics
  ```

---

## 🚀 Training Command
```bash
yolo task=detect mode=train model=yolov8s.pt data=data/tooth_data.yaml epochs=100 imgsz=640 batch=16
```

---

## 📊 Results
- Precision: 90.3%
- Recall: 88.2%
- mAP@50: 93.3%
- mAP@50–95: 65.1%


---

## 📌 Notes
- Dataset size: 497 images (train: 397, val: 50, test: 50)
- Label format: YOLO (class_id x_center y_center width height)
- Classes: 32 FDI teeth numbers (11–48)
