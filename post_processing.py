import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("tooth_yolo_train/yolov8s_tooth/weights/best.pt")

# Inference on new image
results = model(r"C:\Users\prepared_dataset\val\images\cate10-00084_jpg.rf.85ec6a4ebe490441c7810204ac7349e8.jpg")[0]  # single image

# Extract boxes + classes
boxes = results.boxes.xyxy.cpu().numpy()
classes = results.boxes.cls.cpu().numpy()
scores = results.boxes.conf.cpu().numpy()

# Step 1: Separate upper vs lower (Y-axis)
median_y = np.median([(y1+y2)/2 for (x1,y1,x2,y2) in boxes])
upper_teeth = []
lower_teeth = []
for box, cls, conf in zip(boxes, classes, scores):
    x1,y1,x2,y2 = box
    cy = (y1+y2)/2
    if cy < median_y:
        upper_teeth.append((box, cls, conf))
    else:
        lower_teeth.append((box, cls, conf))

# Step 2: Split left vs right quadrants (X-midline)
def split_quadrants(teeth_list):
    teeth_sorted = sorted(teeth_list, key=lambda t: (t[0][0]+t[0][2])/2)  # sort by center x
    mid_x = np.median([(b[0][0]+b[0][2])/2 for b in teeth_sorted])
    left = [t for t in teeth_sorted if (t[0][0]+t[0][2])/2 < mid_x]
    right = [t for t in teeth_sorted if (t[0][0]+t[0][2])/2 >= mid_x]
    return left, right

upper_left, upper_right = split_quadrants(upper_teeth)
lower_left, lower_right = split_quadrants(lower_teeth)

# Step 3: Assign FDI numbers sequentially
def assign_fdi(teeth, quadrant):
    # quadrant: 1=upper-right, 2=upper-left, 3=lower-left, 4=lower-right
    fdi_numbers = []
    start = quadrant*10
    for i, (box, cls, conf) in enumerate(teeth, start=1):
        fdi_numbers.append((start+i, box, cls, conf))
    return fdi_numbers

upper_right_fdi = assign_fdi(upper_right, 1)
upper_left_fdi  = assign_fdi(upper_left, 2)
lower_left_fdi  = assign_fdi(lower_left, 3)
lower_right_fdi = assign_fdi(lower_right, 4)

# Final merged list
final_teeth = upper_right_fdi + upper_left_fdi + lower_left_fdi + lower_right_fdi

print("Assigned FDI numbering:", [f[0] for f in final_teeth])
