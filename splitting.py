import os
from ultralytics import YOLO


import os
import random
import shutil
from pathlib import Path


#splitting the data 
DATASET_ROOT = "C:\\Users\\dataset"  
OUTPUT_ROOT = "C:\\Users\\prepared_dataset"

# Extensions
IMG_EXTS = [".jpg"]

# Creating output directories
for split in ["train", "val", "test"]:
    for folder in ["images", "labels"]:
        Path(f"{OUTPUT_ROOT}/{split}/{folder}").mkdir(parents=True, exist_ok=True)

# Collect all image files
images = [f for f in os.listdir(f"{DATASET_ROOT}/images") if Path(f).suffix.lower() in IMG_EXTS]
random.shuffle(images)

# Split sizes
n_total = len(images)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

train_files = images[:n_train]
val_files = images[n_train:n_train+n_val]
test_files = images[n_train+n_val:]

def move_files(files, split):
    for img_file in files:
        label_file = img_file.rsplit(".", 1)[0] + ".txt"
        # Move image
        shutil.copy(f"{DATASET_ROOT}/images/{img_file}", f"{OUTPUT_ROOT}/{split}/images/{img_file}")
        # Move label if exists
        if os.path.exists(f"{DATASET_ROOT}/labels/{label_file}"):
            shutil.copy(f"{DATASET_ROOT}/labels/{label_file}", f"{OUTPUT_ROOT}/{split}/labels/{label_file}")

# Perform moves
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print(f"Split done! Total: {n_total}, Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")



