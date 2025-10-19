import os
import shutil
import random

# Paths
raw_data_dir = "../archive"   # path to your extracted dataset
output_dir = "data"        # where we’ll create train/val/test

# Train/Val/Test split
split_ratio = [0.7, 0.15, 0.15]  # 70% train, 15% val, 15% test

classes = {"0": "benign", "1": "malignant"}

# Create output folders
for split in ["train", "val", "test"]:
    for cls in classes.values():
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# Collect all images
all_images = []
for patient_id in os.listdir(raw_data_dir):
    patient_folder = os.path.join(raw_data_dir, patient_id)
    if os.path.isdir(patient_folder):
        for label in classes.keys():
            label_folder = os.path.join(patient_folder, label)
            if os.path.exists(label_folder):
                for img_file in os.listdir(label_folder):
                    all_images.append((os.path.join(label_folder, img_file), classes[label]))

# Shuffle for randomness
random.shuffle(all_images)

# Split indices
n_total = len(all_images)
n_train = int(split_ratio[0] * n_total)
n_val = int(split_ratio[1] * n_total)

train_files = all_images[:n_train]
val_files = all_images[n_train:n_train+n_val]
test_files = all_images[n_train+n_val:]

# Helper to copy
def copy_files(file_list, split):
    for src, label in file_list:
        dst = os.path.join(output_dir, split, label, os.path.basename(src))
        shutil.copy(src, dst)

# Copy images
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("✅ Dataset reorganized into train/val/test folders!")
