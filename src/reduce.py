import os
import random
import shutil

# Path to your local dataset folder
source_folder = "data/val/benign"  # adjust if needed

# Path to Google Drive folder (after installing Google Drive for Desktop)
dest_folder = "temp/val/benign"  # adjust if needed

# Make sure destination exists
os.makedirs(dest_folder, exist_ok=True)

# List all images in source
all_images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Pick 1000 random images
sampled_images = random.sample(all_images, 1000)

# Copy files
for img in sampled_images:
    shutil.copy(os.path.join(source_folder, img), os.path.join(dest_folder, img))

print("✅ Copied 1000 images to", dest_folder)
