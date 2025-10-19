import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import random

# ==============================================================
# 🔹 Configuration for VS Code Environment
# ==============================================================

# Dataset directories (relative to this script)
data_dir = "data"  # should contain "train" and "val" subfolders
model_path = "../models/resnet50.pth"

# Create output model directory if not present
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ==============================================================
# 🔹 Data Transformations
# ==============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================================================
# 🔹 Load Datasets
# ==============================================================

train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
val_data   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)

# ⚡ Limit dataset size (useful for testing/training on local machines)
def get_subset(dataset, max_samples=1000):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return Subset(dataset, indices[:max_samples])

train_data = get_subset(train_data, max_samples=1500)
val_data   = get_subset(val_data, max_samples=600)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=16)

# ==============================================================
# 🔹 Model Setup: ResNet50
# ==============================================================

model = models.resnet50(pretrained=True)

# Freeze feature extractor layers
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier head for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

# ==============================================================
# 🔹 Load Checkpoint (if available)
# ==============================================================

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded existing model weights from: {model_path}")

# ==============================================================
# 🔹 Training Loop
# ==============================================================

num_epochs = 5  # Adjust based on your hardware
print(f"📚 Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)

    # Validation phase
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(imgs)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] ➜ Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

# ==============================================================
# 🔹 Save Final Model
# ==============================================================

torch.save(model.state_dict(), model_path)
print(f"💾 Model successfully saved at: {model_path}")
