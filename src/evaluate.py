import torch
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load test dataset
test_data = datasets.ImageFolder("data/test", transform=transform)

# ✅ Balanced subset: pick equal benign & malignant
benign_indices = [i for i, (_, label) in enumerate(test_data.samples) if label == 0]
malignant_indices = [i for i, (_, label) in enumerate(test_data.samples) if label == 1]

# pick up to 500 from each (or smaller if not enough samples)
n_samples = min(500, len(benign_indices), len(malignant_indices))
sampled_indices = benign_indices[:n_samples] + malignant_indices[:n_samples]

test_subset = Subset(test_data, sampled_indices)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

# Load model (must match training definition)
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(128, 1),
    torch.nn.Sigmoid()
)

# Load trained weights
model.load_state_dict(torch.load("../models/resnet50.pth", map_location=device))
model = model.to(device)
model.eval()

y_true, y_pred = [], []

# Evaluation loop
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)

        outputs = model(imgs)
        preds = (outputs > 0.5).int().cpu().numpy()

        y_true.extend(labels.cpu().numpy().flatten())
        y_pred.extend(preds.flatten())

# Reports
print("✅ Classification Report (balanced subset):\n",
      classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))
print("✅ Confusion Matrix (balanced subset):\n", confusion_matrix(y_true, y_pred))
