# src/infer.py
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 🔹 Define preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(model_path="models/resnet50.pth"):
    """Load trained ResNet50 model"""
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def predict(image_path, model):
    """Predict benign/malignant for a given image"""
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # batch dimension
    with torch.no_grad():
        output = model(img_tensor)
        prob = output.item()
        pred = "Malignant" if prob > 0.5 else "Benign"
    return pred, prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--model", default="models/resnet50.pth", help="Path to model file")
    args = parser.parse_args()

    model = load_model(args.model)
    label, confidence = predict(args.image, model)
    print(f"🔍 Prediction: {label} (confidence: {confidence:.4f})")
