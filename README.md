# 🩺 Breast Cancer Histopathology Classification

An AI-powered web application and PyTorch training pipeline for classifying breast cancer histopathology images using a fine-tuned ResNet-50 architecture.

## 🚀 Overview
This project uses Deep Learning (ResNet-50 pre-trained on ImageNet) to analyze breast histopathology images and predict whether they indicate **Benign** or **Malignant** tissue. It includes a complete training pipeline and a user-friendly frontend powered by Streamlit.

## 📊 Dataset
This project uses the **Breast Histopathology Images** dataset from Kaggle. 
* **Download here:** [Breast Histopathology Images on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
* **Setup:** Download the dataset, extract the archive, and place the contents into an `archive/` directory in the root of this project. (The `archive/` folder is intentionally ignored by git to save space).

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Madhu-014/Breast-Cancer-Histopathology.git
cd Breast-Cancer-Histopathology
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Usage

### Training the Model
To train the ResNet50 model from scratch on the dataset:
```bash
python src/train.py
```
*(This will save the trained weights to `models/resnet50.pth`)*

### Running the Web App
To start the Streamlit web interface and test your model with uploaded images:
```bash
streamlit run app.py
```
