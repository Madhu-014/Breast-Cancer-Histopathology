🧠 ResNet50 Image Classification (PyTorch)

This project implements a binary image classifier using ResNet-50 (a deep convolutional neural network pre-trained on ImageNet) in PyTorch.
It can be easily trained and tested locally on your machine using VS Code or any Python IDE.



🚀 Features

Transfer Learning using ResNet50 (pretrained on ImageNet)

Supports binary classification

Training & validation with progress tracking

Model checkpointing (resnet50.pth auto-saves after training)

Lightweight subset sampling for faster local runs

Works seamlessly in VS Code or Jupyter Notebook


🛠️ Setup Instructions
1. Clone the Repository
git clone https://github.com/<your-username>/resnet50-image-classification.git
cd resnet50-image-classification

2. Create and Activate a Virtual Environment
🧩 On Windows:
python -m venv venv
venv\Scripts\activate

🧩 On macOS/Linux:
python -m venv venv
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt



▶️ How to Run the Project
1. Start Training

Run the following command from your project root:

python train.py

2. Monitor Output

During training, you’ll see:

Epoch progress bars via tqdm

Loss and validation accuracy after each epoch

Model saving message upon completion:

💾 Model successfully saved at: ../models/resnet50.pth