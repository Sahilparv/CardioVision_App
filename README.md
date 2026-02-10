❤️ CardioVision: AI-Driven ECG Classification

CardioVision is an end-to-end Deep Learning application designed for automated heart attack detection from ECG images. By leveraging a fine-tuned VGG16 Convolutional Neural Network (CNN), the system provides high-precision classification into four distinct cardiac health categories.

🚀 Live Demo

🛠️ Technical Features
Deep Learning Architecture: Utilizes a fine-tuned VGG16 model for sophisticated feature extraction from ECG scans.

Multi-Class Detection: Capable of identifying four specific classes: Myocardial Infarction (Heart Attack), Abnormal Heartbeat, History of MI, and Normal.

Automated Preprocessing: Implements real-time image resizing (224x224) and normalization (1/255) for consistent inference.

Web-Based Deployment: Built with Streamlit to offer a responsive and accessible user interface.

🧠 Model Methodology
The core of this project is a VGG16 model adapted for medical image classification.

Input: 224x224 RGB ECG images.

Normalization: Pixel values scaled to [0, 1].

Explainable AI: Future updates will include Grad-CAM integration to visualize which parts of the ECG the model is "looking at" to make a decision.

📂 Project Structure
app.py: The main Streamlit application script containing preprocessing and prediction logic.

ecg_heart_model.keras: The trained fine-tuned VGG16 model (managed via Git LFS).

requirements.txt: Configuration file for the cloud environment.

⚙️ Installation & Usage
Clone the repository:

Install dependencies:

Run the application: