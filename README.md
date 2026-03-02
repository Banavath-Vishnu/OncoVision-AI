# 🧠 OncoVision AI

### AI-Powered Breast Ultrasound Classification System

OncoVision AI is an end-to-end deep learning system that classifies breast ultrasound images into **Benign**, **Malignant**, and **Normal** categories.

⚠️ **Disclaimer:** This project is developed for educational and research purposes only. It is not intended for clinical or diagnostic use.

---

## 🚀 Project Overview

Early detection plays a critical role in breast cancer survival rates. Many AI startups today are building medical imaging systems to assist radiologists with faster and more consistent screening.

OncoVision AI explores the engineering foundations behind such systems by:

* Training a CNN-based image classifier
* Handling dataset imbalance
* Improving inference robustness
* Deploying a real-time prediction web application

---

# 📸 Application Preview
## 🔹 Main Interface

![Main UI](assets/ui_home.png)

## 🔹 Prediction Output

![Prediction Result](assets/prediction_result.png)
---

# 📊 Dataset

This project uses the **Breast Ultrasound Images Dataset (BUSI)**.

Kaggle Link:
[https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

---

## 📥 How to Download Dataset (Manual Method)

1. Open the Kaggle link above
2. Sign in to your Kaggle account
3. Click **Download**
4. Extract the ZIP file
5. Rename the extracted folder to:

```
dataset
```

6. Place it in your project root directory

Final structure should look like:

```
dataset/
    benign/
    malignant/
    normal/
```

---

# 🏗 Project Structure

```
OncoVision-AI/
│
├── dataset/
├── model/
│   └── breast_cancer_model.keras
├── static/
│   └── uploads/
├── templates/
│   └── index.html
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

---

# 🧠 Model Architecture

* EfficientNetB0 backbone
* Global Average Pooling
* Batch Normalization
* Dropout (0.5)
* Dense layer (ReLU)
* Softmax output (3 classes)

---

# ⚙️ Training Strategy

### Stage 1 – Head Training

* Backbone frozen
* Train classification layers
* Learning rate: 1e-3

### Stage 2 – Fine-Tuning

* Unfreeze last 15 layers
* Learning rate: 1e-5
* Prevent catastrophic forgetting

### Additional Enhancements

* Class-weight balancing
* Early stopping
* Learning rate scheduling
* Model checkpointing
* Test-Time Augmentation (TTA)

---

# 🔍 Inference Features

* Test-Time Augmentation
* Confidence score visualization
* Confidence-based rejection mechanism
* Real-time Flask deployment
* Clean Tailwind UI

---

# 🖥 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/onco-vision-ai.git
cd onco-vision-ai
```

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Train the Model

```bash
python train.py
```

Model will be saved to:

```
model/breast_cancer_model.keras
```

---

# 🌐 Run the Application

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000/
```

---

# 🐍 Python Version

This project was developed and tested using:

**Python 3.10**

It is recommended to use Python 3.10 to avoid compatibility issues with TensorFlow.

---

# 📦 Requirements

Main dependencies:

* TensorFlow
* Flask
* NumPy
* Scikit-learn
* Pillow
* SciPy

See `requirements.txt` for full list.

---

# 🧠 Key Learnings

* Transfer learning for medical imaging
* Fine-tuning deep CNNs safely
* Handling small and imbalanced datasets
* Managing softmax overconfidence
* Building safer AI inference pipelines
* Deploying ML models as web applications

---

# 🚀 Future Improvements

* Add Ultrasound vs Non-Ultrasound classifier
* Grad-CAM explainability
* Model calibration (temperature scaling)
* Vision Transformer experimentation
* Cloud deployment

---

# 👨‍💻 Author

**Banavath Vishnu**
Machine Learning & AI Enthusiast

LinkedIn:
[https://www.linkedin.com/in/vishnu-banavath/](https://www.linkedin.com/in/vishnu-banavath/)

---

⭐ If you found OncoVision AI useful, consider giving it a star!
