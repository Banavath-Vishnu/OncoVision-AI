# OncoVision-AI
## 🩺 Breast Cancer Detection using Deep Learning

An end-to-end deep learning system for classifying breast ultrasound images into **Benign**, **Malignant**, and **Normal** categories using Transfer Learning with EfficientNetB0.

⚠️ **Disclaimer:** This project is for educational and research purposes only. Not intended for clinical use.

---

## 🚀 Project Overview

Early detection of breast cancer significantly improves survival rates. This project explores how deep learning can assist in ultrasound image classification for screening support.

The system includes:

* Transfer learning using EfficientNetB0
* Two-stage fine-tuning strategy
* Class imbalance handling
* Test-Time Augmentation (TTA)
* Confidence-based rejection mechanism
* Flask-based web deployment

---

## 🖼 Application Preview

> 📌 Add screenshots inside an `assets/` folder in your repository.

### 🔹 Main Interface

```
assets/ui_home.png
```

### 🔹 Prediction Output

```
assets/prediction_result.png
```

Then embed them like this:

```markdown
![Main UI](assets/ui_home.png)
![Prediction Result](assets/prediction_result.png)
```

---

## 📊 Dataset

This project uses the **Breast Ultrasound Images Dataset (BUSI)**.

Kaggle Link:
[https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

---

## 📥 How to Download the Dataset (Manual Method)

1. Open the Kaggle dataset link above.
2. Sign in to your Kaggle account.
3. Click the **Download** button.
4. Extract the downloaded `.zip` file.
5. Rename the extracted folder to:

```
dataset
```

6. Place it inside your project root directory.

Final folder structure should look like:

```
dataset/
    benign/
    malignant/
    normal/
```

---

## 🏗 Project Structure

```
project-folder/
│
├── dataset/
│   ├── benign/
│   ├── malignant/
│   └── normal/
│
├── model/
│   └── breast_cancer_model.keras
│
├── static/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

* Backbone: EfficientNetB0 (ImageNet pretrained)
* Global Average Pooling
* Batch Normalization
* Dropout (0.5)
* Dense Layer (128 units, ReLU)
* Output Layer (3-class Softmax)

---

## ⚙️ Training Strategy

### Stage 1: Head Training

* Backbone frozen
* Train classification layers
* Learning rate: 1e-3

### Stage 2: Fine-Tuning

* Unfreeze last 15 layers
* Micro learning rate: 1e-5
* Prevent catastrophic forgetting

### Additional Techniques

* Class-weight balancing
* Early stopping
* Learning rate reduction
* Model checkpointing
* Test-Time Augmentation (TTA)

---

## 🖥 Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/breast-cancer-detection.git
cd breast-cancer-detection
```

### 2️⃣ Create Virtual Environment

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

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

```bash
python train.py
```

The trained model will be saved to:

```
model/breast_cancer_model.keras
```

---

## 🌐 Running the Web Application

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000/
```

Upload an ultrasound image to test predictions.

---

## 🔍 Inference Features

* Test-Time Augmentation (TTA)
* Confidence score visualization
* Confidence-based rejection for uncertain inputs
* Clean Tailwind CSS UI
* Real-time image classification

---

## 🧠 Key Learnings

* Transfer learning for medical imaging
* Fine-tuning CNNs without catastrophic forgetting
* Handling dataset imbalance
* Managing model overconfidence
* Deploying deep learning models as web apps

---

## 📦 Requirements

Main dependencies:

* TensorFlow
* Flask
* NumPy
* Scikit-learn
* Pillow
* SciPy

See `requirements.txt` for full list.

---

## 🚀 Future Improvements

* Add Ultrasound vs Non-Ultrasound detector
* Implement Grad-CAM for explainability
* Add temperature scaling for calibration
* Deploy on cloud (AWS/GCP)
* Convert backend to FastAPI
* Experiment with Vision Transformers

---

## 👨‍💻 Author

**Banavath Vishnu**
Machine Learning & AI Enthusiast

LinkedIn:
[https://www.linkedin.com/in/vishnu-banavath/](https://www.linkedin.com/in/vishnu-banavath/)

---

⭐ If you found this project useful, consider giving it a star!
