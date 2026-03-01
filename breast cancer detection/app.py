import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)

# CONFIGURATION
# Use the correct .keras extension and the 3-class model path
MODEL_PATH = "model/breast_cancer_model.keras"
UPLOAD_FOLDER = "static/uploads"
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup
model = tf.keras.models.load_model(MODEL_PATH)

def get_tta_prediction(filepath, tta_steps=5):
    """Applying the TTA logic we discussed to the web app."""
    img = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # NO DIVIDING BY 255

    predictions = [model.predict(img_array, verbose=0)]
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True, rotation_range=10, zoom_range=0.1
    )

    for _ in range(tta_steps):
        aug_img = next(datagen.flow(img_array, batch_size=1))
        predictions.append(model.predict(aug_img, verbose=0))

    mean_pred = np.mean(predictions, axis=0)[0]
    class_idx = np.argmax(mean_pred)
    confidence = mean_pred[class_idx] * 100
    return CLASS_NAMES[class_idx], round(float(confidence), 2)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400
    
    file = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Use the TTA prediction function we built
    class_name, confidence = get_tta_prediction(filepath)

    # Map the class names to your specific messages
    if class_name == "Malignant":
        result_message = "Malignant (CANCEROUS)"
        status_color = "danger" # Red for UI
    elif class_name == "Normal":
        result_message = "NORMAL"
        status_color = "success" # Green for UI
    else: # Benign
        result_message = "BENIGN (Non-Cancerous)"
        status_color = "info" # Blue for UI

    return render_template(
        "index.html",
        result=result_message,
        confidence=confidence,
        image_path=filepath,
        status_color=status_color
    )

if __name__ == "__main__":
    app.run(debug=True)