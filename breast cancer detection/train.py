import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight

# --- Setup & Hyperparameters ---
IMG_SIZE = 224
BATCH_SIZE = 32
DATASET_PATH = "dataset"

# 1. Load Data with proper seed for reproducibility
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# 2. Calculate Class Weights (Crucial for the 3-class imbalance)
y_train = np.concatenate([y for x, y in train_ds], axis=0)
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(cw))

# Prefetch for performance
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# 3. Build Model (EfficientNetB0)
def create_model():
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  # Start frozen
    
    model = models.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        # NO Rescaling layer - EfficientNet has it built-in!
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5), # Higher dropout to fight overfitting on 600 images
        layers.Dense(128, activation="relu"),
        layers.Dense(3, activation="softmax") # 3 Classes: Normal, Benign, Malignant
    ])
    return model, base_model

model, base_model = create_model()

# 4. STAGE 1: Warm-up (Training only the Head)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(patience=2, factor=0.5, monitor='val_loss'),
    ModelCheckpoint("model/breast_cancer_model.keras", save_best_only=True)
]

print("\n--- Starting Stage 1: Head Training ---")
model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks, class_weight=class_weights)

# 5. STAGE 2: Fine-Tuning (The Critical Part)
# We only unfreeze the LAST few layers to prevent the "Catastrophic Forgetting" you saw
base_model.trainable = True
for layer in base_model.layers[:-15]: # Only unfreeze the last 15 layers
    layer.trainable = False

# We use a MICRO learning rate. 1e-5 or 1e-6. 
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5), 
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n--- Starting Stage 2: Surgical Fine-Tuning ---")
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks, class_weight=class_weights)

print("\nTraining Complete. Model saved to model/breast_cancer_model.keras")