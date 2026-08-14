import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2
import tensorflow as tf

# -------------------------
# Dataset path
# -------------------------
data_dir = "augmented2"

image_paths, labels = [], []
for class_name in os.listdir(data_dir):
    class_folder = os.path.join(data_dir, class_name)
    if os.path.isdir(class_folder):
        for img_file in os.listdir(class_folder):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_folder, img_file))
                labels.append(class_name)

df = pd.DataFrame({"filename": image_paths, "class": labels})
print(f"✅ Total Images: {len(df)}")

# -------------------------
# Step 2: Train-Test Split
# -------------------------
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["class"], random_state=42)

# -------------------------
# Step 3: ImageDataGenerator (no extra augmentation)
# -------------------------
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = datagen.flow_from_dataframe(
    train_df, x_col="filename", y_col="class",
    target_size=(224, 224), class_mode="categorical",
    batch_size=16, shuffle=True
)

test_data = datagen.flow_from_dataframe(
    test_df, x_col="filename", y_col="class",
    target_size=(224, 224), class_mode="categorical",
    batch_size=16, shuffle=False
)

num_classes = len(train_data.class_indices)

# --- IMPORTANT: create ordered label list (index -> class name) ---
# class_indices maps class_name -> index, so invert it and build ordered list
inv_map = {v: k for k, v in train_data.class_indices.items()}
ordered_labels = [inv_map[i] for i in range(num_classes)]

# -------------------------
# Step 4: Model Setup
# -------------------------
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 1️⃣ Phase 1: Freeze entire base model
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile with slightly higher LR initially for warm-up
model.compile(optimizer=Adam(learning_rate=3e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.3, min_lr=1e-6, verbose=1)

print("\n🔹 Phase 1: Training top layers only...")
history1 = model.fit(train_data, validation_data=test_data,
                     epochs=10, callbacks=[early_stop, reduce_lr], verbose=1)

# 2️⃣ Phase 2: Gentle fine-tuning
# ✅ Only unfreeze top 20 layers, keep BatchNorm frozen
for layer in base_model.layers:
    if 'batch_normalization' in layer.name:
        layer.trainable = False
    else:
        layer.trainable = False

for layer in base_model.layers[-20:]:
    layer.trainable = True

# ✅ Much smaller learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=5e-6),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

print("\n🔹 Phase 2: Fine-tuning EfficientNet backbone (gentle)...")
history2 = model.fit(train_data, validation_data=test_data,
                     epochs=5, callbacks=[early_stop, reduce_lr], verbose=1)

# -------------------------
# Step 5: Evaluation
# -------------------------
# Print official evaluated loss & accuracy on the test set
eval_loss, eval_acc = model.evaluate(test_data, verbose=1)
print(f"\nModel.evaluate -> Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}")

# Predictions & classification report using ordered_labels
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=ordered_labels))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=ordered_labels,
            yticklabels=ordered_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# -------------------------
# Step 6: New Image Prediction (fixed)
# -------------------------
new_image_path = "download.jpg"

# Load and preprocess image exactly like training data
img = cv2.imread(new_image_path)
if img is None:
    raise ValueError(f"Image not found at path: {new_image_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img.astype("float32")
img = preprocess_input(img)  # same EfficientNet preprocessing
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img, verbose=0)

# Use the same ordered label list created earlier
pred_class_idx = np.argmax(prediction, axis=1)[0]
predicted_label = ordered_labels[pred_class_idx]

print("\n🔹 Prediction probabilities:")
for i, prob in enumerate(prediction[0]):
    print(f"Class '{ordered_labels[i]}': {prob*100:.2f}%")

print(f"\n✅ Predicted Class: {predicted_label}")
