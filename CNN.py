import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import cv2

# Dataset path
data_dir = "augmented2"
# -------------------------
# Step 1: Collect image paths
# -------------------------
image_paths = []
labels = []

for class_name in os.listdir(data_dir):
    class_folder = os.path.join(data_dir, class_name)
    if os.path.isdir(class_folder):
        for img_file in os.listdir(class_folder):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_folder, img_file))
                labels.append(class_name)

df = pd.DataFrame({"filename": image_paths, "class": labels})

# -------------------------
# Step 2: Train-Test Split
# -------------------------
train_df, test_df = train_test_split(df, test_size=0.25, stratify=df["class"], random_state=42)

# -------------------------
# Step 3: ImageDataGenerator
# -------------------------
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_dataframe(
    train_df, x_col="filename", y_col="class",
    target_size=(128, 128), class_mode="categorical", batch_size=32
)

test_data = datagen.flow_from_dataframe(
    test_df, x_col="filename", y_col="class",
    target_size=(128, 128), class_mode="categorical", batch_size=32, shuffle=False
)

# -------------------------
# Step 4: CNN Model
# -------------------------
num_classes = len(train_data.class_indices)
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# -------------------------
# Step 5: Early Stopping
# -------------------------
early_stop = EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

# -------------------------
# Step 6: Training
# -------------------------
start_time = time.time()
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=20,
    callbacks=[early_stop],
    verbose=1
)
train_time = time.time() - start_time

# -------------------------
# Step 7: Testing & Evaluation
# -------------------------
start_time = time.time()
y_pred = model.predict(test_data)
test_time = time.time() - start_time

y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes

print(classification_report(y_true, y_pred_classes, target_names=list(test_data.class_indices.keys())))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_data.class_indices.keys(),
            yticklabels=test_data.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

mse = mean_squared_error(y_true, y_pred_classes)
print("Mean Squared Error:", mse)
print(f"⏱ Training Time: {train_time:.2f} sec")
print(f"⏱ Testing Time: {test_time:.2f} sec")

# -------------------------
# Step 8: New Image Prediction with probabilities
# -------------------------
new_image_path = "download.jpg"

start_pred_time = time.time()

# Load and preprocess image
img = cv2.imread(new_image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128, 128))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

# Make prediction
prediction = model.predict(img)  # shape: (1, num_classes)
pred_time = time.time() - start_pred_time

# Get class probabilities
class_labels = list(train_data.class_indices.keys())
for i, prob in enumerate(prediction[0]):
    print(f"Class '{class_labels[i]}': Probability = {prob:.4f}")

# Get predicted class
pred_class = np.argmax(prediction, axis=1)[0]
predicted_label = class_labels[pred_class]

print(f"\n✅ Predicted Class: {predicted_label}")
print(f"⏱ Prediction Time: {pred_time:.4f} sec")

