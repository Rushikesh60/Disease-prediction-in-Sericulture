import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier   # ✅ Random Forest
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2

# Dataset path
data_dir = "augmanted"

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
    target_size=(128, 128), class_mode="categorical", batch_size=32, shuffle=False
)

test_data = datagen.flow_from_dataframe(
    test_df, x_col="filename", y_col="class",
    target_size=(128, 128), class_mode="categorical", batch_size=32, shuffle=False
)

# -------------------------
# Step 4: CNN Feature Extractor
# -------------------------
num_classes = len(train_data.class_indices)
cnn_model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5)
])

# -------------------------
# Step 5: Feature Extraction
# -------------------------
start_time = time.time()
X_train_features = cnn_model.predict(train_data, verbose=1)
X_test_features = cnn_model.predict(test_data, verbose=1)

y_train = train_data.classes
y_test = test_data.classes

print(f"✅ Feature extraction done in {time.time() - start_time:.2f} sec")
print(f"Feature shape: {X_train_features.shape}")

# -------------------------
# Step 6: Train Random Forest Classifier
# -------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)

start_time = time.time()
rf.fit(X_train_features, y_train)
train_time = time.time() - start_time

# -------------------------
# Step 7: Testing & Evaluation
# -------------------------
start_time = time.time()
y_pred = rf.predict(X_test_features)
y_pred_proba = rf.predict_proba(X_test_features)
test_time = time.time() - start_time

print(classification_report(y_test, y_pred, target_names=list(test_data.class_indices.keys())))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_data.class_indices.keys(),
            yticklabels=test_data.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print(f"⏱ Training Time (RF): {train_time:.2f} sec")
print(f"⏱ Testing Time (RF): {test_time:.2f} sec")

# -------------------------
# Step 8: New Image Prediction
# -------------------------
new_image_path = "601cdf67-eb0b-4489-8762-99161f9841ad.png"

# Load and preprocess image
img = cv2.imread(new_image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128, 128))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

# Extract features from CNN
img_features = cnn_model.predict(img)

# Predict with Random Forest
start_pred_time = time.time()
rf_prediction = rf.predict(img_features)
rf_pred_proba = rf.predict_proba(img_features)
pred_time = time.time() - start_pred_time

class_labels = list(train_data.class_indices.keys())
for i, prob in enumerate(rf_pred_proba[0]):
    print(f"Class '{class_labels[i]}': Probability = {prob:.4f}")

predicted_label = class_labels[rf_prediction[0]]
print(f"\n✅ Predicted Class: {predicted_label}")
print(f"⏱ Prediction Time: {pred_time:.4f} sec")
