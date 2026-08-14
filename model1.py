import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, request, jsonify
from flask_cors import CORS
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

# Build ordered index->class list (important for correct label lookup)
class_indices = train_data.class_indices  # dict: class_name -> index
idx_to_class = {v: k for k, v in class_indices.items()}
class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]

# -------------------------
# Step 4: CNN Model
# -------------------------
num_classes = len(class_labels)
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
# Step 6.5: Save model + label mapping for integration
# -------------------------
# Save Keras HDF5 model (easy to load server-side)
MODEL_H5 = "my_model.h5"
CLASS_LIST_JSON = "class_list.json"

try:
    model.save(MODEL_H5)
    # Save idx->class map and ordered class list
    with open("idx_to_class.json", "w") as f:
        # keys saved as strings (JSON requirement)
        json.dump({str(k): v for k, v in idx_to_class.items()}, f)
    with open(CLASS_LIST_JSON, "w") as f:
        json.dump(class_labels, f)
    print(f"Saved model to {MODEL_H5} and labels to {CLASS_LIST_JSON}")
except Exception as e:
    print("Warning: failed to save model/labels:", e)

# -------------------------
# Step 7: Testing & Evaluation
# -------------------------
start_time = time.time()
y_pred = model.predict(test_data)
test_time = time.time() - start_time

y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes

print(classification_report(y_true, y_pred_classes, target_names=class_labels))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

mse = mean_squared_error(y_true, y_pred_classes)
print("Mean Squared Error:", mse)
print(f"⏱ Training Time: {train_time:.2f} sec")
print(f"⏱ Testing Time: {test_time:.2f} sec")

# -------------------------
# Step 8: New Image Prediction with probabilities (standalone)
# -------------------------
new_image_path = "download.jpg"   # <-- change this if you want to test locally

start_pred_time = time.time()

# Load and preprocess image safely
if os.path.exists(new_image_path):
    img = cv2.imread(new_image_path)
    if img is None:
        print(f"Warning: could not read {new_image_path}")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model.predict(img)  # shape: (1, num_classes)
        pred_time = time.time() - start_pred_time

        # Get class probabilities using ordered class_labels
        for i, prob in enumerate(prediction[0]):
            print(f"Class '{class_labels[i]}': Probability = {prob:.4f}")

        # Get predicted class
        pred_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[pred_class]

        print(f"\n✅ Predicted Class: {predicted_label}")
        print(f"⏱ Prediction Time: {pred_time:.4f} sec")
else:
    print(f"No test image found at '{new_image_path}', skipping standalone prediction.")

# -------------------------
# Step 9: Single-file Flask server to integrate with your website
# -------------------------
app = Flask(__name__)
CORS(app)  # allow cross-origin requests; adjust for production

# Try to use in-memory model & labels we already have; if not, attempt to load from disk.
_serving_model = model if 'model' in globals() else None
_serving_labels = class_labels if 'class_labels' in globals() else None

if _serving_model is None:
    # attempt to load saved model
    try:
        _serving_model = load_model(MODEL_H5)
        print(f"Loaded model from {MODEL_H5} for serving.")
    except Exception as e:
        print("Model not available for serving:", e)
if _serving_labels is None:
    try:
        with open(CLASS_LIST_JSON, "r") as f:
            _serving_labels = json.load(f)
        print(f"Loaded labels from {CLASS_LIST_JSON} for serving.")
    except Exception as e:
        print("Labels not available for serving:", e)

def _preprocess_bytes_image(file_bytes, target_size=(128,128)):
    """Preprocess raw image bytes to model input."""
    # decode with cv2 from bytes
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "model_loaded": _serving_model is not None,
        "labels_loaded": _serving_labels is not None
    }

@app.route("/labels", methods=["GET"])
def labels():
    if _serving_labels is None:
        return jsonify({"error": "labels not loaded"}), 500
    return jsonify({"labels": _serving_labels})

@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Accepts multipart/form-data with key 'file' (image).
    Returns JSON:
      {
        "predicted": "<class>",
        "predicted_prob": 0.123,
        "top_k": [{"class": "...", "prob": 0.12}, ...],
        "raw_probs": [...],
        "inference_time_sec": 0.012
      }
    """
    if _serving_model is None:
        return jsonify({"error": "model not loaded"}), 500
    if _serving_labels is None:
        return jsonify({"error": "labels not loaded"}), 500

    # get file from form or raw body
    if 'file' in request.files:
        f = request.files['file']
        file_bytes = f.read()
    else:
        file_bytes = request.get_data()
        if not file_bytes:
            return jsonify({"error": "no file provided"}), 400

    try:
        x = _preprocess_bytes_image(file_bytes, target_size=(128,128))
    except Exception as e:
        return jsonify({"error": "failed to preprocess image", "details": str(e)}), 400

    t0 = time.time()
    preds = _serving_model.predict(x)
    infer_time = time.time() - t0

    probs = preds.flatten().tolist()
    # top-k
    top_k_idx = np.argsort(probs)[::-1][:5]
    top_k = [{"class": _serving_labels[int(i)], "prob": float(probs[int(i)])} for i in top_k_idx]

    top_pred = top_k[0] if top_k else {"class": None, "prob": 0.0}

    return jsonify({
        "predicted": top_pred["class"],
        "predicted_prob": top_pred["prob"],
        "top_k": top_k,
        "raw_probs": probs,
        "inference_time_sec": infer_time
    })

# Run server (for development). In production use gunicorn/uvicorn as appropriate.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_flag = os.environ.get("DEBUG", "0") == "1"
    print(f"Starting server on 0.0.0.0:{port} (debug={debug_flag})")
    app.run(host="0.0.0.0", port=port, debug=debug_flag)
