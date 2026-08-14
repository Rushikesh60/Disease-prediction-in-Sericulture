import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# -------------------------
# Configuration
# -------------------------
input_dir = "E:\Program Files\PycharmProjects\project3\dataset2"        # Original dataset folder
output_dir = "augmented2"  # Output folder for augmented + denoised images
target_size = (128, 128)
augment_per_class = 80       # Target total images per class after augmentation

os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Define ImageDataGenerator for realistic augmentations
# -------------------------
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.3],
    channel_shift_range=25,
    fill_mode='nearest'
)

# -------------------------
# Function: Augment & Denoise Images
# -------------------------
def augment_and_denoise(class_path, save_path, target_count):
    img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n_original = len(img_files)

    print(f"\n🔹 Processing class '{os.path.basename(class_path)}' ({n_original} images)...")

    # Create output folder
    os.makedirs(save_path, exist_ok=True)

    # If class already has enough images, just copy and denoise originals
    if n_original >= target_count:
        for img_name in img_files:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            cv2.imwrite(os.path.join(save_path, img_name), denoised)
        return

    # Calculate how many augmentations per image needed
    aug_per_image = int(np.ceil(target_count / n_original))

    for img_name in img_files:
        img_path = os.path.join(class_path, img_name)
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        aug_iter = datagen.flow(img_array, batch_size=1)

        for i in range(aug_per_image):
            aug_img = next(aug_iter)[0].astype('uint8')

            # Denoise
            denoised = cv2.fastNlMeansDenoisingColored(aug_img, None, 10, 10, 7, 21)

            # Save
            out_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(save_path, out_name), cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR))

    print(f"✅ Augmented {len(os.listdir(save_path))} images saved to {save_path}")

# -------------------------
# Run augmentation for all classes
# -------------------------
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    save_path = os.path.join(output_dir, class_name)

    if os.path.isdir(class_path):
        augment_and_denoise(class_path, save_path, augment_per_class)

print("\n🎯 Data augmentation and denoising complete. Balanced dataset created in:", output_dir)
