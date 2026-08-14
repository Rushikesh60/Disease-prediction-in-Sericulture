import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img

# Paths
input_dir = "dataset"        # original dataset folder with subfolders for each class
output_dir = "augmanted"  # where augmented + denoised images will be saved
os.makedirs(output_dir, exist_ok=True)

# ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Process each class folder
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    save_path = os.path.join(output_dir, class_name)
    os.makedirs(save_path, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # Load and convert to array
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Data augmentation (generate 3 augmented images per original)
        aug_iter = datagen.flow(img_array, batch_size=1)
        for i in range(10):
            aug_img = next(aug_iter)[0].astype('uint8')

            # Apply denoising using OpenCV
            denoised = cv2.fastNlMeansDenoisingColored(aug_img, None, 10, 10, 7, 21)

            # Save processed image
            out_file = os.path.join(save_path, f"{os.path.splitext(img_name)[0]}_aug{i}.jpg")
            cv2.imwrite(out_file, cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR))

print("Data augmentation and noise reduction complete. Processed dataset saved.")
