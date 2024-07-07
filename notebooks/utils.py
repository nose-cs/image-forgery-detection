import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


def load_images_from_directory(directory_path, n, i=0, target_size=(224, 224)):
    images = []
    for filename in os.listdir(directory_path)[i:n]:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(directory_path, filename)
            # Read the file
            imagen_original = tf.io.read_file(image_path)
            # Decode the image
            imagen_decoded = tf.image.decode_image(imagen_original, channels=3)
            # Resize the image
            imagen_resized = tf.image.resize(imagen_decoded, target_size)
            # Convert back to bytes
            imagen_encoded = tf.io.encode_jpeg(tf.cast(imagen_resized, tf.uint8))
            images.append(imagen_encoded)
    return images

# def load_images_from_directory(directory_path, n, i=0):
#     images = []
#     for filename in os.listdir(directory_path)[i:n]:
#         if filename.lower().endswith((".jpg", ".jpeg", ".png")):
#             image_path = os.path.join(directory_path, filename)
#             imagen_original = tf.io.read_file(image_path)
#             images.append(imagen_original)
#     return images


def prepare_image_forgery_dataset(authentic_dir, tampered_dir, authentic_number=3000,
                                  tampered_number=3000, test_size=0.2, random_state=42):
    # Get file lists and labels
    authentic_files = load_images_from_directory(authentic_dir, authentic_number)
    tampered_files = load_images_from_directory(tampered_dir, tampered_number)
    authentic_labels = [0] * len(authentic_files)
    tampered_labels = [1] * len(tampered_files)

    # Combine authentic and tampered data
    all_files = authentic_files + tampered_files
    all_labels = authentic_labels + tampered_labels

    all_files = np.array(all_files)
    all_labels = np.array(all_labels)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        all_files, all_labels, test_size=test_size, random_state=random_state
    )

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    return X_train, X_test, y_train, y_test
