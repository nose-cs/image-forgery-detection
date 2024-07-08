import os
import tensorflow as tf
import numpy as np
import random


def get_file_list(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_list.append(os.path.join(root, file))
    return file_list


def load_images_from_directory(directory_path, n: int, target_size=(128, 128), seed=None):
    if seed is not None:
        random.seed(seed)

    images = []
    labels = []

    authentic_files = [f for f in get_file_list(directory_path) if 'authentic' in f.lower()]
    tampered_files = [f for f in get_file_list(directory_path) if 'tampered' in f.lower()]

    # Calculate the number of authentic images to select (3/5 of n)
    n_authentic = min(len(authentic_files), int(3/5 * n))

    # Select authentic images
    selected_authentic = random.sample(authentic_files, n_authentic)

    # Select tampered images
    n_tampered = min(len(tampered_files), n - n_authentic)
    selected_tampered = random.sample(tampered_files, n_tampered)

    selected_files = selected_authentic + selected_tampered
    random.shuffle(selected_files)  # Shuffle to mix authentic and tampered

    for image_path in selected_files:
        if image_path.lower().endswith((".jpg", ".jpeg", ".png")):
            label = 0 if 'authentic' in image_path.lower() else 1
            imagen_original = tf.io.read_file(image_path)
            imagen_decoded = tf.image.decode_image(imagen_original, channels=3)
            imagen_resized = tf.image.resize(imagen_decoded, target_size)
            imagen_encoded = tf.io.encode_jpeg(tf.cast(imagen_resized, tf.uint8))
            images.append(imagen_encoded)
            labels.append(label)
        else:
            raise ValueError(f"Unable to process file: {image_path}")
    return np.array(images), np.array(labels)


def prepare_image_forgery_dataset(total_number, data_directory, test_size=0.2, random_state=42):
    X_train, y_train = load_images_from_directory(os.path.join(data_directory, 'train'),
                                                  int((total_number - total_number*test_size) // 1), seed=random_state)
    X_test, y_test = load_images_from_directory(os.path.join(data_directory, 'test'),
                                                int((total_number*test_size) // 1), seed=random_state)
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    return X_train, X_test, y_train, y_test
