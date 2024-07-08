import os
import tensorflow as tf
import numpy as np
import random
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from prettytable import PrettyTable


def display_images(images, labels, classnames, n=9):
    """
    Displays the first n images and their corresponding labels.

    Parameters:
    images (np.array): Array of image data.
    labels (np.array): Array of labels corresponding to the images.
    cassnames (np.array): Array of class names corresponding to the labels.
    n (int): Number of images to display (default is 9).
    """
    n = min(9, n)
    plt.figure(figsize=(5, 5))
    for i in range(n):
        plt.subplot(3, 3, i+1)
        plt.imshow(tf.image.decode_image(images[i], channels=3))
        plt.title(f"Label: {classnames[labels[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_file_list(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_list.append(os.path.join(root, file))
    return file_list


def load_images_from_directory(directory_path, n: int, authentic_size, target_size=(128, 128), seed=None):
    if seed is not None:
        random.seed(seed)

    images = []
    labels = []

    authentic_files = [f for f in get_file_list(directory_path) if 'authentic' in f.lower()]
    tampered_files = [f for f in get_file_list(directory_path) if 'tampered' in f.lower()]

    # Calculate the number of authentic images to select (authentic_size of n)
    n_authentic = min(len(authentic_files), int(authentic_size * n))

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


def prepare_image_forgery_dataset(total_number, data_directory, test_size=0.2, random_state=42,
                                  authentic_size=3/5, img_size=(128, 128)):
    X_train, y_train = load_images_from_directory(os.path.join(data_directory, 'train'),
                                                  int((total_number - total_number*test_size) // 1), seed=random_state,
                                                  authentic_size=authentic_size, target_size=img_size)
    X_test, y_test = load_images_from_directory(os.path.join(data_directory, 'test'),
                                                int((total_number*test_size) // 1), seed=random_state,
                                                authentic_size=authentic_size, target_size=img_size)
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    return X_train, X_test, y_train, y_test


def print_model_performance_metrics(y_true, y_pred):
    """
    Prints a nicely formatted table with the following metrics:
    - False Positives (FP)
    - False Negatives (FN)
    - True Positives (TP)
    - True Negatives (TN)
    - Precision
    - Recall
    - F1-score
    - Overall Accuracy

    Parameters:
    y_true (array-like): True values of the target variable.
    y_pred (array-like): Predicted values by the model.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        "False Positives (FP)": fp,
        "False Negatives (FN)": fn,
        "True Positives (TP)": tp,
        "True Negatives (TN)": tn,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Overall Accuracy": accuracy
    }

    # Create a table with headers and alignment
    table = PrettyTable(["Metric", "Value"])
    table.align["Metric"] = "l"
    table.align["Value"] = "r"

    for key, value in metrics.items():
        table.add_row([key, str(value)])

    print(table.get_string())
