from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.model_selection import train_test_split
import random


class CNNImageForgeryDetector(BaseEstimator, ClassifierMixin):
    """
    A Convolutional Neural Network (CNN) based model for detecting image forgery using recompression analysis.

    This class implements an advanced image forgery detection system that leverages the principles of
    JPEG compression artifacts. It employs a unique approach of comparing an original image with its
    recompressed version to identify potential manipulations.

    Key features:
    1. Recompression-based analysis: Utilizes JPEG compression artifacts for forgery detection.
    2. CNN architecture: Employs a custom CNN for learning and detecting forgery patterns.
    3. Preprocessing pipeline: Includes image decoding, recompression, and difference computation.
    4. Compatibility: Inherits from sklearn's BaseEstimator and ClassifierMixin for easy integration.

    Detection process:
    1. Decodes the input JPEG image.
    2. Recompresses the image using a specified quality factor.
    3. Computes the difference between the original and recompressed images.
    4. Feeds this difference into a CNN for classification.

    The CNN architecture is designed to learn subtle patterns in the compression artifact
    differences that indicate potential forgery. This method is particularly effective for
    detecting manipulated areas in JPEG images, as these areas often respond differently
    to recompression compared to authentic regions.

    Usage:
    - Initialize the detector with desired parameters (compression quality, image size, etc.)
    - Use the fit() method to train the model on a set of images
    - Use the predict() method to detect forgery in new images

    Note: This detector is optimized for JPEG images and may require adaptation for other formats.
    """

    def __init__(self, compression_quality: int = 90, img_size=(128, 128), epochs=10, learning_rate=0.1):
        """
        Initialize the CNN model for image forgery detection.

        :param compression_quality: JPEG compression quality to use in preprocessing, default is 90
        :type compression_quality: int
        :param img_size: The target size for input images (height, width)
        :type img_size: tuple
        """
        self.learning_rate = learning_rate
        self.compression_quality = compression_quality
        self.img_size = img_size
        self.img_shape = self.img_size + (1,)
        self.epochs = epochs
        # Define and compile the CNN model
        model = Sequential([
            Input(shape=self.img_shape),
            Conv2D(32, (3, 3), activation='relu'),
            Conv2D(32, (3, 3), activation='relu'),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(2, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def recompress_image_tf(self, image):
        """
        Recompress the input image using JPEG compression.

        :param image: Input image tensor
        :return: Recompressed image tensor
        """
        jpeg_image = tf.image.encode_jpeg(image, quality=self.compression_quality)
        recompressed_image = tf.image.decode_jpeg(jpeg_image)
        return recompressed_image

    def preprocess_image(self, image):
        """
        Preprocess a single image for forgery detection.

        :param image: Raw image data
        :return: Preprocessed image tensor
        """
        # Decode the JPEG image
        original_image = tf.image.decode_jpeg(image, channels=3)
        # Recompress the image
        compressed_image = self.recompress_image_tf(original_image)
        # Compute the absolute difference between original and recompressed images
        diff = tf.abs(tf.cast(original_image, tf.int32) - tf.cast(compressed_image, tf.int32))
        # Convert the difference to grayscale
        grayscale_diff = tf.image.rgb_to_grayscale(tf.cast(diff, tf.uint8))
        # Resize the image to match the input size of the CNN
        resized_diff = tf.image.resize(grayscale_diff, self.img_size)
        return resized_diff

    def prepare_dataset(self, X):
        """
        Prepare a batch of images for model input.

        :param X: List of raw image data
        :return: Numpy array of preprocessed images
        """
        X_processed = [self.preprocess_image(image) for image in X]
        return np.array(X_processed)

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model to the training data.

        :param X: Training images
        :param y: Target labels
        :param sample_weight: Sample weights for training
        """
        data = list(zip(X, y))
        random.shuffle(data)
        X, y = zip(*data)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        X_train = self.prepare_dataset(X_train)
        X_val = self.prepare_dataset(X_val)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
        y_val = tf.keras.utils.to_categorical(y_val, num_classes=2)

        return self.model.fit(X_train, y_train,
                              sample_weight=sample_weight,
                              epochs=self.epochs,
                              validation_data=(X_val, y_val))

    def predict(self, X):
        """
        Predict forgery probability for a set of images.

        :param X: Images to predict on
        :return: Array of forgery probabilities
        """
        X_processed = self.prepare_dataset(X)
        predictions = self.model.predict(X_processed)
        result = []
        for i in range(len(predictions)):
            predicted_label = 1 if predictions[i][0] < predictions[i][1] else 0
            result.append(predicted_label)
        return np.array(result)
