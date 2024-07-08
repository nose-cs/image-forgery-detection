import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class TransferLearningImageForgeryDetector(BaseEstimator, ClassifierMixin):
    """
    A transfer learning-based model for detecting image forgery.

    It utilizes a pre-trained MobileNetV2 model as the base architecture
    and adds custom layers for forgery detection. The model is designed to learn and
    detect subtle patterns that indicate potential image manipulation.

    Key features:
    - Transfer Learning: Uses pre-trained MobileNetV2 for feature extraction
    - Data Augmentation: Implements random flipping and rotation for improved generalization
    - Fine-tuning: Includes a two-step training process with initial training and fine-tuning
    - Compatibility: Inherits from sklearn's BaseEstimator and ClassifierMixin for easy integration

    The detector preprocesses input images, performs feature extraction using the base model,
    and then uses the custom layers to make forgery predictions.

    The model architecture includes:
    1. Data augmentation layers
    2. Pre-trained MobileNetV2 base (with option for fine-tuning)
    3. Global Average Pooling
    4. Dropout for regularization
    5. Dense layer for final prediction

    Usage:
    - Initialize the detector with desired parameters (image size, dropout rate, learning rate, etc.)
    - Use the fit() method to train the model on a set of images
    - Use the predict() method to detect forgery in new images

    Note: This detector assumes input images are in JPEG format and resizes them during preprocessing.
    """

    def __init__(self, img_size=(160, 160), dropout_rate=0.2, learning_rate=0.0001,
                 initial_epochs=10, fine_tune_epochs=10):
        """
        Initialize the TransferLearningImageForgeryDetector.

        :param img_size: The target size for input images (height, width)
        :type img_size: tuple
        :param dropout_rate: The dropout rate for the model
        :type dropout_rate: float
        :param learning_rate: The initial learning rate for the optimizer
        :type learning_rate: float
        """
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.initial_epochs = initial_epochs
        self.fine_tune_epochs = fine_tune_epochs

        # Image configuration
        self.img_size = img_size
        self.img_shape = self.img_size + (3,)

        # Data Augmentation
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])
        self.preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        # Base model
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.img_shape,
            include_top=False,
            weights='imagenet'
        )
        self.base_model.trainable = False

        # Additional layers
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.prediction_layer = tf.keras.layers.Dense(1)

        # Model construction
        inputs = tf.keras.Input(shape=self.img_shape)
        x = self.data_augmentation(inputs)
        x = self.preprocess_input(x)
        x = self.base_model(x, training=False)
        x = self.global_average_layer(x)
        x = self.dropout_layer(x)
        outputs = self.prediction_layer(x)
        self.model = tf.keras.Model(inputs, outputs)

        # Model compilation
        self.base_learning_rate = learning_rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.base_learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def preprocess_image(self, image):
        """
        Preprocess a single image.

        :param image: Raw image data
        :type image: bytes
        :return: Preprocessed image tensor
        :rtype: tf.Tensor
        """
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        return image

    def prepare_dataset(self, X):
        """
        Prepare a batch of images for model input.

        :param X: List of raw image data
        :type X: list
        :return: Array of preprocessed images
        :rtype: np.array
        """
        X_processed = [self.preprocess_image(image) for image in X]
        return np.array(X_processed)

    def fit(self, X, y, sample_weight):
        """
        Fit the model to the training data.

        This method includes both initial training and fine-tuning steps.

        :param X: List of raw image data
        :type X: list
        :param y: Target values
        :type y: array-like
        :param sample_weight: Weight for each sample
        :type sample_weight: array-like
        :return: Returns an instance of self
        :rtype: TransferLearningImageForgeryDetector
        """
        X_processed = self.prepare_dataset(X)

        # Initial training
        history = self.model.fit(X_processed, y, epochs=self.initial_epochs, sample_weight=sample_weight)

        # Fine-tuning
        self.base_model.trainable = True
        fine_tune_at = 100

        # Freeze layers before the `fine_tune_at` layer
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False

        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.base_learning_rate/10),
            metrics=['accuracy']
        )

        total_epochs = self.initial_epochs + self.fine_tune_epochs

        self.model.fit(X_processed, y, epochs=total_epochs, initial_epoch=history.epoch[-1])

        return self

    def predict(self, X):
        """
        Predict the probability of forgery for given images.

        :param X: List of raw image data
        :type X: list
        :return: Array of forgery probabilities
        :rtype: np.array
        """
        X_processed = self.prepare_dataset(X)
        return self.model.predict(X_processed)
