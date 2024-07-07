import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class TransferLearningImageForgeryPredictorModel(BaseEstimator, ClassifierMixin):
    def __init__(self, img_size=(160, 160), dropout_rate=0.2, learning_rate=0.0001):
        # Configuración de la imagen
        self.img_size = img_size
        self.img_shape = self.img_size + (3,)

        # Data Augmentation
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])
        self.preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        # Modelo base
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=self.img_shape, include_top=False,
                                                            weights='imagenet')
        self.base_model.trainable = False

        # Capas adicionales
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.prediction_layer = tf.keras.layers.Dense(1)

        # Construcción del modelo
        self.inputs = tf.keras.Input(shape=self.img_shape)
        x = self.data_augmentation(self.inputs)
        x = self.preprocess_input(x)
        x = self.base_model(x, training=False)
        x = self.global_average_layer(x)
        x = self.dropout_layer(x)
        self.outputs = self.prediction_layer(x)
        self.model = tf.keras.Model(self.inputs, self.outputs)

        # Compilación del modelo
        self.base_learning_rate = learning_rate
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.base_learning_rate),
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        return image

    def prepare_dataset(self, X):
        X_processed = [self.preprocess_image(image) for image in X]
        return np.array(X_processed)

    def fit(self, X, y, sample_weight):
        X_processed = self.prepare_dataset(X)

        initial_epochs = 10
        history = self.model.fit(X_processed, y, epochs=initial_epochs, sample_weight=sample_weight)
        self.base_model.trainable = True

        # Fine-tune from this layer onwards
        fine_tune_at = 100

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False

        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.base_learning_rate/10),
                           metrics=['accuracy'])

        fine_tune_epochs = 10
        total_epochs = initial_epochs + fine_tune_epochs

        self.model.fit(X_processed, y, epochs=total_epochs, initial_epoch=history.epoch[-1])

    def predict(self, X):
        X_processed = self.prepare_dataset(X)
        return self.model.predict(X_processed)
