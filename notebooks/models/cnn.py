from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class CNNImageForgeryPredictorModel(BaseEstimator, ClassifierMixin):
    def __init__(self, compression_quality: int = 90):
        self.compression_quality = compression_quality
        model = Sequential([
            Input(shape=(128, 128, 1)),
            Conv2D(32, (3, 3), activation='relu'),
            Conv2D(32, (3, 3), activation='relu'),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    def recomprimir_imagen_tf(self, imagen):
        imagen_jpeg = tf.image.encode_jpeg(imagen, quality=self.compression_quality)
        imagen_recomprimida = tf.image.decode_jpeg(imagen_jpeg)
        return imagen_recomprimida

    def preprocess_image(self, image):
        imagen_original = tf.image.decode_jpeg(image, channels=3)
        image_compressed = self.recomprimir_imagen_tf(imagen_original)
        diff = tf.abs(tf.cast(imagen_original, tf.int32) - tf.cast(image_compressed, tf.int32))
        diferencia_gris = tf.image.rgb_to_grayscale(tf.cast(diff, tf.uint8))
        resized = tf.image.resize(diferencia_gris, (128, 128))
        return resized

    def prepare_dataset(self, X):
        X_processed = [self.preprocess_image(image) for image in X]
        return np.array(X_processed)

    def fit(self, X, y, sample_weight):
        X_processed = self.prepare_dataset(X)
        self.model.fit(X_processed, y, sample_weight=sample_weight)

    def predict(self, X):
        X_processed = self.prepare_dataset(X)
        return self.model.predict(X_processed)
