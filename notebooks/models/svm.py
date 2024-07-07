import cv2
import numpy as np
from sklearn import svm
from skimage.feature import graycomatrix, graycoprops
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf


class SVMImageForgeryPredictorModel(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = svm.SVC()

    @staticmethod
    def fourier_transform(image):
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        magnitude_spectrum[np.isinf(magnitude_spectrum)] = 0  # Reemplazar infinitos con 0
        return magnitude_spectrum

    @staticmethod
    def noise_features(image):
        # modelo de ruido básico
        mean_noise = np.mean(image)
        std_noise = np.std(image)
        return mean_noise, std_noise

    @staticmethod
    def edge_detection(image):
        edges = cv2.Canny(image, 100, 200)
        return edges

    @staticmethod
    def texture_features(image):
        g = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
        contrast = graycoprops(g, 'contrast')
        return np.mean(contrast)

    @staticmethod
    # Compatible with grey scale
    def segment_image(image, k=4):
        # Flatten the image to a 1D array suitable for k-means
        Z = image.reshape((-1, 1))

        # Convert to float32
        Z = np.float32(Z)

        # Criteria and k-means application
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert back to uint8 and map centers to the original image
        center = np.uint8(center)
        res = center[label.flatten()]
        segmented_image = res.reshape((image.shape))

        return segmented_image

    def extract_features(self, image):
        ft = self.fourier_transform(image).ravel()  # Aplana el resultado de la transformada de Fourier 70
        nf = [], []  # noise_features(image)  # Retorna dos escalares 72
        ed = []  # edge_detection(image).ravel()  # Aplana los bordes detectados 77
        tf = []  # np.array([texture_features(image)])  # Envuelve el escalar en un arreglo 72
        seg = self.segment_image(image).ravel()  # Aplana la imagen segmentada 58
        # Concatena todas las características en un solo arreglo 1D
        return np.hstack([ft, nf[0], nf[1], ed, tf, seg])

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.rgb_to_grayscale(tf.cast(image, tf.float32))
        image = tf.image.resize(image, (256, 384))
        image = image.numpy()
        feat = self.extract_features(image)
        return feat

    def prepare_dataset(self, images):
        images = np.array(images)
        features = [self.preprocess_image(image) for image in images]
        return np.array(features)

    def fit(self, X, y, sample_weight):
        X_processed = self.prepare_dataset(X)
        self.model.fit(X_processed, y, sample_weight=sample_weight)

    def predict(self, X):
        X_processed = self.prepare_dataset(X)
        return self.model.predict(X_processed)
