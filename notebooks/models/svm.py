import cv2
import numpy as np
from sklearn import svm
from skimage.feature import graycomatrix, graycoprops
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf


class SVMImageForgeryDetector(BaseEstimator, ClassifierMixin):
    """
    Support Vector Machine (SVM) based Image Forgery Detector.

    This class implements an image forgery detection system using Support Vector Machines.
    It extracts various features from images, including Fourier transform, noise characteristics,
    edge detection, texture features, and image segmentation. These features are then used
    to train an SVM classifier to detect forged images.

    Key Features:
    - Fourier transform analysis
    - Noise feature extraction
    - Edge detection using Canny algorithm
    - Texture analysis using Gray-Level Co-Occurrence Matrix (GLCM)
    - Image segmentation using K-means clustering
    - SVM-based classification
    - Compatibility: Inherits from sklearn's BaseEstimator and ClassifierMixin for easy integration

    The detector preprocesses images, extracts relevant features, and uses these to train
    an SVM model or make predictions on new images.

    Usage:
    - Initialize the detector with desired image size
    - Use fit() method to train the model on a set of images
    - Use predict() method to detect forgery in new images

    Note: This detector assumes input images are in JPEG format and converts them to
    grayscale during preprocessing.
    """

    def __init__(self, img_size=(128, 128), use_fourier=True, use_noise=True,
                 use_edges=True, use_texture=True, use_segmentation=True):
        """
        Initialize the SVM Image Forgery Detector.

        :param img_size: Tuple of (height, width) for resizing input images
        :param use_fourier: Boolean to include Fourier transform features
        :param use_noise: Boolean to include noise features
        :param use_edges: Boolean to include edge detection features
        :param use_texture: Boolean to include texture features
        :param use_segmentation: Boolean to include segmentation features
        """
        self.img_size = img_size
        self.use_fourier = use_fourier
        self.use_noise = use_noise
        self.use_edges = use_edges
        self.use_texture = use_texture
        self.use_segmentation = use_segmentation
        self.model = svm.SVC()

    @staticmethod
    def compute_fourier_transform(image):
        """Compute the Fourier transform of the image."""
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift))
        magnitude_spectrum[np.isinf(magnitude_spectrum)] = 0  # Replace infinities with 0
        return magnitude_spectrum

    @staticmethod
    def extract_noise_features(image):
        """Extract basic noise model features."""
        mean_noise = np.mean(image)
        std_noise = np.std(image)
        return mean_noise, std_noise

    @staticmethod
    def detect_edges(image):
        """Detect edges in the image using Canny edge detection."""
        edges = cv2.Canny(image, 100, 200)
        return edges

    @staticmethod
    def compute_texture_features(image):
        """Compute texture features using gray-level co-occurrence matrix."""
        glcm = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
        contrast = graycoprops(glcm, 'contrast')
        return np.mean(contrast)

    @staticmethod
    def segment_image(image, k=4):
        """Segment the image using k-means clustering."""
        # Flatten the image to a 1D array suitable for k-means
        pixels = image.reshape((-1, 1))
        pixels = np.float32(pixels)

        # Define criteria and apply k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert back to uint8 and reshape to original image dimensions
        centers = np.uint8(centers)
        segmented_pixels = centers[labels.flatten()]
        segmented_image = segmented_pixels.reshape(image.shape)

        return segmented_image

    def extract_features(self, image):
        """
        Extract various features from the image.

        :param image: Input grayscale image
        :return: 1D array of concatenated features
        """
        if self.use_fourier:
            ft = self.compute_fourier_transform(image).ravel()  # Flattens the result of the Fourier transform 70
        else:
            ft = []

        if self.use_noise:
            nf = self.extract_noise_features(image)  # Returns two scalars 72
        else:
            nf = [], []

        if self.use_edges:
            ed = self.detect_edges(image).ravel()  # Flattens the detected edges 77
        else:
            ed = []

        if self.use_texture:
            tf = np.array([self.compute_texture_features(image)])  # Wraps the scalar in an array 72
        else:
            tf = []

        if self.use_segmentation:
            seg = self.segment_image(image).ravel()  # Flattens the segmented image 58
        else:
            seg = []

        # Concatenates all features into a single 1D array
        return np.hstack([ft, nf[0], nf[1], ed, tf, seg])

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.rgb_to_grayscale(tf.cast(image, tf.float32))
        image = tf.image.resize(image, self.img_size)
        image = image.numpy()
        feat = self.extract_features(image)
        return feat

    def prepare_dataset(self, images):
        images = np.array(images)
        features = [self.preprocess_image(image) for image in images]
        return np.array(features)

    def fit(self, X, y, sample_weight):
        """
        Fit the SVM model to the preprocessed data.

        :param X: List of input images
        :param y: Target labels
        :param sample_weight: Sample weights for training
        :return: Self (fitted model)
        """
        X_processed = self.prepare_dataset(X)
        self.model.fit(X_processed, y, sample_weight=sample_weight)

    def predict(self, X):
        """
        Predict forgery probability for a set of images.

        :param X: Images to predict on
        :return: Array of forgery probabilities
        """
        X_processed = self.prepare_dataset(X)
        return self.model.predict(X_processed)
