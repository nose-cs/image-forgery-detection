import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class AdaBoostingImageForgeryDetector(BaseEstimator, ClassifierMixin):
    """
    An AdaBoost-based model for detecting image forgery using an ensemble of base classifiers.

    This class implements an advanced image forgery detection system that leverages the power of
    AdaBoost (Adaptive Boosting) algorithm. It combines multiple weak learners to create a strong
    classifier capable of identifying potential image manipulations.

    Key features:
    1. Ensemble learning: Utilizes multiple base models to improve overall detection accuracy.
    2. Adaptive weighting: Adjusts the importance of misclassified samples during training.
    3. Flexible base models: Can incorporate various types of classifiers as base models.
    4. Validation monitoring: Tracks performance on a validation set during training.
    5. Compatibility: Inherits from sklearn's BaseEstimator and ClassifierMixin for easy integration.

    Detection process:
    1. Initializes with a set of base models and hyperparameters.
    2. Trains each base model sequentially, adjusting sample weights based on misclassifications.
    3. Combines predictions from all base models using a weighted voting scheme.
    4. Produces final classification based on the ensemble's collective decision.

    The AdaBoost algorithm iteratively trains base models, focusing more on samples that were
    misclassified in previous iterations. This approach allows the ensemble to learn complex
    decision boundaries and potentially capture subtle forgery indicators that individual
    models might miss.

    Usage:
    - Initialize the detector with desired base models and parameters.
    - Use the fit() method to train the model on a set of image features.
    - Use the predict() method to detect forgery in new images.

    Note: This detector's performance depends on the choice and diversity of base models.
    It's particularly effective when base models have complementary strengths in detecting
    different types of image forgeries.
    """

    def __init__(self, base_models, learning_rate=0.1, validation_fraction=0.1, random_state=42):
        """
        Initialize the AdaBoostingClassifier.

        :param base_models: List of base model instances to be used in boosting
        :param learning_rate: Step size shrinkage used in update to prevents overfitting
        :param validation_fraction: Proportion of training data to set aside as validation set
        :param random_state: Seed for random number generation
        """
        self.base_models = base_models
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.weights = []  # List to store model weights
        self.validation_errors = []  # List to store validation errors
        self.validation_accurracy = []  # List to store validation accurracy
        self.models = []

    def print_history(self):
        """
        Print the history for each boosting iteration.
        """
        print("="*50)
        print("Validation Error History:")
        for i, error in enumerate(self.validation_errors, 1):
            print(f"{i}: {error:.4f}")
        print("="*50)
        print("Validation Accurracy History:")
        for i, acc in enumerate(self.validation_accurracy, 1):
            print(f"{i}: {acc:.4f}")

    def fit(self, X, y):
        """
        Fit the boosting classifier on the training data.

        :param X: Training data features
        :param y: Training data labels
        :return: self
        """
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_fraction,
                                                          random_state=self.random_state)
        self.classes_ = np.unique(y_train)
        n_samples = X_train.shape[0]
        sample_weights = np.ones(n_samples) / n_samples  # Initialize sample weights

        # Find the best model for current weighted samples
        for model in self.base_models:
            model.fit(X_train, y_train, sample_weight=sample_weights)
            predictions = model.predict(X_train).ravel()  # Ensure predictions are 1D

            # Ensure y_train is 1D as well
            y_train = y_train.ravel()

            error = np.sum(sample_weights * (predictions != y_train)) / np.sum(sample_weights)

            # Avoid division by zero or log(0)
            epsilon = 1e-10
            error = max(epsilon, min(error, 1 - epsilon))

            # Calculate model weight
            model_weight = self.learning_rate * np.log((1 - error) / error)

            # Update sample weights
            sample_weights *= np.exp(model_weight * (predictions != y_train))
            sample_weights /= np.sum(sample_weights)  # Normalize weights

            # Store the best model and its weight
            self.weights.append(model_weight)
            self.models.append(model)

            # Evaluate on validation set
            val_pred = self.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            self.validation_accurracy.append(val_acc)
            val_error = np.mean(val_pred != y_val.ravel())
            self.validation_errors.append(val_error)

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        :param X: The input samples
        :return: Predicted class labels
        """
        predictions = np.zeros((len(self.models), X.shape[0]))
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            # Ensure predictions are 1D
            predictions[i] = pred.ravel()

        # Compute weighted sum of predictions
        weighted_preds = np.sum(np.array(self.weights)[:, np.newaxis] * predictions, axis=0)

        # Convert to class labels
        return self.classes_[(weighted_preds > 0).astype(int)]
