import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


class CustomBoostingImageForgeryDetector(BaseEstimator, ClassifierMixin):
    """
    Custom Boosting Classifier for ensemble learning.

    This classifier implements a custom boosting algorithm similar to AdaBoost.
    It combines multiple base models to create a strong ensemble classifier.

    Key features:
    - Flexible base model selection: Uses a list of provided base models.
    - Iterative learning: Improves prediction by focusing on misclassified samples.
    - Weighted voting: Final prediction is based on weighted votes of base models.
    - Validation error tracking: Monitors performance on a validation set.

    The algorithm works by:
    1. Iteratively training base models on weighted training data.
    2. Selecting the best performing model in each iteration.
    3. Updating sample weights to focus on misclassified samples.
    4. Combining model predictions using learned weights.

    This classifier is compatible with scikit-learn's API, inheriting from
    BaseEstimator and ClassifierMixin.
    """

    def __init__(self, base_models, n_estimators=10, learning_rate=0.1, validation_fraction=0.1, random_state=42):
        """
        Initialize the CustomBoostingClassifier.

        :param base_models: List of base model instances to be used in boosting
        :param n_estimators: Number of boosting iterations
        :param learning_rate: Step size shrinkage used in update to prevents overfitting
        :param validation_fraction: Proportion of training data to set aside as validation set
        :param random_state: Seed for random number generation
        """
        self.base_models = base_models
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.models = []  # List to store selected models
        self.weights = []  # List to store model weights
        self.validation_errors = []  # List to store validation errors

    def print_history(self):
        """
        Print the history for each boosting iteration.
        """
        print("="*50)
        print("Validation Error History:")
        for i, error in enumerate(self.validation_errors, 1):
            print(f"{i}: {error:.4f}")
        print("="*50)
        print("Best Model History:")
        for i, model in enumerate(self.models, 1):
            print(f"{i}: {model.__class__.__name__}")

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

        for _ in range(self.n_estimators):
            best_model = None
            best_error = float('inf')

            # Find the best model for current weighted samples
            for model in self.base_models:
                model.fit(X_train, y_train, sample_weight=sample_weights)
                predictions = model.predict(X_train)
                error = np.sum(sample_weights * (predictions != y_train)) / np.sum(sample_weights)

                if error < best_error:
                    best_model = model
                    best_error = error

            # Calculate model weight
            model_weight = self.learning_rate * np.log((1 - best_error) / best_error)

            # Update sample weights
            predictions = best_model.predict(X_train)
            sample_weights *= np.exp(model_weight * (predictions != y_train))
            sample_weights /= np.sum(sample_weights)  # Normalize weights

            # Store the best model and its weight
            self.models.append(best_model)
            self.weights.append(model_weight)

            # Evaluate on validation set
            val_pred = self.predict(X_val)
            val_error = np.mean(val_pred != y_val)
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
            predictions[i] = model.predict(X)
            # Uncomment below line if working with probabilities
            # predictions[i] = (predictions[i].ravel() > 0.5).astype(int)  # Convert probabilities to labels

        # Compute weighted sum of predictions
        weighted_preds = np.sum(np.array(self.weights)[:, np.newaxis] * predictions, axis=0)

        # Convert to class labels
        return self.classes_[(weighted_preds > 0).astype(int)]
