import numpy as np

class DecisionStumpClassifier:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_prediction = None
        self.right_prediction = None
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        best_error = float('inf')
        index = 0
        for feature_index in range(num_features):
            temp = X.iloc[:, feature_index]
            if temp.isnull().any():  # Sprawdza, czy w kolumnie występuje wartość null
                continue
            if isinstance(temp.iloc[0], str) or temp.iloc[0] is None:
                continue
            unique_values = np.unique(temp)
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
            for threshold in thresholds:
                prediction = np.where(X.iloc[:, feature_index] <= threshold, 1, 0)
                error = np.sum(prediction != y)

                if error < best_error:
                    best_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold

                    left_indices = X.iloc[:, feature_index] <= threshold
                    right_indices = ~left_indices

                    self.left_prediction = self._majority_vote(y[left_indices])
                    self.right_prediction = self._majority_vote(y[right_indices])
    def _majority_vote(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    
    def predict(self, X):
        predictions = np.where(X.iloc[:, self.feature_index] <= self.threshold, self.left_prediction, self.right_prediction)
        return predictions