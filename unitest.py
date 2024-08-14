import unittest
import pandas as pd
import numpy as np
from core.model import Classifier  # Assuming the class code is saved in a file named classifier.py


class TestClassifier(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case by initializing the Classifier and loading the model and scaler.
        """
        # Paths to the model and scaler
        self.model_path = 'model/best_model.pkl'
        self.scaler_path = 'model/normalizer.pkl'

        # Initialize the classifier
        self.clf = Classifier()

        # Load the pre-trained model and scaler
        self.clf.load_model(self.model_path, self.scaler_path)

        # Create a small sample dataset for testing (replace with appropriate columns)
        self.sample_data = pd.DataFrame({
            'distance': [10, 16, 2],
            'prix': [100, 2, 50],
            'euro_per_km': [10, 0.125, 25]
        })

    def test_normalization(self):
        """
        Test that the normalization process is working correctly.
        """
        # Normalize the sample data
        normalized_data = self.clf.normalize(self.sample_data)

        # Check that the normalization output shape matches the input shape
        self.assertEqual(normalized_data.shape, self.sample_data.shape)

        # Check that the data is not identical (indicating some transformation has occurred)
        self.assertFalse(np.array_equal(normalized_data, self.sample_data.values))

    def test_prediction(self):
        """
        Test that the prediction process is working correctly.
        """
        # Normalize the sample data
        normalized_data = self.clf.normalize(self.sample_data)

        # Make predictions
        predictions, predictions_proba = self.clf.predict(normalized_data)

        # Check that the predictions output has the correct length
        self.assertEqual(len(predictions), len(self.sample_data))

        # Check that the probabilities output has the correct shape
        self.assertEqual(predictions_proba.shape, (len(self.sample_data), 2))  # Assuming binary classification


if __name__ == '__main__':
    unittest.main()
