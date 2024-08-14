import argparse
import pandas as pd
import numpy as np
from core.model import Classifier  # Assuming the class code is saved in a file named classifier.py


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test and evaluate a predictive model.')

    parser.add_argument('--model_path', type=str, default='model/best_model.pkl',
                        help='Path to the model.')
    parser.add_argument('--normalizer_path', type=str, default='model/normalizer.pkl',
                        help='Path to the normalizer.')
    parser.add_argument('--test_set_path', type=str, default='data/test.txt',
                        help='Path to a test set.')
    parser.add_argument('--output_path', type=str, default='data/predictions.csv',
                        help='Path to save the results in a csv file.')
    parser.add_argument('--verbose', type=bool, default=True, help='Printing step by step.')
    parser.add_argument('--normalize', type=bool, default=True,
                        help='Whether to normalize the dataset or not.')
    args = parser.parse_args()

    # Paths to the model, normalizer, and the new dataset
    model_path = args.model_path
    normalizer_path = args.normalizer_path

    # Load the new dataset
    '''
    test.txt contains:
    [['distance', 'prix', 'euro_per_km']
    [10, 100, 10],
    [16, 2, 0.125],
    [2, 50, 25],
    [10, 50, 5]]
    
    Which mimic the wanted test of a new job that pays 100 euros 10 KM away and three others random tests.
    '''
    new_data = pd.read_csv(args.test_set_path).values

    # Load the classifier and the pre-trained model and scaler
    clf = Classifier(verbose=args.verbose)
    clf.load_model(model_path, normalizer_path)

    # Assuming the new_data does not contain the target variable
    # Normalize the data
    if args.normalize:
        new_data = clf.normalize(new_data)

    # Make predictions
    predictions, predictions_proba = clf.predict(new_data)

    # Save the predictions to a CSV file
    results = pd.DataFrame({
        'Prediction': predictions,
        'Probability_class_0': np.round(predictions_proba[:, 0], 2),
        'Probability_class_1': np.round(predictions_proba[:, 1], 2)  # Assuming binary classification
    })
    results.to_csv(args.output_path, index=False)

    print(f'- Predictions saved to {args.output_path}')
