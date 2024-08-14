import argparse
import json
from core.model import Classifier
from core.data_processor import DataProcessor
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":

    """
    Only for training, no testing or evaluating is done in this script.
    """

    parser = argparse.ArgumentParser(description='Train a predictive model.')
    parser.add_argument('--origin', type=str, default='data/origin.txt', help='Path to the origin file.')
    parser.add_argument('--accepted', type=str, default='data/127_accepted.txt',
                        help='Path to the accepted proposals file.')
    parser.add_argument('--refused', type=str, default='data/127_refused.txt',
                        help='Path to the refused proposals file.')
    parser.add_argument('--model', type=str, default='knn', help='Model type to use (dt, lr, knn or gbt).')
    parser.add_argument('--model_path', type=str, default='models/model.pkl',
                        help='Path to save the model.')
    parser.add_argument('--normalizer_path', type=str, default='models/normalizer.pkl',
                        help='Path to save the normalizer.')
    parser.add_argument('--verbose', type=bool, default=True, help='Printing step by step.')
    parser.add_argument('--ft_en', type=bool, default=True,
                        help='Whether to feature engineer the dataset or not.')
    parser.add_argument('--normalize', type=bool, default=True,
                        help='Whether to normalize the dataset or not.')
    parser.add_argument('--fine_tune', type=bool, default=True,
                        help='Whether to fine-tune the hyper-parameters or not.')
    parser.add_argument('--model_params', type=str, default=None,
                        help='Model parameters in case no grid-search is required.')
    parser.add_argument('--grid_params', type=str, default=None,
                        help='Model parameters in case no grid-search is required.')
    parser.add_argument('--cv', type=int, default=5,
                        help='number of folds to do while fine-tuning the hyper-parameters.')
    parser.add_argument('--score', type=str, default='f1', help='Grid-search quality measure')

    args = parser.parse_args()

    origin = args.origin
    accepted = args.accepted
    refused = args.refused

    # Load and preprocess the dataset
    data_loader = DataProcessor(origin_file=origin, accepted_file=accepted, refused_file=refused)
    data_loader.preprocess_data()
    data_loader.describe_data()

    # We can visualize the data too.
    data_loader.visualize_data()

    features_list = ['distance', 'prix']
    if args.ft_en:
        data_loader.feature_engineering()
        features_list.append('euro_per_km')

    # weights = {'distance': 1, 'price': 2}

    print('- Finished pre-processing the dataset ...')
    print('-----------------------------------------------------------------------------------------------')
    print()
    # Prepare data for modeling
    X = data_loader.data[features_list]

    y = data_loader.data['accepted']

    # Initialize the models
    model_obj = Classifier(model_type=args.model, verbose=args.verbose)

    # Normalizing the data
    normalizer = None
    if args.normalize:
        normalizer = StandardScaler()

    # Loading the model parameters
    model_params = None
    if args.model_params is not None:
        model_params = json.load(args.model_params)

    # Loading the grid-search parameters
    grid_params = None
    if args.grid_params is not None:
        grid_params = json.load(args.grid_params)

    # Building the model with the given parameters.
    model_obj.build_model(normalizer=normalizer, cv=args.cv, grid_params=grid_params,
                          model_prams=model_params, scoring=args.score)

    # Normalizing the dataset.
    if args.normalize:
        X = model_obj.fit_normalize(X)

    # Grid-Search the hyperparameters.
    if args.fine_tune:
        model_obj.tune_hyperparameters(X, y)

    # Train the model with the best parameters, default ones or the given parameters.
    model_obj.fit(X, y)

    # Save the model.
    model_obj.save_model()

    print('- Done training the model on the dataset ...')
