import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class Classifier:
    def __init__(self, model_type='knn', verbose=False):
        """
        Initializes the Classifier with the given model type and normalizer.

        Parameters:
        - model_type (str): The type of model to use ('knn', 'dt', 'lr', 'gbt').
        - verbose (bool): to print process steps
        """
        self.model_type = model_type
        self.model = None
        self.normalizer = None
        self.grid_params = None
        self.grid_search = None
        self.verbose = verbose

    def build_model(self, model=None, normalizer=None, model_prams=None, grid_params=None, cv=5, scoring='f1'):
        """
        Builds the model with optional grid search for hyperparameter tuning.

        Parameters:
        - model (object): A Scikit-learn model object (optional).
        - normalizer (object): A scikit-learn Scaler object.
        - model_params (dict): In case of GridSearch is not required.
        - param_grid (dict): The parameter grid for GridSearchCV (optional).
        - cv (int): Number of cross-validation folds (default: 5).
        - scoring (str): Scoring metric for GridSearchCV (default: 'f1').
        """
        if model_prams is None:
            model_prams = dict()

        model_mapping = {
            'knn': KNeighborsClassifier(**model_prams),
            'dt': DecisionTreeClassifier(**model_prams),
            'lr': LogisticRegression(**model_prams),
            'gbt': GradientBoostingClassifier(**model_prams)
        }

        param_grid_mapping = {
            'knn': {'n_neighbors': [3, 5, 7, 9, 11]},
            'dt': {'max_depth': [2, 3, 5], 'min_samples_leaf': [0.1, 2], 'random_state': [42],
                   'class_weight': [{0: 1, 1: 1}, {0: 2, 1: 1}, {0: 3, 1: 1}]},
            'lr': {'class_weight': [{0: 1, 1: 1}, {0: 2, 1: 1}, {0: 3, 1: 1}], 'C': [0.1, 1, 10]},
            'gbt': {'max_depth': [2, 3, 5], 'learning_rate': [0.05, 0.1],
                    'n_estimators': [50, 100], 'random_state': [42]}
        }

        self.normalizer = normalizer
        self.model = model if model is not None else model_mapping.get(self.model_type)
        if self.model is None:
            raise ValueError("Unsupported model type. Choose either 'dt', 'knn', 'gbt', or 'lr'.")

        self.grid_params = grid_params if grid_params is not None else param_grid_mapping.get(self.model_type)
        self.grid_search = GridSearchCV(self.model, self.grid_params, cv=cv, scoring=scoring)

        if self.verbose:
            print('- The model was built with the following settings:')
            print({'- Model type': self.model_type, 'Parameters Grid': self.grid_params, 'Score measure': scoring,
                   'Normalizer': self.normalizer, 'Model Parameters': model_prams})

    def fit_normalize(self, x_data):
        """
        Fits the normalizer to a training data and returns the normalized data.

        Parameters:
        - x_data (array-like): Training data to normalize.

        Returns:
        - Transformed training data.
        """
        if self.normalizer is None:
            raise TypeError('Normalizer is not defined. Try again after defining the normalizer.')

        if self.verbose:
            print('- Fit-normalize training data ...')

        return self.normalizer.fit_transform(x_data)

    def normalize(self, x_test):
        """
        Normalizes the test data using the fitted normalizer.

        Parameters:
        - x_test (array-like): Test data to normalize.

        Returns:
        - Transformed test data.
        """
        if self.normalizer is None:
            raise TypeError('Normalizer is not defined. Try again after defining the normalizer.')

        if self.verbose:
            print('- Normalizing data ...')

        return self.normalizer.transform(x_test)

    def tune_hyperparameters(self, x_train, y_train):
        """
        Performs grid search validation and updates the model with the best estimator.

        Parameters:
        - x_train (array-like): Training data.
        - y_train (array-like): Training labels.
        """

        if self.verbose:
            print('- Searching for best hyper-parameters ...')
        self.grid_search.fit(x_train, y_train)
        self.model = self.grid_search.best_estimator_

        if self.verbose:
            print('- Best model:', self.model, self.model.get_params())

    def fit(self, x_train, y_train):
        """
        Fits the model on the training data.

        Parameters:
        - x_train (array-like): Training data.
        - y_train (array-like): Training labels.
        """

        if self.verbose:
            print('- Fitting the model on the training data ...')
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        """
        Predicts the labels and probabilities for the test data.

        Parameters:
        - x_test (array-like): Test data.

        Returns:
        - predictions (array): Predicted labels.
        - predictions_proba (array): Predicted probabilities.
        """

        if self.verbose:
            print('- Predicting test data classes ...')
        predictions = self.model.predict(x_test)
        predictions_proba = self.model.predict_proba(x_test)

        return predictions, predictions_proba

    def evaluate(self, x_test, y_test):
        """
        Evaluates the model on the test data, returning accuracy, precision, recall, f1 and confusion matrix.

        Parameters:
        - x_test (array-like): Test data.
        - y_test (array-like): True labels for the test data.

        Returns:
        - accuracy (float): Accuracy of the model.
        - roc_auc (float): ROC AUC score of the model.
        - conf_matrix (array): Confusion matrix.
        """

        if self.verbose:
            print('- Evaluating the model on the test data ...')
        predictions = self.model.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)

        if self.verbose:
            print('- Results of running the model on the dataset:')
            print(f'- Accuracy: {accuracy:.2f}')
            print(f'- Precision: {precision:.2f}')
            print(f'- Recall: {recall:.2f}')
            print(f'- f1: {f1:.2f}')
            print('- Confusion Matrix:')
            print(conf_matrix)
            print()

        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1,
                'Confusion Matrix': conf_matrix}

    def save_model(self, model_path='model/best_model.pkl', normalizer_path='model/normalizer.pkl'):
        """
        Saves the trained model and normalizer to disk.

        Parameters:
        - model_path (str): Path to save the model (default: 'best_model.pkl').
        - normalizer_path (str): Path to save the normalizer (default: 'normalizer.pkl').
        """

        if self.verbose:
            print('- Saving the models ...')

        with open(model_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)

        if self.normalizer is not None:
            with open(normalizer_path, 'wb') as normalizer_file:
                pickle.dump(self.normalizer, normalizer_file)

    def load_model(self, model_path='model/best_model.pkl', normalizer_path=None):
        """
        Loads the trained model and normalizer from disk.

        Parameters:
        - model_path (str): Path to load the model from (default: 'best_model.pkl').
        - normalizer_path (str): Path to load the normalizer from (default: 'normalizer.pkl').
        """

        if self.verbose:
            print('- Loading the models ...')

        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)

        if normalizer_path is not None:
            with open(normalizer_path, 'rb') as normalizer_file:
                self.normalizer = pickle.load(normalizer_file)
