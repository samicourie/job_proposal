
## Job Proposal Predictor
The goal of this project is to predict whether a professional will accept a new job proposal based on the distance and the price of the proposal.

The solution is split into two parties:
1. A jupyter notebook where I show the general approach on how to solve the problem.
2. Structured code that mimics a production scenario, allowing us to train and test the model separately.

### Problem
The goal is to develop a predictive model that only considers the variables of distance and price. For a given professional, based on the history of proposals they have accepted or declined, the aim is to predict whether they will accept a future proposal at a given distance and price.


### Code Requirements
Using python 3.12.3, you can install separately:
```
pip install pandas==2.2.2
pip install numpy==1.26.4
pip install matplotlib==3.9.1
pip install seaborn==0.13.2
pip install geopy==2.4.1
pip install scikit-learn==1.4.2
```

or all together:
```
pip install -r requirements.txt
```

### Jupyter Notebook Code
In **core.ipynb**, a step by step solution is given:
1. Load the data and identify redundant and NaN rows.
2. Calculating of the distance (KM).
3. Perform feature engineering by adding a third feature: **Euro Per KM**.
4. Plot the dataset relative to the target value (whether a job was accepted or not),
5. Fine-tuning and compare different machine learning models in term of accuracy, precision, etc.
6. Explain the results and discuss potential improvements.

### Structured Solution
The code is organized as follows:
```
├── data/
│   ├── 127_accepted.txt
|   ├── 127_refused.txt
|   ├── origin.txt
|   ├── predictions.csv
│   └── test.txt
|
├── model/ 
│   ├── best_model.pkl 
│   └── normalizer.pkl 
│ 
├── core/ 
│   ├── data_processor.py 
│   ├── model.py 
│   └── utils.py  
│ 
├── core.ipynb 
├── figure_1.png 
├── figure_2.png
├── requirements.txt 
├── README.md
├── test.py 
├── train.py 
└── unitest.py
```
1. **data**: The **test.txt** file mimics new cases to test, including a job proposal of 100 euros 10 KM away. **predictions.csv** stores the results of running the model on a new test set (**test.txt**).
2. model: Stores the fitted model and the normalizer.
3. core: **data_processor.py** includes the pre-processing step (describing, cleaning and visualizing the data). **model.py** includes the different steps to train, validate, save and load a model. And finally, **utils.py** includes a function that applies weights on the dataset.
4. **train.py**: Trains a model on a dataset with different parameters.
5. **test.py**: Tests the pre-trained model on a new dataset.
6. **unitest.py**: A unit test to ensure the code runs correctly.

### Traning a New Model
Use **train.py** to train a new model with the following options:
```
options:
  -h, --help                		show this help message and exit
  --origin ORIGIN           		Path to the origin file.
  --accepted ACCEPTED       		Path to the accepted proposals file.
  --refused REFUSED         		Path to the refused proposals file.
  --model MODEL             		Model type to use (dt, lr, knn or gbt).
  --model_path MODEL_PATH   		Path to save the model.
  --normalizer_path 				Path to save the normalizer.
  --verbose VERBOSE     			Printing step by step.
  --ft_en FT_EN         			Whether to feature engineer the dataset or not.
  --normalize NORMALIZE				Whether to normalize the dataset or not.
  --fine_tune FINE_TUNE				Whether to fine-tune the hyper-parameters or not.
  --model_params MODEL_PARAMS		Model parameters in case no grid-search is required.
  --grid_params GRID_PARAMS			Model parameters in case no grid-search is required.
  --cv CV               			number of folds to do while fine-tuning the hyper-parameters.
  --score SCORE         			Grid-search quality measure
```

Where **'dt'** represents DecisionTreeClassifier, **'lr'** represents LogisticRegression, **'knn'** represents KNearestNeighbors, and **'gbt'*** represents GradientBoostingClassifier.

Running **train.py** with the default parameters,
```
python train.py
```

Produces the following output:
```
- Calculating the distance in KM ...
- Merging accepted and refused dataset ...
- Calculating the price in euros instead of cents ...
- Data description with pandas ...
          latitude    longitude         prix     distance     accepted
count  1000.000000  1000.000000  1000.000000  1000.000000  1000.000000
mean     48.847029     2.312354    49.952130     9.381560     0.752000
std       0.040118     0.056752    31.342118     4.330834     0.432068
min      48.261854     1.847944    18.000000     0.509887     0.000000
25%      48.829058     2.280951    29.000000     6.590341     1.000000
50%      48.845598     2.305581    36.500000     8.658094     1.000000
75%      48.876206     2.343973    65.000000    11.512534     1.000000
max      48.936420     2.691792   248.000000    75.975176     1.000000

- Rows that contain NaN ...
latitude     0
longitude    0
prix         0
distance     0
accepted     0
dtype: int64

- Duplicated: 235

- Adding euro_per_km feature to the dataset ...
- Finished pre-processing the dataset ...
-----------------------------------------------------------------------------------------------

- The model was built with the following settings:
{'- Model type': 'knn', 'Parameters Grid': {'n_neighbors': [3, 5, 7, 9, 11]}, 'Score measure': 'f1', 'Normalizer': StandardScaler(), 'Model Parameters': {}}
- Fit-normalize training data ...
- Searching for best hyper-parameters ...
- Best model: KNeighborsClassifier() {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}
- Fitting the model on the training data ...
- Saving the models ...
- Done training the model on the dataset ...
```
The best model will be saved in **model/best_model.pkl**, and the normalizer will be saved alongside it. This model can then be used to test on a new dataset. You can also experiment with different models and parameters, such as a decision tree with a maximum depth of 4.

### Testing the Model
To test the model, we can use the script **test.py**. The following are the input parameters:
```
Test and evaluate a predictive model.

options:
  -h, --help            show this help message and exit
  --model_path 			Path to the model.
  --normalizer_path     Path to the normalizer.
  --test_set_path       Path to a test set.
  --output_path         Path to save the results in a csv file.
  --verbose VERBOSE     Printing step by step.
  --normalize           Whether to normalize the dataset or not.
```

And if we run **test.py** with the default parameters, test dataset is **data/test.txt** that contains a job proposal if 100 euros 10KM away and three other random cases, using the model we learned before, we get:
```
- Predicting test data classes ...
- Predictions saved to data/predictions.csv
```

The results will be saved by default in **predictions.csv**:
```
Prediction,Probability_class_0,Probability_class_1  
1,0.0,1.0  			
0,0.8,0.2  
1,0.0,1.0  
1,0.0,1.0
```
The first line (100 euros proposal, 10 KM away) suggests accepting the proposal with 100% confidence. The second line (2 euros proposal, 16 KM away) suggests not accepting the proposal with 80% confidence, and so on.

### Improvement of Results
1. **Data Imbalance**: The data is unbalanced. Balancing techniques such as k-medoids could reduce the number of 'accepted' jobs and balance the data. However, caution is needed to avoid losing valuable statistical information.
2. **Additional Features**: By visualizing the data (see **figure_1.png** and **figure_2.png**), we notice some overlap in the accept/refuse decisions based on distance and price. Adding features like the day and time of the job might improve the results. For instance, did professionals accept jobs on weekends, after 6 PM, or during rush hours?

### Improvement of Code
1. **Error Handling**: A more robust error-handling system is required.
2. **Unit Tests**: More detailed unit tests should be implemented.
3. **Logging**: Using a logger system instead of a simple verbose mode.
4. **Monitoring**: Throwing an alert when the quality of the results fell under a certain threshold (e.g. precision <= 0.7).
