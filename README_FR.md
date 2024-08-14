
## Test de Data Scientist WeCaSa

L'objectif de ce test est de prédire si un professionnel acceptera une nouvelle proposition de travail en fonction de la distance et du prix de la proposition.

La solution est divisée en deux parties :
    1. Un notebook Jupyter où je montre l'approche générale pour résoudre le problème.
    2. Un code structuré qui simule un scénario de production, permettant d'entraîner et de tester le modèle séparément.

### Code Requirements
En utilisant de Python 3.12.3
```
pip install pandas==2.2.2
pip install numpy==1.26.4
pip install matplotlib==3.9.1
pip install seaborn==0.13.2
pip install geopy==2.4.1
pip install scikit-learn==1.4.2
```

### Code du Notebook Jupyter
Dans **core.ipynb**, une solution étape par étape est donnée :
1. Charger les données et identifier les lignes redondantes et NaN.
 2. Calcul de la distance (en KM).
 3. Effectuer **feature engineering** en ajoutant une troisième feature: Euro par KM.
 4. Visualizer le jeu de données par rapport à la valeur cible (si un travail a été accepté ou non).
 5. Entrainer et comparer différents modèles d'apprentissage automatique en termes de précision, de rappel, etc.
 6. Expliquer les résultats et discuter des améliorations potentielles.

### Solution Structurée
Le code est organisé comme suit :
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
1. **data** : Le fichier **test.txt** simule de nouveaux cas à tester, y compris une proposition de travail de 100 euros à 10 KM de distance. **predictions.csv** stocke les résultats de l'exécution du modèle sur un nouveau jeu de test (**test.txt**).
2. model : Contient le modèle ajusté et le normaliseur.
3. core : **data_processor.py** inclut l'étape de prétraitement (description, nettoyage et visualisation des données). **model.py** inclut les différentes étapes pour entraîner, valider, sauvegarder et charger un modèle. Enfin, **utils.py** inclut une fonction qui applique des poids sur le jeu de données.
4. **train.py** : Entraîne un modèle sur un jeu de données avec différents paramètres.
5. **test.py** : Teste le modèle pré-entraîné sur un nouveau jeu de données.
6. **unitest.py** : Un test unitaire pour s'assurer que le code fonctionne correctement.

### Entraînement d'un Nouveau Modèle
Utilisez **train.py** pour entraîner un nouveau modèle avec les options suivantes :
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

Où 'dt' représente DecisionTreeClassifier, 'lr' représente LogisticRegression, 'knn' représente KNearestNeighbors, et 'gbt' représente GradientBoostingClassifier.

En exécutant **train.py** avec les paramètres par défaut :
```
python train.py
```

Donne la sortie suivante :
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
Le meilleur modèle sera sauvegardé dans **model/best_model.pkl**, et le normaliseur sera sauvegardé avec. Ce modèle peut ensuite être utilisé pour tester sur un nouveau jeu de données. Vous pouvez également expérimenter avec différents modèles et paramètres, comme un arbre de décision avec une profondeur maximale de 4.


### Test du Modèle
Pour tester le modèle, nous pouvons utiliser le script **test.py**. Les paramètres d'entrée sont les suivants :
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

Et si nous exécutons **test.py** avec les paramètres par défaut, et le jeu de test qui est dans **data/test.txt** qui contient une proposition de travail de 100 euros à 10 KM de distance et trois autres cas aléatoires, et en utilisant le modèle que nous avons appris précédemment, nous obtenons :

```
- Predicting test data classes ...
- Predictions saved to data/predictions.csv
```

Les résultats seront sauvegardés par défaut dans  **predictions.csv**:
```
Prediction,Probability_class_0,Probability_class_1  
1,0.0,1.0  			
0,0.8,0.2  
1,0.0,1.0  
1,0.0,1.0
```
La première ligne (proposition de 100 euros, 10 KM de distance) suggère d'accepter la proposition avec une confiance de 100%. La deuxième ligne (proposition de 2 euros, 16 KM de distance) suggère de ne pas accepter la proposition avec une confiance de 80%, et ainsi de suite.

### Amélioration des Résultats
1. **Déséquilibre des Données** : Les données sont déséquilibrées. Des techniques d'équilibrage telles que les k-medoids pourraient réduire le nombre de travaux "accepts" et équilibrer les données. Cependant, il faut faire attention à ne pas perdre d'informations statistiques importantes.
    
2. **Additional Features** : En visualisant les données (voir **figure_1.png** et **figure_2.png**), nous remarquons un certain chevauchement dans les décisions d'acceptation/refus en fonction de la distance et du prix. L'ajout de features comme le jour et l'heure du travail pourrait améliorer les résultats. Par exemple, les professionnels ont-ils accepté des travaux le week-end, après 18h, ou pendant les heures de pointe ?


### Amélioration du Code
1. **Gestion des Erreurs** : Un système de gestion des erreurs plus robuste est nécessaire.
2. **Tests Unitaires** : Des tests unitaires plus détaillés devraient être implémentés.
3. **Logging** : Utiliser un système de '*logging*' au lieu d'un simple mode '*verbose*'.
4. **Monitoring** : Lancer une alerte lorsque la qualité des résultats tombe en dessous d'un certain seuil (par exemple, précision <= 0,7).