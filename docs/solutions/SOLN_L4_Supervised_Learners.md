<a href="https://colab.research.google.com/github/wesleybeckner/data_science_foundations/blob/main/notebooks/exercises/E4_Supervised_Learners.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Data Science Foundations <br> Lab 4: Practice with Supervised Learners

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com

---

<br>

In this lab we will continue to practice creation of pipelines, feature engineering, and applying learning algorithms.

Now that we have covered supervised learning methods, and we've covered Grid Search, we will use these tools to do a sophisticated, search of hyperparameter optimization.

<br>

---





```python
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import random
import scipy.stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import seaborn as sns; sns.set()
import graphviz 
from sklearn.metrics import accuracy_score
from ipywidgets import interact, interactive, widgets
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
```


```python
wine = pd.read_csv("https://raw.githubusercontent.com/wesleybeckner/"\
      "ds_for_engineers/main/data/wine_quality/winequalityN.csv")
# infer str cols
str_cols = list(wine.select_dtypes(include='object').columns)

#set target col
target = 'quality'

enc = OneHotEncoder()
imp = SimpleImputer()

enc.fit_transform(wine[str_cols])
X_cat = enc.transform(wine[str_cols]).toarray()
X = wine.copy()
[X.pop(i) for i in str_cols]
y = X.pop(target)
X = imp.fit_transform(X)
X = np.hstack([X_cat, X])

cols = [i.split("_")[1] for i in enc.get_feature_names_out()]
cols += list(wine.columns)
cols.remove(target)
[cols.remove(i) for i in str_cols]

scaler = StandardScaler()
X[:,2:] = scaler.fit_transform(X[:,2:])

wine = pd.DataFrame(X, columns=cols)
wine[target] = y
```

to compare, here is our results performing classification on this set of data with just logistic regression:


```python
model = LogisticRegression(max_iter=1e4)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


```python
print(classification_report(y_test, y_pred, zero_division=0))
```

                  precision    recall  f1-score   support
    
               3       0.00      0.00      0.00         2
               4       0.60      0.07      0.12        46
               5       0.58      0.61      0.59       420
               6       0.52      0.68      0.59       579
               7       0.44      0.19      0.26       221
               8       0.00      0.00      0.00        32
    
        accuracy                           0.54      1300
       macro avg       0.36      0.26      0.26      1300
    weighted avg       0.51      0.54      0.50      1300
    



```python
fig, ax = plt.subplots(1, 1, figsize = (8,7))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, ax=ax)
```




    <AxesSubplot:>




    
![png](SOLN_L4_Supervised_Learners_files/SOLN_L4_Supervised_Learners_7_1.png)
    


## 🏎️ Q1:

Evaluate the performance of a Random Forest on classifying wine quality



```python
# Code Cell for L1 Q2
model = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


```python
print(classification_report(y_test, y_pred, zero_division=0))
```

                  precision    recall  f1-score   support
    
               3       0.00      0.00      0.00         2
               4       1.00      0.15      0.26        46
               5       0.72      0.77      0.75       420
               6       0.67      0.78      0.72       579
               7       0.71      0.51      0.59       221
               8       1.00      0.22      0.36        32
    
        accuracy                           0.70      1300
       macro avg       0.68      0.41      0.45      1300
    weighted avg       0.71      0.70      0.68      1300
    



```python
fig, ax = plt.subplots(1, 1, figsize = (8,7))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, ax=ax)
```




    <AxesSubplot:>




    
![png](SOLN_L4_Supervised_Learners_files/SOLN_L4_Supervised_Learners_11_1.png)
    


## 🔬 Q2:

Do a grid search to optimize your Random Forest model, use whatever hyperparameters you would like




```python
RandomForestClassifier().get_params()
```




    {'bootstrap': True,
     'ccp_alpha': 0.0,
     'class_weight': None,
     'criterion': 'gini',
     'max_depth': None,
     'max_features': 'auto',
     'max_leaf_nodes': None,
     'max_samples': None,
     'min_impurity_decrease': 0.0,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'n_estimators': 100,
     'n_jobs': None,
     'oob_score': False,
     'random_state': None,
     'verbose': 0,
     'warm_start': False}




```python
# Code Cell for L1 Q3

from sklearn.model_selection import GridSearchCV

param_grid = {'bootstrap': [True, False],
              'criterion': ['gini', 'entropy'],
              'min_samples_split': [2, 4, 6],
              'min_samples_leaf': [1, 3, 5],
              'max_features': ['auto', 'sqrt', 'log2'],
              'class_weight': ['balanced', 'balanced_subsample', None]}

grid = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid, cv=7)
```


```python
grid.fit(X_train, y_train)
print(grid.best_params_)
```

    /home/wbeckner/anaconda3/envs/py39/lib/python3.9/site-packages/sklearn/model_selection/_split.py:676: UserWarning: The least populated class in y has only 5 members, which is less than n_splits=7.
      warnings.warn(


    {'bootstrap': True, 'class_weight': None, 'criterion': 'entropy', 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2}



```python
model = grid.best_estimator_
```


```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
fig, ax = plt.subplots(1, 1, figsize = (8,7))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, ax=ax)
```

                  precision    recall  f1-score   support
    
               3       0.00      0.00      0.00         2
               4       0.75      0.13      0.22        46
               5       0.70      0.76      0.73       420
               6       0.66      0.78      0.72       579
               7       0.73      0.48      0.58       221
               8       1.00      0.25      0.40        32
    
        accuracy                           0.69      1300
       macro avg       0.64      0.40      0.44      1300
    weighted avg       0.70      0.69      0.67      1300
    





    <AxesSubplot:>




    
![png](SOLN_L4_Supervised_Learners_files/SOLN_L4_Supervised_Learners_17_2.png)
    

