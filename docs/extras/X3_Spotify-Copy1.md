## What makes a playlist successful?

**Analysis**

* Simple metric (dependent variable)
    * mau
    * mau_previous_month
    * mau_both_months
    * monthly_stream30s
    * stream30s
* Design metric (dependent variable)
    * 30s listens/tot listens (listen conversions)
    * Users both months/users prev month (user conversions)
    * Best small time performers (less than X total monthly listens + high conversion)
    * Best new user playlist (owner has only 1 popular playlist)
* Define "top"
    * Top 10%
        * mau_previous_month: 9.0
        * mau_both_months: 2.0
        * mau: 9.0
        * monthly_stream30s: 432.0
        * stream30s: 17.0
    * Top 1%
        * mau_previous_month: 130.0
        * mau_both_months: 19.0
        * mau: 143.0
        * monthly_stream30s: 2843.0
        * stream30s: 113.0
* Independent variables
    * moods and genres (categorical)
    * number of tracks, albums, artists, and local tracks (continuous)
        
The analysis will consist of:

1. understand the distribution characteristics of the dependent and independent variables
2. quantify the dependency of the dependent/independent variables for each of the simple and design metrics
    1. chi-square test
    2. bootstrap/t-test

**Key Conclusions**

for the simple metrics, key genres and moods were **Romantic, Latin, Children's, Lively, Traditional, and Jazz**. Playlists that included these genres/moods had a positive multiplier effect (usually in the vicinicty of 2x more likely) on the key simple metric (i.e. playlists with latin as a primary genre were 2.5x more likely to be in the top 10% of streams longer than 30 seconds)

skippers - is this associated with the current playlist 

| Column Name             | Description                                                                                              |
|-------------------------|----------------------------------------------------------------------------------------------------------|
| playlist_uri            | The key, Spotify uri of the playlist                                                                     |
| owner                   | Playlist owner, Spotify username                                                                         |
| streams                 | Number of streams from the playlist today                                                                |
| stream30s               | Number of streams over 30 seconds from playlist today                                                    |
| dau                     | Number of Daily Active Users, i.e. users with a stream over 30 seconds from playlist today               |
| wau                     | Number of Weekly Active Users, i.e. users with a stream over 30 seconds from playlist in past week       |
| mau                     | Number of Monthly Active Users, i.e. users with a stream over 30 seconds from playlist in the past month |
| mau_previous_months     | Number of Monthly Active users in the month prior to this one                                            |
| mau_both_months         | Number of users that were active on the playlist both this and the previous month                        |
| users                   | Number of users streaming (all streams) from this playlist this month                                    |
| skippers                | Number of users who skipped more than 90 percent of their streams today                                  |
| owner_country           | Country of the playlist owner                                                                            |
| n_tracks                | Number of tracks in playlist                                                                             |
| n_local_tracks          | Change in number of tracks on playlist since yesterday                                                   |
| n_artists               | Number of unique artists in playlist                                                                     |
| n_albums                | Number of unique albums in playlist                                                                      |
| monthly_stream30s       | Number of streams over 30 seconds this month                                                             |
| monthly_owner_stream30s | Number of streams over 30 seconds by playlist owner this month                                           |
| tokens                  | List of playlist title tokens, stopwords and punctuation removed                                         |
| genre_1                 | No. 1 Genre by weight of playlist tracks, from Gracenote metadata                                        |
| genre_2                 | No. 2 Genre by weight of playlist tracks, from Gracenote metadata                                        |
| genre_3                 | No. 3 Genre by weight of playlist tracks, from Gracenote metadata                                        |
| mood_1                  | No. 1 Mood by weight of playlist tracks, from Gracenote metadata                                         |
| mood_2                  | No. 2 Mood by weight of playlist tracks, from Gracenote metadata                                         |
| mood_3                  | No. 3 Mood by weight of playlist tracks, from Gracenote metadata                                         |

## Imports


```python
# basic packages
import pandas as pd
pd.set_option('display.max_columns', 500)
import numpy as np
import random
import copy

# visualization packages
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns; sns.set()
import graphviz 

# stats packages
import scipy.stats as stats
from scipy.spatial.distance import cdist
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.outliers_influence import variance_inflation_factor

# sklearn preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight

# sklearn modeling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.mixture import GaussianMixture

# sklearn evaluation
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score

```


```python
df = pd.read_csv("../../data/playlist_summary_external-4.txt", delimiter='\t')
```

#### Putting it All Together


```python
sub_targets = ['mau_previous_month', 'mau_both_months', 'monthly_stream30s', 'stream30s']
des_features = ['mood_1', 'mood_2', 'mood_3', 'genre_1', 'genre_2', 'genre_3']
```

#### Models (Multi-Feature Analysis)

##### Deciles - Random Forest


```python
sub_targets
```


```python
target = sub_targets[-2]
```


```python
y = df[target].values
labels = y.copy()
names = []
for idx, quant in zip(range(11), np.linspace(0, 1, num=11)):
    if idx == 0:
        prev = quant
        continue
    if idx == 1:
        labels[labels <= np.quantile(y, quant)] = idx
        names += [f"less than {np.quantile(y, quant):.0f} listens"]
    else:
        labels[(labels > np.quantile(y, prev))
              &(labels <= np.quantile(y, quant))] = idx
        names += [f"{np.quantile(y, prev):.0f} < listens <= {np.quantile(y, quant):.0f}"]
    prev = quant
y = labels
```


```python
names
```


```python
X = df[des_features + con_features]
enc = OneHotEncoder()
std = StandardScaler()

X_cat = enc.fit_transform(X[des_features]).toarray()
X_con = std.fit_transform(X[con_features])
X = np.hstack((X_con, X_cat))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)
```


```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
```


```python
y_hat_test = model.predict(X_test)
print(f"Train Acc: {accuracy_score(y_test, y_hat_test):.2f}")
print(f"Test Acc: {accuracy_score(y_test, y_hat_test):.2f}")
```


```python
print(classification_report(y_test, y_hat_test, zero_division=0))
```


```python
fig, ax = plt.subplots(1, 1, figsize = (10, 10))
sns.heatmap(confusion_matrix(y_test, y_hat_test), annot=True, ax=ax, xticklabels=names, yticklabels=names)
```


```python
# grab feature importances
imp = model.feature_importances_

# their std
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

# build feature names
feature_names = con_features + list(enc.get_feature_names_out())

# create new dataframe
feat = pd.DataFrame([feature_names, imp, std]).T
feat.columns = ['feature', 'importance', 'std']
feat = feat.sort_values('importance', ascending=False)
feat = feat.reset_index(drop=True)
feat.dropna(inplace=True)
feat.head(20)
```

##### Quartiles - Random Forest


```python
### Create Categories

y = df[target].values
labels = y.copy()
names = []
lim = 5
for idx, quant in zip(range(lim), np.linspace(0, 1, num=lim)):
    if idx == 0:
        prev = quant
        continue
    if idx == 1:
        labels[labels <= np.quantile(y, quant)] = idx
        names += [f"less than {np.quantile(y, quant):.0f} listens"]
    else:
        labels[(labels > np.quantile(y, prev))
              &(labels <= np.quantile(y, quant))] = idx
        names += [f"{np.quantile(y, prev):.0f} < listens <= {np.quantile(y, quant):.0f}"]
    prev = quant
y = labels

### Create Training Data

X = df[des_features + con_features]
enc = OneHotEncoder()
std = StandardScaler()

X_cat = enc.fit_transform(X[des_features]).toarray()
X_con = std.fit_transform(X[con_features])
X = np.hstack((X_con, X_cat))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)

### Train Model

model = RandomForestClassifier()
model.fit(X_train, y_train)

### Asses Performance

y_hat_test = model.predict(X_test)
y_hat_train = model.predict(X_train)
print(f"Train Acc: {accuracy_score(y_train, y_hat_train):.2f}")
print(f"Test Acc: {accuracy_score(y_test, y_hat_test):.2f}")

print(classification_report(y_test, y_hat_test, zero_division=0))

fig, ax = plt.subplots(1, 1, figsize = (8,7))
sns.heatmap(confusion_matrix(y_test, y_hat_test), annot=True, ax=ax)
```

##### Binary, 90th Percentile, Random Forest


```python
### Create Categories

y = df[target].values
labels = y.copy()
names = []
weights = y.copy()
weights.dtype = 'float'
lim = 5
dom_class_weight = 1 / (lim - 1 - 1)
for idx, quant in zip(range(lim), np.linspace(0, 1, num=lim)):
    if idx < lim - 2:
        prev = quant
        continue
    elif idx == lim - 2:
        weights[y <= np.quantile(y, quant)] = dom_class_weight
        labels[labels <= np.quantile(y, quant)] = idx
        names += [f"less than {np.quantile(y, quant):.0f} listens"]
        
    else:
        labels[(labels > np.quantile(y, prev))
              &(labels <= np.quantile(y, quant))] = idx
        weights[(y > np.quantile(y, prev))
              &(y <= np.quantile(y, quant))] = 1.0
        names += [f"{np.quantile(y, prev):.0f} < listens <= {np.quantile(y, quant):.0f}"]
        
    prev = quant
y = labels
```


```python
### Create Training Data

X = df[des_features + con_features]
enc = OneHotEncoder()
std = StandardScaler()

X_cat = enc.fit_transform(X[des_features]).toarray()
X_con = std.fit_transform(X[con_features])
X = np.hstack((X_con, X_cat))

X_train, X_test, y_train, y_test, weight_train, weight_test = train_test_split(X, y, weights, random_state=42, train_size=0.8)


### Strateification Code

# strat_y0_idx = np.array(random.sample(list(np.argwhere(y_train==3).reshape(-1)), np.unique(y_train, return_counts=True)[1][1]))
# strat_y1_idx = np.argwhere(y_train==4).reshape(-1)
# strat_idx = np.hstack((strat_y0_idx, strat_y1_idx))
# X_train = X_train[strat_idx]
# y_train = y_train[strat_idx]
```


```python
### Train Model

model = RandomForestClassifier()
model.fit(X_train, y_train)

### Assess Performance

y_hat_test = model.predict(X_test)
y_hat_train = model.predict(X_train)
print(f"Train Acc: {accuracy_score(y_train, y_hat_train):.2f}")
print(f"Test Acc: {accuracy_score(y_test, y_hat_test):.2f}")

print(classification_report(y_test, y_hat_test, zero_division=0))

fig, ax = plt.subplots(1, 1, figsize = (8,7))
sns.heatmap(confusion_matrix(y_test, y_hat_test), annot=True, ax=ax)
```

##### Forward Selection Model


```python
### y
print(target)
y = df[target].values
labels = y.copy()
names = []
weights = y.copy()
weights.dtype = 'float'
lim = 11
dom_class_weight = 1 / (lim - 1 - 1)
for idx, quant in zip(range(lim), np.linspace(0, 1, num=lim)):
    if idx < lim - 2:
        prev = quant
        continue
    elif idx == lim - 2:
        weights[y <= np.quantile(y, quant)] = dom_class_weight
        labels[labels <= np.quantile(y, quant)] = 0
        names += [f"less than {np.quantile(y, quant):.0f} listens"]
        
    else:
        labels[(labels > np.quantile(y, prev))
              & (labels <= np.quantile(y, quant))] = 1
        weights[(y > np.quantile(y, prev))
              & (y <= np.quantile(y, quant))] = 1.0
        names += [f"{np.quantile(y, prev):.0f} < listens <= {np.quantile(y, quant):.0f}"]
    prev = quant
y = labels

#### X

X = df[des_features + con_features]

enc = OneHotEncoder()
std = StandardScaler()

X_cat = enc.fit_transform(X[des_features]).toarray()
X_con = std.fit_transform(X[con_features])
X = np.hstack((np.ones((X_con.shape[0], 1)), X_con, X_cat))
feature_names = ['intercept'] + con_features + list(enc.get_feature_names_out())

data = pd.DataFrame(X, columns=feature_names)
print(names)
```


```python
def add_feature(feature_names, basemodel, data, y, r2max=0, model='linear', disp=0):
    feature_max = None
    bestsum = None
    newmodel = None
    for feature in feature_names:
        basemodel[feature] = data[feature]
        X2 = basemodel.values
        est = Logit(y, X2)
        est2 = est.fit(disp=0)
        summ = est2.summary()
        score = float(str(pd.DataFrame(summ.tables[0]).loc[3, 3]))
        if (score > r2max) and not (est2.pvalues > cutoff).any():
            r2max = score
            feature_max = feature
            bestsum = est2.summary()
            newmodel = basemodel.copy()
            if disp == 1:
                print(f"new r2max, {feature_max}, {r2max}")
        basemodel.drop(labels = feature, axis = 1, inplace = True)
    return r2max, feature_max, bestsum, newmodel
```


```python
candidates = feature_names.copy()
basemodel = pd.DataFrame()
r2max = 0
```


```python
with open("canidates.txt", "w+") as f:
    file_data = f.read()
    for i in candidates:
        f.write(f"{i}\n")
```


```python
basemodel.to_csv("basemodel.csv")
```


```python
with open("canidates.txt", "r") as f:
    # file_data = f.read()
    new = []
    for line in f:
        current_place = line[:-1]
        new.append(current_place)
```


```python
new = pd.read_csv("basemodel.csv", index_col=0)
```


```python
with open("fwd_selection_results.txt", "r+") as f:
    for line in f:
        pass
    lastline = line[:-1]
    stuff = lastline.split(", ")
    new = float(stuff[-1])

```


```python
new
```


```python
while True:
    newr2max, feature_max, bestsum, newmodel = add_feature(
        feature_names=candidates, 
        basemodel=basemodel, 
        data=data, 
        y=y,
        r2max=r2max)    
    if newr2max > r2max:
        r2max = newr2max
        print(f"new r2max, {feature_max}, {r2max}")
        with open("fwd_selection_results.txt", "a+") as f:
            file_data = f.read()
            f.write(f"new r2max, {feature_max}, {r2max}\n")
        candidates.remove(feature_max)
        with open("canidates.txt", "w+") as f:
            file_data = f.read()
            for i in candidates:
                f.write(f"{i}\n")
        basemodel = newmodel
        basemodel.to_csv("basemodel.csv")
        continue
    else:
        break
```


```python
X2 = basemodel.values
est = Logit(y, X2)
est2 = est.fit(disp=0)
summ = est2.summary()

res_table = summ.tables[1]
res_df = pd.DataFrame(res_table.data)
cols = res_df.iloc[0]
cols = [str(i) for i in cols]
res_df.drop(0, axis=0, inplace=True)
res_df.set_index(0, inplace=True)
res_df.columns = cols[1:]
res_df.index = basemodel.columns
res_df
```

##### Binary, 99th Percentile


```python
### Create Categories

y = df[target].values
labels = y.copy()
names = []
weights = y.copy()
weights.dtype = 'float'
lim = 11
dom_class_weight = 1 / (lim - 1 - 1)
for idx, quant in zip(range(lim), np.linspace(0, 1, num=lim)):
    if idx < lim - 2:
        prev = quant
        continue
    elif idx == lim - 2:
        weights[y <= np.quantile(y, quant)] = dom_class_weight
        labels[labels <= np.quantile(y, quant)] = idx
        names += [f"less than {np.quantile(y, quant):.0f} listens"]
        
    else:
        labels[(labels > np.quantile(y, prev))
              &(labels <= np.quantile(y, quant))] = idx
        weights[(y > np.quantile(y, prev))
              &(y <= np.quantile(y, quant))] = 1.0
        names += [f"{np.quantile(y, prev):.0f} < listens <= {np.quantile(y, quant):.0f}"]
        
    prev = quant
y = labels

### Create Training Data

X = df[des_features + con_features]
enc = OneHotEncoder()
std = StandardScaler()

X_cat = enc.fit_transform(X[des_features]).toarray()
X_con = std.fit_transform(X[con_features])
X = np.hstack((X_con, X_cat))

X_train, X_test, y_train, y_test, weight_train, weight_test = train_test_split(X, y, weights, random_state=42, train_size=0.8)

### Train Model

model = RandomForestClassifier()
model.fit(X_train, y_train, weight_train)

### Asses Performance

y_hat_test = model.predict(X_test)
y_hat_train = model.predict(X_train)
print(f"Train Acc: {accuracy_score(y_train, y_hat_train):.2f}")
print(f"Test Acc: {accuracy_score(y_test, y_hat_test):.2f}")

print(classification_report(y_test, y_hat_test, zero_division=0))

fig, ax = plt.subplots(1, 1, figsize = (8,7))
sns.heatmap(confusion_matrix(y_test, y_hat_test), annot=True, ax=ax)
```

## Other Metrics

* 30s listens/tot listens (listen conversions) _also like a bounce rate_
* Users both months/users prev month (user conversions)
    * combine with mau > mau_previous_month
* Best small time performers (less than X total monthly listens + high conversion)
* Best new user playlist (owner has only 1 popular playlist)

### Listen and User Conversions, MAU Growing


```python
df['listen_conversions'] = df['stream30s'] / df['streams']
df['listen_conversions'].fillna(value=0, inplace=True)

df['user_retention'] = df['mau_both_months'] / df['mau_previous_month']
df['user_retention'].fillna(value=0, inplace=True)

df['user_conversions'] = df['mau'] / df['users']
df['user_conversions'].fillna(value=0, inplace=True)


df['mau_growing'] = df['mau'] > df['mau_previous_month']
df['mau_growth'] = df['mau'] / df['mau_previous_month']
df['mau_growth'].fillna(value=0, inplace=True)
df['mau_growth'].replace([np.inf, -np.inf], 1, inplace=True)
```


```python
new_metrics = ['listen_conversions', 'user_conversions', 'user_retention', 'mau_growth']
```


```python
df[new_metrics].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listen_conversions</th>
      <th>user_conversions</th>
      <th>user_retention</th>
      <th>mau_growth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>403366.000000</td>
      <td>403366.000000</td>
      <td>403366.000000</td>
      <td>403366.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.334701</td>
      <td>0.724072</td>
      <td>0.571070</td>
      <td>1.513218</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.399968</td>
      <td>0.261708</td>
      <td>0.392073</td>
      <td>17.459669</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.020348</td>
      <td>0.000000</td>
      <td>0.031250</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.200000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.666667</td>
      <td>0.500000</td>
      <td>1.066667</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.730769</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7859.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['listen_conversions'].plot(kind='hist', bins=10)
```




    <AxesSubplot:ylabel='Frequency'>




    
![png](X3_Spotify-Copy1_files/X3_Spotify-Copy1_45_1.png)
    



```python
df['user_conversions'].plot(kind='hist', bins=10)
```




    <AxesSubplot:ylabel='Frequency'>




    
![png](X3_Spotify-Copy1_files/X3_Spotify-Copy1_46_1.png)
    



```python
df['user_retention'].plot(kind='hist', bins=10)
```




    <AxesSubplot:ylabel='Frequency'>




    
![png](X3_Spotify-Copy1_files/X3_Spotify-Copy1_47_1.png)
    



```python
df.loc[df['mau_growth'] < 10]['mau_growth'].plot(kind='hist', bins=20)
```




    <AxesSubplot:ylabel='Frequency'>




    
![png](X3_Spotify-Copy1_files/X3_Spotify-Copy1_48_1.png)
    



```python
df['mau_growing'].value_counts().plot(kind='bar')
```




    <AxesSubplot:>




    
![png](X3_Spotify-Copy1_files/X3_Spotify-Copy1_49_1.png)
    



```python
df['new_success'] = df[new_metrics].apply(lambda x: (x > 0.5) if (max(x) == 1) else (x > 1)).all(axis=1)
```


```python
df['new_success'].value_counts()
```




    False    362869
    True      40497
    Name: new_success, dtype: int64




```python
df.loc[df['new_success'] == True]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playlist_uri</th>
      <th>owner</th>
      <th>streams</th>
      <th>stream30s</th>
      <th>dau</th>
      <th>wau</th>
      <th>mau</th>
      <th>mau_previous_month</th>
      <th>mau_both_months</th>
      <th>users</th>
      <th>skippers</th>
      <th>owner_country</th>
      <th>n_tracks</th>
      <th>n_local_tracks</th>
      <th>n_artists</th>
      <th>n_albums</th>
      <th>monthly_stream30s</th>
      <th>monthly_owner_stream30s</th>
      <th>tokens</th>
      <th>genre_1</th>
      <th>genre_2</th>
      <th>genre_3</th>
      <th>mood_1</th>
      <th>mood_2</th>
      <th>mood_3</th>
      <th>listen_conversions</th>
      <th>user_retention</th>
      <th>user_conversions</th>
      <th>mau_growing</th>
      <th>mau_growth</th>
      <th>new_success</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>spotify:user:9a3580868994077be27d244788d494cd:...</td>
      <td>9a3580868994077be27d244788d494cd</td>
      <td>28</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>US</td>
      <td>321</td>
      <td>0</td>
      <td>170</td>
      <td>205</td>
      <td>83</td>
      <td>77</td>
      <td>["sunny", "daze"]</td>
      <td>Alternative</td>
      <td>Indie Rock</td>
      <td>Electronica</td>
      <td>Brooding</td>
      <td>Excited</td>
      <td>Sensual</td>
      <td>0.535714</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>True</td>
      <td>2.000000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>spotify:user:7abbdbd3119687473b8f2986e73e2ad6:...</td>
      <td>7abbdbd3119687473b8f2986e73e2ad6</td>
      <td>9</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>US</td>
      <td>373</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>11</td>
      <td>[]</td>
      <td>Pop</td>
      <td>Alternative</td>
      <td>Indie Rock</td>
      <td>Empowering</td>
      <td>Excited</td>
      <td>Urgent</td>
      <td>0.555556</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>True</td>
      <td>2.000000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>spotify:user:838141e861005b6a955cb389c19671a5:...</td>
      <td>838141e861005b6a955cb389c19671a5</td>
      <td>32</td>
      <td>25</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>US</td>
      <td>904</td>
      <td>0</td>
      <td>81</td>
      <td>125</td>
      <td>327</td>
      <td>253</td>
      <td>["metalcore", "forever"]</td>
      <td>Punk</td>
      <td>Metal</td>
      <td>Rock</td>
      <td>Defiant</td>
      <td>Urgent</td>
      <td>Aggressive</td>
      <td>0.781250</td>
      <td>1.0</td>
      <td>0.800000</td>
      <td>True</td>
      <td>1.333333</td>
      <td>True</td>
    </tr>
    <tr>
      <th>36</th>
      <td>spotify:user:2217942070bcaa5f1e651e27744b4402:...</td>
      <td>2217942070bcaa5f1e651e27744b4402</td>
      <td>18</td>
      <td>17</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>US</td>
      <td>141</td>
      <td>1</td>
      <td>122</td>
      <td>131</td>
      <td>567</td>
      <td>0</td>
      <td>["chill"]</td>
      <td>Rap</td>
      <td>Dance &amp; House</td>
      <td>Alternative</td>
      <td>Excited</td>
      <td>Defiant</td>
      <td>Energizing</td>
      <td>0.944444</td>
      <td>1.0</td>
      <td>0.800000</td>
      <td>True</td>
      <td>1.333333</td>
      <td>True</td>
    </tr>
    <tr>
      <th>59</th>
      <td>spotify:user:dfde15dd16b4ad87a75036276b4c9f66:...</td>
      <td>dfde15dd16b4ad87a75036276b4c9f66</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>US</td>
      <td>84</td>
      <td>0</td>
      <td>73</td>
      <td>78</td>
      <td>254</td>
      <td>239</td>
      <td>["vegas"]</td>
      <td>Rock</td>
      <td>Pop</td>
      <td>R&amp;B</td>
      <td>Upbeat</td>
      <td>Excited</td>
      <td>Empowering</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.666667</td>
      <td>True</td>
      <td>2.000000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>403329</th>
      <td>spotify:user:358b83239c6a2557fbfb053330d49a41:...</td>
      <td>358b83239c6a2557fbfb053330d49a41</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>US</td>
      <td>33</td>
      <td>0</td>
      <td>28</td>
      <td>31</td>
      <td>271</td>
      <td>32</td>
      <td>["one", "dirt", "road"]</td>
      <td>Country &amp; Folk</td>
      <td>Rock</td>
      <td>-</td>
      <td>Yearning</td>
      <td>Empowering</td>
      <td>Gritty</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>True</td>
      <td>3.000000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>403336</th>
      <td>spotify:user:a0781a2de47beb8bd693f3022f316327:...</td>
      <td>a0781a2de47beb8bd693f3022f316327</td>
      <td>856</td>
      <td>855</td>
      <td>3</td>
      <td>10</td>
      <td>10</td>
      <td>5</td>
      <td>5</td>
      <td>10</td>
      <td>0</td>
      <td>US</td>
      <td>168</td>
      <td>0</td>
      <td>6</td>
      <td>9</td>
      <td>33747</td>
      <td>1391</td>
      <td>["evning", "song"]</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.998832</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>True</td>
      <td>2.000000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>403338</th>
      <td>spotify:user:06f6dd666f1bbf9148c792b87ed4d22f:...</td>
      <td>06f6dd666f1bbf9148c792b87ed4d22f</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>US</td>
      <td>59</td>
      <td>0</td>
      <td>34</td>
      <td>46</td>
      <td>21</td>
      <td>9</td>
      <td>["rhc"]</td>
      <td>Religious</td>
      <td>Pop</td>
      <td>Alternative</td>
      <td>Empowering</td>
      <td>Upbeat</td>
      <td>Brooding</td>
      <td>0.800000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>True</td>
      <td>2.000000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>403348</th>
      <td>spotify:user:c6af258245d55221cebedb1175f08d83:...</td>
      <td>c6af258245d55221cebedb1175f08d83</td>
      <td>13</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>US</td>
      <td>31</td>
      <td>0</td>
      <td>30</td>
      <td>29</td>
      <td>208</td>
      <td>206</td>
      <td>["zumba", "val", "silva", "playlist"]</td>
      <td>Latin</td>
      <td>Pop</td>
      <td>Dance &amp; House</td>
      <td>Aggressive</td>
      <td>Excited</td>
      <td>Defiant</td>
      <td>0.846154</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>True</td>
      <td>2.000000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>403353</th>
      <td>spotify:user:5461b6b460dd512d7b4fd4fb488f3520:...</td>
      <td>5461b6b460dd512d7b4fd4fb488f3520</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>US</td>
      <td>146</td>
      <td>0</td>
      <td>115</td>
      <td>123</td>
      <td>405</td>
      <td>321</td>
      <td>["myfavorites"]</td>
      <td>Indie Rock</td>
      <td>Electronica</td>
      <td>Alternative</td>
      <td>Yearning</td>
      <td>Energizing</td>
      <td>Brooding</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>True</td>
      <td>2.000000</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>40497 rows × 31 columns</p>
</div>




```python
chidf = pd.DataFrame()
target = 'new_success'
chidf[target] = df[target]
# quant_value = 0.90
# tar_value = np.quantile(chidf[target], quant_value)
# chidf[target] = chidf[target] > tar_value
chisum = pd.DataFrame()
cutoff = 0.0001
pop = chidf[target].values

for ind in des_features:
    # ind = des_features[0]
    chidf[ind] = df[ind]

    for grp_label in df[ind].unique():
    # grp_label = df[ind].unique()[0]
        try:
            cTable = chidf.groupby(chidf[ind] == grp_label)[target].value_counts().values.reshape(2,2).T
            chi2, p, dof, ex = stats.chi2_contingency(cTable, correction=True, lambda_=None)
            ratio = cTable[1]/cTable[0]
            pos = ratio[1]/ratio[0]
            chisum = pd.concat([chisum, pd.DataFrame([[ind, grp_label, chi2, p, cTable, pos, p<cutoff]])])
        except:
            pass

chisum.columns = ['feature', 'group', 'chi', 'p-value', 'cTable', 'multiplier', 'reject null']
chisum = chisum.sort_values('p-value').reset_index(drop=True)
```


```python
chisum.loc[chisum['reject null'] == True].sort_values('multiplier', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>group</th>
      <th>chi</th>
      <th>p-value</th>
      <th>cTable</th>
      <th>multiplier</th>
      <th>reject null</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>genre_1</td>
      <td>Dance &amp; House</td>
      <td>231.225731</td>
      <td>3.221322e-52</td>
      <td>[[334768, 28101], [36487, 4010]]</td>
      <td>1.309267</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>genre_1</td>
      <td>Indie Rock</td>
      <td>386.328998</td>
      <td>5.212769e-86</td>
      <td>[[300809, 62060], [31986, 8511]]</td>
      <td>1.289733</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mood_1</td>
      <td>Excited</td>
      <td>289.821405</td>
      <td>5.438394e-65</td>
      <td>[[306376, 56493], [32871, 7626]]</td>
      <td>1.258184</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mood_1</td>
      <td>Defiant</td>
      <td>285.014998</td>
      <td>6.064223e-64</td>
      <td>[[291222, 71647], [31065, 9432]]</td>
      <td>1.234123</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>genre_2</td>
      <td>Electronica</td>
      <td>124.733558</td>
      <td>5.820843e-29</td>
      <td>[[335186, 27683], [36772, 3725]]</td>
      <td>1.226540</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>mood_1</td>
      <td>Somber</td>
      <td>30.852148</td>
      <td>2.784538e-08</td>
      <td>[[361994, 875], [40456, 41]]</td>
      <td>0.419270</td>
      <td>True</td>
    </tr>
    <tr>
      <th>0</th>
      <td>genre_3</td>
      <td>-</td>
      <td>1404.327669</td>
      <td>2.410008e-307</td>
      <td>[[324633, 38236], [38610, 1887]]</td>
      <td>0.414947</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>genre_2</td>
      <td>-</td>
      <td>861.809401</td>
      <td>1.968786e-189</td>
      <td>[[342541, 20328], [39619, 878]]</td>
      <td>0.373430</td>
      <td>True</td>
    </tr>
    <tr>
      <th>24</th>
      <td>mood_1</td>
      <td>Other</td>
      <td>81.806778</td>
      <td>1.500630e-19</td>
      <td>[[361232, 1637], [40439, 58]]</td>
      <td>0.316494</td>
      <td>True</td>
    </tr>
    <tr>
      <th>42</th>
      <td>genre_1</td>
      <td>Spoken &amp; Audio</td>
      <td>58.779116</td>
      <td>1.764037e-14</td>
      <td>[[361755, 1114], [40460, 37]]</td>
      <td>0.296965</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 7 columns</p>
</div>




```python
chisum.loc[chisum['reject null'] == True].sort_values('multiplier', ascending=True)[:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>group</th>
      <th>chi</th>
      <th>p-value</th>
      <th>cTable</th>
      <th>multiplier</th>
      <th>reject null</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>genre_1</td>
      <td>Spoken &amp; Audio</td>
      <td>58.779116</td>
      <td>1.764037e-14</td>
      <td>[[361755, 1114], [40460, 37]]</td>
      <td>0.296965</td>
      <td>True</td>
    </tr>
    <tr>
      <th>24</th>
      <td>mood_1</td>
      <td>Other</td>
      <td>81.806778</td>
      <td>1.500630e-19</td>
      <td>[[361232, 1637], [40439, 58]]</td>
      <td>0.316494</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>genre_2</td>
      <td>-</td>
      <td>861.809401</td>
      <td>1.968786e-189</td>
      <td>[[342541, 20328], [39619, 878]]</td>
      <td>0.373430</td>
      <td>True</td>
    </tr>
    <tr>
      <th>0</th>
      <td>genre_3</td>
      <td>-</td>
      <td>1404.327669</td>
      <td>2.410008e-307</td>
      <td>[[324633, 38236], [38610, 1887]]</td>
      <td>0.414947</td>
      <td>True</td>
    </tr>
    <tr>
      <th>70</th>
      <td>mood_1</td>
      <td>Somber</td>
      <td>30.852148</td>
      <td>2.784538e-08</td>
      <td>[[361994, 875], [40456, 41]]</td>
      <td>0.419270</td>
      <td>True</td>
    </tr>
    <tr>
      <th>73</th>
      <td>genre_1</td>
      <td>Easy Listening</td>
      <td>30.613123</td>
      <td>3.149562e-08</td>
      <td>[[361984, 885], [40455, 42]]</td>
      <td>0.424642</td>
      <td>True</td>
    </tr>
    <tr>
      <th>40</th>
      <td>mood_2</td>
      <td>-</td>
      <td>60.796108</td>
      <td>6.330294e-15</td>
      <td>[[361087, 1782], [40411, 86]]</td>
      <td>0.431224</td>
      <td>True</td>
    </tr>
    <tr>
      <th>43</th>
      <td>mood_1</td>
      <td>-</td>
      <td>57.600397</td>
      <td>3.211607e-14</td>
      <td>[[361161, 1708], [40414, 83]]</td>
      <td>0.434269</td>
      <td>True</td>
    </tr>
    <tr>
      <th>37</th>
      <td>mood_3</td>
      <td>-</td>
      <td>64.489845</td>
      <td>9.703118e-16</td>
      <td>[[360957, 1912], [40404, 93]]</td>
      <td>0.434536</td>
      <td>True</td>
    </tr>
    <tr>
      <th>48</th>
      <td>genre_1</td>
      <td>Children's</td>
      <td>52.188042</td>
      <td>5.043231e-13</td>
      <td>[[361298, 1571], [40420, 77]]</td>
      <td>0.438111</td>
      <td>True</td>
    </tr>
    <tr>
      <th>32</th>
      <td>mood_1</td>
      <td>Easygoing</td>
      <td>72.784800</td>
      <td>1.445861e-17</td>
      <td>[[360451, 2418], [40371, 126]]</td>
      <td>0.465255</td>
      <td>True</td>
    </tr>
    <tr>
      <th>56</th>
      <td>mood_3</td>
      <td>Serious</td>
      <td>43.083601</td>
      <td>5.245004e-11</td>
      <td>[[361404, 1465], [40420, 77]]</td>
      <td>0.469948</td>
      <td>True</td>
    </tr>
    <tr>
      <th>59</th>
      <td>genre_2</td>
      <td>Other</td>
      <td>41.614387</td>
      <td>1.111721e-10</td>
      <td>[[361446, 1423], [40422, 75]]</td>
      <td>0.471283</td>
      <td>True</td>
    </tr>
    <tr>
      <th>82</th>
      <td>mood_2</td>
      <td>Other</td>
      <td>25.423296</td>
      <td>4.603257e-07</td>
      <td>[[361970, 899], [40449, 48]]</td>
      <td>0.477800</td>
      <td>True</td>
    </tr>
    <tr>
      <th>60</th>
      <td>genre_1</td>
      <td>Traditional</td>
      <td>39.228043</td>
      <td>3.770852e-10</td>
      <td>[[361402, 1467], [40416, 81]]</td>
      <td>0.493733</td>
      <td>True</td>
    </tr>
    <tr>
      <th>39</th>
      <td>genre_3</td>
      <td>Easy Listening</td>
      <td>61.357952</td>
      <td>4.758655e-15</td>
      <td>[[360552, 2317], [40368, 129]]</td>
      <td>0.497272</td>
      <td>True</td>
    </tr>
    <tr>
      <th>47</th>
      <td>genre_2</td>
      <td>Easy Listening</td>
      <td>53.106215</td>
      <td>3.159911e-13</td>
      <td>[[360858, 2011], [40385, 112]]</td>
      <td>0.497648</td>
      <td>True</td>
    </tr>
    <tr>
      <th>65</th>
      <td>mood_2</td>
      <td>Stirring</td>
      <td>34.226638</td>
      <td>4.905289e-09</td>
      <td>[[361548, 1321], [40423, 74]]</td>
      <td>0.501033</td>
      <td>True</td>
    </tr>
    <tr>
      <th>57</th>
      <td>mood_1</td>
      <td>Serious</td>
      <td>42.044137</td>
      <td>8.923632e-11</td>
      <td>[[361247, 1622], [40406, 91]]</td>
      <td>0.501590</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>genre_1</td>
      <td>Soundtrack</td>
      <td>169.038371</td>
      <td>1.200050e-38</td>
      <td>[[356345, 6524], [40127, 370]]</td>
      <td>0.503642</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(2, 2, figsize=(10,10), sharex='col', sharey='row')

ind_feature = 'genre_1'
target = 'new_success'

genre_list = chisum.loc[(chisum['feature'] == ind_feature)
                       & (chisum['reject null'] == True)].sort_values('multiplier', ascending=False)['group'].values



chisum.loc[(chisum['feature'] == ind_feature)
                       & (chisum['reject null'] == True)].sort_values('multiplier', ascending=False).to_excel('table2.xlsx')

dff = pd.DataFrame(df.groupby([ind_feature])[target].value_counts(sort=False))
dff.columns = ['percent']
dff = dff.reset_index()
dff.loc[dff[target] == True, 'percent'] = dff.loc[dff[target] == True, 'percent'] / dff.loc[dff[target] == True, 'percent'].sum() 
dff.loc[dff[target] == False, 'percent'] = dff.loc[dff[target] == False, 'percent'] / dff.loc[dff[target] == False, 'percent'].sum() 
dff = dff.set_index(ind_feature).loc[genre_list,:]
dff = dff.reset_index()

sns.barplot(data=dff.iloc[:10,:], hue=target, y=ind_feature, x='percent', ax=ax[0,0])
ax[0,0].set_title('Best  and Worst Genres, Percent')
ax[0,0].set_ylabel('')
ax[0,0].set_xlabel('')
sns.barplot(data=dff.iloc[-10:,:], hue=target, y=ind_feature, x='percent', ax=ax[1,0])
# ax[1,0].set_title('Worst Primary Genres')
ax[1,0].set_ylabel('')

dff = pd.DataFrame(df.groupby([ind_feature])[target].value_counts(sort=False))
dff.columns = ['count']
dff = dff.reset_index()
# dff.loc[dff[target] == True, 'percent'] = dff.loc[dff[target] == True, 'percent'] / dff.loc[dff[target] == True, 'percent'].sum() 
# dff.loc[dff[target] == False, 'percent'] = dff.loc[dff[target] == False, 'percent'] / dff.loc[dff[target] == False, 'percent'].sum() 
dff = dff.set_index(ind_feature).loc[genre_list,:]
dff = dff.reset_index()

# fix, ax = plt.subplots(2, 2, figsize=(10,10))
sns.barplot(data=dff.iloc[:10,:], hue=target, y=ind_feature, x='count', ax=ax[0,1])
ax[0,1].set_title('Best and Worst Genres, Count')
ax[0,1].set_ylabel('')
ax[0,1].set_xlabel('')
sns.barplot(data=dff.iloc[-10:,:], hue=target, y=ind_feature, x='count', ax=ax[1,1])
# ax[1,1].set_title('Worst Primary Genres')
ax[1,1].set_ylabel('')
plt.tight_layout()

ax[0,0].get_legend().remove()
ax[1,1].get_legend().remove()
ax[1,0].get_legend().remove()
ax[0,1].legend(framealpha=1, facecolor='white', title="Success")

fig.savefig("new_categorical_dependency.svg")
```


    
![png](X3_Spotify-Copy1_files/X3_Spotify-Copy1_56_0.png)
    



```python
fig, ax = plt.subplots(2, 2, figsize=(10,10), sharex='col', sharey='row')
ind_feature = 'mood_1'
target = 'new_success'

genre_list = chisum.loc[(chisum['feature'] == ind_feature)
                       & (chisum['reject null'] == True)].sort_values('multiplier', ascending=False)['group'].values

chisum.loc[(chisum['feature'] == ind_feature)
                       & (chisum['reject null'] == True)].sort_values('multiplier', ascending=False).to_excel('table.xlsx')

dff = pd.DataFrame(df.groupby([ind_feature])[target].value_counts(sort=False))
dff.columns = ['percent']
dff = dff.reset_index()
dff.loc[dff[target] == True, 'percent'] = dff.loc[dff[target] == True, 'percent'] / dff.loc[dff[target] == True, 'percent'].sum() 
dff.loc[dff[target] == False, 'percent'] = dff.loc[dff[target] == False, 'percent'] / dff.loc[dff[target] == False, 'percent'].sum() 
dff = dff.set_index(ind_feature).loc[genre_list,:]
dff = dff.reset_index()

sns.barplot(data=dff.iloc[:10,:], hue=target, y=ind_feature, x='percent', ax=ax[0,0])
ax[0,0].set_title('Best  and Worst Genres, Percent')
ax[0,0].set_ylabel('')
ax[0,0].set_xlabel('')
sns.barplot(data=dff.iloc[-10:,:], hue=target, y=ind_feature, x='percent', ax=ax[1,0])
# ax[1,0].set_title('Worst Primary Genres')
ax[1,0].set_ylabel('')

dff = pd.DataFrame(df.groupby([ind_feature])[target].value_counts(sort=False))
dff.columns = ['count']
dff = dff.reset_index()
# dff.loc[dff[target] == True, 'percent'] = dff.loc[dff[target] == True, 'percent'] / dff.loc[dff[target] == True, 'percent'].sum() 
# dff.loc[dff[target] == False, 'percent'] = dff.loc[dff[target] == False, 'percent'] / dff.loc[dff[target] == False, 'percent'].sum() 
dff = dff.set_index(ind_feature).loc[genre_list,:]
dff = dff.reset_index()

# fix, ax = plt.subplots(2, 2, figsize=(10,10))
sns.barplot(data=dff.iloc[:10,:], hue=target, y=ind_feature, x='count', ax=ax[0,1])
ax[0,1].set_title('Best and Worst Genres, Count')
ax[0,1].set_ylabel('')
ax[0,1].set_xlabel('')
sns.barplot(data=dff.iloc[-10:,:], hue=target, y=ind_feature, x='count', ax=ax[1,1])
# ax[1,1].set_title('Worst Primary Genres')
ax[1,1].set_ylabel('')
plt.tight_layout()

ax[0,0].get_legend().remove()
ax[1,1].get_legend().remove()
ax[1,0].get_legend().remove()
ax[0,1].legend(framealpha=1, facecolor='white', title="Success")

fig.savefig("new_categorical_dependency_mood.svg")
```


    
![png](X3_Spotify-Copy1_files/X3_Spotify-Copy1_57_0.png)
    



```python
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2, figsize=(10,10))
con_features = ['n_albums', 'n_artists', 'n_tracks', 'n_local_tracks']
chidf = pd.DataFrame()
target = "new_success"
chidf[target] = df[target]
welchsum = pd.DataFrame()
cutoff = 0.0001
pop = chidf[target].values

for ind, ax in zip(con_features, [ax1, ax2, ax3, ax4]):
    chidf[ind] = df[ind]
    a = []
    b = []
    for i in range(100):
        boot1 = random.sample(
                    list(
                        chidf.loc[
                            (chidf[target] == True)
                        ][ind].values),
                    k=1000)
        boot2 = random.sample(
                    list(
                        chidf.loc[
                            (chidf[target] == False)
                        ][ind].values),
                    k=1000)
        a.append(np.mean(boot1))
        b.append(np.mean(boot2))
    testt, p = stats.ttest_ind(a, b, equal_var=False)
    a_avg = np.mean(a)
    b_avg = np.mean(b)
    welchsum = pd.concat([welchsum, pd.DataFrame([[ind, testt, p, a_avg, b_avg, p<cutoff]])])
    sns.histplot(a, color='tab:orange', label=f"{target} == True", ax=ax)
    sns.histplot(b, label=f"{target} == False", ax=ax)
    ax.set_title(ind)

welchsum.columns = ['feature', 'test stat', 'p-value', 'upper q avg', 'lower q avg', 'reject null']
welchsum = welchsum.sort_values('p-value').reset_index(drop=True)

welchsum.to_excel("new_ttest.xlsx")
ax.legend()
fig.savefig("new_ttest.svg")
```


    
![png](X3_Spotify-Copy1_files/X3_Spotify-Copy1_58_0.png)
    



```python
chidf = pd.DataFrame()
target = "success"
chidf[target] = df[target]
# chidf.iloc[:int(chidf.shape[0]/2),:] = True
# chidf.iloc[int(chidf.shape[0]/2):,:] = False
# quant_value = 0.99
# tar_value = np.quantile(chidf[target], quant_value)
# chidf[target] = chidf[target] > tar_value
welchsum = pd.DataFrame()
cutoff = 0.0001
pop = chidf[target].values

for ind in con_features:
    # ind = con_features[0]
    chidf[ind] = df[ind]

    # for grp_label in df[ind].unique():
    # try:
    a = []
    b = []
    for i in range(100):
        boot1 = random.sample(
                    list(
                        chidf.loc[
                            (chidf[target] == True)
                        ][ind].values),
                    k=1000)
        boot2 = random.sample(
                    list(
                        chidf.loc[
                            (chidf[target] == False)
                        ][ind].values),
                    k=1000)
        a.append(np.mean(boot1))
        b.append(np.mean(boot2))
    testt, p = stats.ttest_ind(a, b, equal_var=False)
    a_avg = np.mean(a)
    b_avg = np.mean(b)
    welchsum = pd.concat([welchsum, pd.DataFrame([[ind, testt, p, a_avg, b_avg, p<cutoff]])])
    sns.histplot(a, color='tab:orange', label=f"{target} == True")
    sns.histplot(b, label=f"{target} == False")
    plt.title(ind)
    plt.legend()
    plt.show()
    # except:
    #     pass

welchsum.columns = ['feature', 'test stat', 'p-value', 'upper q avg', 'lower q avg', 'reject null']
welchsum = welchsum.sort_values('p-value').reset_index(drop=True)
```

### Considering outliers


```python
df = df.loc[df[targets].apply(lambda x: (x < 3*x.std()) if (x.dtype == int or x.dtype == float) else x).all(axis=1)]
```


```python
df = df.loc[df['owner'] != 'spotify']
```

### Multiple Criteria for Success


```python
df['success'] = df[sub_targets].apply(lambda x: x > np.quantile(x, 0.75)).all(axis=1)
```


```python
chidf = pd.DataFrame()
target = 'success'
chidf[target] = df[target]
# quant_value = 0.90
# tar_value = np.quantile(chidf[target], quant_value)
# chidf[target] = chidf[target] > tar_value
chisum = pd.DataFrame()
cutoff = 0.0001
pop = chidf[target].values

for ind in des_features:
    # ind = des_features[0]
    chidf[ind] = df[ind]

    for grp_label in df[ind].unique():
    # grp_label = df[ind].unique()[0]
        try:
            cTable = chidf.groupby(chidf[ind] == grp_label)[target].value_counts().values.reshape(2,2).T
            chi2, p, dof, ex = stats.chi2_contingency(cTable, correction=True, lambda_=None)
            ratio = cTable[1]/cTable[0]
            pos = ratio[1]/ratio[0]
            chisum = pd.concat([chisum, pd.DataFrame([[ind, grp_label, chi2, p, cTable, pos, p<cutoff]])])
        except:
            pass

chisum.columns = ['feature', 'group', 'chi', 'p-value', 'cTable', 'multiplier', 'reject null']
chisum = chisum.sort_values('p-value').reset_index(drop=True)
```


```python
chisum.loc[chisum['reject null'] == True].sort_values('multiplier', ascending=False)
```


```python
fig, ax = plt.subplots(2, 2, figsize=(10,10), sharex='col', sharey='row')

genre_list = chisum.loc[chisum['feature'] == 'genre_1'].sort_values('multiplier', ascending=False)['group'].values
dff = pd.DataFrame(df.groupby(['genre_1'])['success'].value_counts(sort=False))
dff.columns = ['percent']
dff = dff.reset_index()
dff.loc[dff['success'] == True, 'percent'] = dff.loc[dff['success'] == True, 'percent'] / dff.loc[dff['success'] == True, 'percent'].sum() 
dff.loc[dff['success'] == False, 'percent'] = dff.loc[dff['success'] == False, 'percent'] / dff.loc[dff['success'] == False, 'percent'].sum() 
dff = dff.set_index('genre_1').loc[genre_list,:]
dff = dff.reset_index()

sns.barplot(data=dff.iloc[:10,:], hue='success', y='genre_1', x='percent', ax=ax[0,0])
ax[0,0].set_title('Best  and Worst Genres, Percent')
ax[0,0].set_ylabel('')
ax[0,0].set_xlabel('')
sns.barplot(data=dff.iloc[-10:,:], hue='success', y='genre_1', x='percent', ax=ax[1,0])
# ax[1,0].set_title('Worst Primary Genres')
ax[1,0].set_ylabel('')

dff = pd.DataFrame(df.groupby(['genre_1'])['success'].value_counts(sort=False))
dff.columns = ['count']
dff = dff.reset_index()
# dff.loc[dff['success'] == True, 'percent'] = dff.loc[dff['success'] == True, 'percent'] / dff.loc[dff['success'] == True, 'percent'].sum() 
# dff.loc[dff['success'] == False, 'percent'] = dff.loc[dff['success'] == False, 'percent'] / dff.loc[dff['success'] == False, 'percent'].sum() 
dff = dff.set_index('genre_1').loc[genre_list,:]
dff = dff.reset_index()

# fix, ax = plt.subplots(2, 2, figsize=(10,10))
sns.barplot(data=dff.iloc[:10,:], hue='success', y='genre_1', x='count', ax=ax[0,1])
ax[0,1].set_title('Best and Worst Genres, Count')
ax[0,1].set_ylabel('')
ax[0,1].set_xlabel('')
sns.barplot(data=dff.iloc[-10:,:], hue='success', y='genre_1', x='count', ax=ax[1,1])
# ax[1,1].set_title('Worst Primary Genres')
ax[1,1].set_ylabel('')
plt.tight_layout()

ax[0,0].get_legend().remove()
ax[1,1].get_legend().remove()
ax[1,0].get_legend().remove()
ax[0,1].legend(framealpha=1, facecolor='white', title="Success")

fig.savefig("categorical_dependency.svg")
```


```python
ind = 'n_tracks'
target = 'wau'
mean_wau_vs_track = []
for track in range(1, 201):
    means = [] 
    for i in range(10):
        boot = random.sample(
                    list(
                        df.loc[
                            (df['success'] == True) 
                            & (df[ind] == track)
                        ][target].values),
                    k=min(len(list(
                        df.loc[
                            (df['success'] == True) 
                            & (df[ind] == track)
                        ][target].values)), 1000))
        means.append(np.mean(boot))
    mean_wau_vs_track.append(np.mean(means))
```


```python
fig, ax = plt.subplots(figsize=(10,10))
plt.plot(range(len(mean_wau_vs_track)), mean_wau_vs_track, ls='', marker='.')
# ax.set_ylim(0,5)
```


```python
len(df.loc[
                            (df['success'] == True) 
                            & (df[ind] == track)
                        ][target].values)
```

### Dependency


```python
master = pd.DataFrame()
for target in new_metrics:
    # target = sub_targets[0]
    chidf = pd.DataFrame()
    chidf[target] = df[target]
    quant_value = 0.90
    tar_value = np.quantile(chidf[target], quant_value)
    tar_value = 0.8
    chidf[target] = chidf[target] >= tar_value
    chisum = pd.DataFrame()
    cutoff = 0.0001
    pop = chidf[target].values

    for ind in des_features:
        # ind = des_features[0]
        chidf[ind] = df[ind]

        for grp_label in df[ind].unique():
        # grp_label = df[ind].unique()[0]
            try:
                cTable = chidf.groupby(chidf[ind] == grp_label)[target].value_counts().values.reshape(2,2).T
                chi2, p, dof, ex = stats.chi2_contingency(cTable, correction=True, lambda_=None)
                ratio = cTable[1]/cTable[0]
                pos = ratio[1]/ratio[0]
                chisum = pd.concat([chisum, pd.DataFrame([[target, ind, grp_label, chi2, p, cTable, pos, p<cutoff]])])
            except:
                pass

    chisum.columns = ['target', 'feature', 'group', 'chi', 'p-value', 'cTable', 'multiplier', 'reject null']
    chisum = chisum.sort_values('p-value').reset_index(drop=True)
    # chisum = chisum.loc[(chisum['reject null'] == True) & (chisum['multiplier'] > 2)].sort_values('multiplier', ascending=False)
    master = pd.concat((master, chisum))
    
master
```


```python
master.loc[(master['reject null'] == True) & (master['multiplier'] > 1.5)]
```


```python
master.loc[(master['reject null'] == True) & (master['multiplier'] < .5)]
```


```python
new_master = pd.DataFrame()
for target in new_metrics:
    # target = sub_targets[2]
    chidf = pd.DataFrame()
    chidf[target] = df[target]
    chidf['n_tracks'] = df['n_tracks']
    quant_value = 0.90
    tar_value = np.quantile(chidf[target], quant_value)
    tar_value = 0.8
    chidf[target] = chidf[target] >= tar_value
    welchsum = pd.DataFrame()
    cutoff = 0.0001
    pop = chidf[target].values

    for ind in con_features:
        # ind = con_features[0]
        chidf[ind] = df[ind]

        # for grp_label in df[ind].unique():
        # try:
        a = []
        b = []
        for i in range(100):
            boot1 = random.sample(
                        list(
                            chidf.loc[
                                (chidf[target] == True)
                                & (chidf['n_tracks'] > 9)
                                & (chidf['n_tracks'] < 999)
                            ][ind].values),
                        k=1000)
            boot2 = random.sample(
                        list(
                            chidf.loc[
                                (chidf[target] == False)
                                & (chidf['n_tracks'] > 9)
                                & (chidf['n_tracks'] < 999)
                            ][ind].values),
                        k=1000)
            a.append(np.mean(boot1))
            b.append(np.mean(boot2))
        testt, p = stats.ttest_ind(a, b, equal_var=False)
        a_avg = np.mean(a)
        b_avg = np.mean(b)
        welchsum = pd.concat([welchsum, pd.DataFrame([[target, ind, testt, p, a_avg, b_avg, p<cutoff]])])
        sns.histplot(a, color='tab:orange', label=f"{target} >= {tar_value:.0f}")
        sns.histplot(b, label=f"{target} < {tar_value:.0f}")
        plt.title(f"{target}, {ind}")
        plt.legend()
        plt.show()
        # except:
        #     pass

    welchsum.columns = ['target', 'feature', 'test stat', 'p-value', 'upper q avg', 'lower q avg', 'reject null']
    welchsum = welchsum.sort_values('p-value').reset_index(drop=True)
    new_master = pd.concat((new_master, welchsum))
new_master
```

## Conclusions

### Discrete, Independent Variables

We note that there is class imbalance in the discrete independent variables:


```python
fig, ax = plt.subplots(1, 2, figsize=(10,10))

dff = pd.DataFrame(df[des_features[0]].value_counts()).join(
    pd.DataFrame(df[des_features[1]].value_counts())).join(
    pd.DataFrame(df[des_features[2]].value_counts()))
dff = dff.reset_index().melt(id_vars='index')
dff.columns = ['mood', 'order', 'count']
sns.barplot(data=dff, hue='order', y='mood', x='count', orient='h', ax=ax[0])

dff = pd.DataFrame(df[des_features[3]].value_counts()).join(
    pd.DataFrame(df[des_features[4]].value_counts())).join(
    pd.DataFrame(df[des_features[5]].value_counts()))
dff = dff.reset_index().melt(id_vars='index')
dff.columns = ['genre', 'order', 'count']
sns.barplot(data=dff, hue='order', y='genre', x='count', orient='h', ax=ax[1])

plt.tight_layout()
```

This class imbalance can have a variety of effects (and might be derived from a variety of sources).

For example, users will have more choice when listening to popular genres likeIndie Rock and Rap, and less choice with genres like Blues and Easy listening. As it so happens, when we look to the relationship between genre/mood and the dependent variables, many of the genre/moods with smaller class sizes will have a positive multiplier effect on the dependent variable

### Continuous, Independent Variables

The four continuous variables of focus in this dataset are highly tailed. Due to this, our statistical tests will require bootstrapping.


```python
quant = 0.999
con_features = ['n_albums', 'n_artists', 'n_tracks', 'n_local_tracks']
for target in con_features:
    cutoff = np.quantile(df[target], quant)
    y = df.loc[df[target] < cutoff]
    removed = df.loc[~(df[target] < cutoff)]
    print(f"removed items: {removed.shape[0]}")
    y.plot(kind='hist', y=target, bins=100, density=True)
    plt.show()
```

    removed items: 404



    
![png](X3_Spotify-Copy1_files/X3_Spotify-Copy1_81_1.png)
    


    removed items: 405



    
![png](X3_Spotify-Copy1_files/X3_Spotify-Copy1_81_3.png)
    


    removed items: 404



    
![png](X3_Spotify-Copy1_files/X3_Spotify-Copy1_81_5.png)
    


    removed items: 406



    
![png](X3_Spotify-Copy1_files/X3_Spotify-Copy1_81_7.png)
    


an example of bootstrapping `n_albums`


```python
means = []
ind = con_features[0]
for i in range(100):
    boot = random.sample(
                list(
                    df.loc[
                        (df[ind] > 9) 
                        & (df[ind] < 999)
                    ][ind].values),
                k=1000)
    means.append(np.mean(boot))
stuff = plt.hist(means, bins=100, density=True)
```

### Discrete, Dependent Variables

For the purposes of investigating a "successful" playlist, there are 5 primary metrics:


```python
targets
```


```python
df[sub_targets].describe().round(1).to_excel("file.xlsx")
```

and "top" performers in each of these metrics were based on top 10% and top 1% quantiles:


```python
print('p99 targets')
for target in sub_targets:
    space = ' '* (20 - len(str(target)))
    print(f"{target}: {space} {np.quantile(df[target], 0.99)}")
print()
print('p90 targets')
for target in sub_targets:
    space = ' '* (20 - len(str(target)))
    print(f"{target}: {space} {np.quantile(df[target], 0.90)}")
```

You can imagine with these metrics, some concerns are:

* what if a playlist was made in the current month, or even current day?
    * playlist is not properly represented by the data
* how do we normalize by playlists that already have a high visibility? i.e. what if a playlist is "good" but just isn't getting noticed?
    * can compute conversion metrics:
        * 30 second listens / total listens
        * mau both months / mau previous month
        
While noting these shortcomings, to keep the analysis focused I singled out the previously mentioned targets, with a focus on `monthly_stream30s` as the north star metric. `monthly_stream30s` is advantageous as a nort star metric since it contains data from the entire month (reducing variance) only contains relevant listens (greater than 30 seconds long). Some disadvantages of this metric are that it doesn't account for just a few listeners who may be providing the majority of listens, and playlists that were made in the current month will be undervalued.

### Dependency

#### Chi Square

In the chi-square test, the contigency table was used to calculate a `multiplier` effect. This is a ratio of ratios: the count of upper quantile over bottom quantile for the given group over the count of upper quantile over bottom quantile for non-group. In other words, it articulates how much more likely a sample in the given group is likely to be in the upper quantile vs a sample not in the given group


```python
chisq_results = pd.read_csv("chi_square_results.csv", index_col=0)
chisq_results.head()
```


```python
chisq_results['target'].unique()
```


```python
chisq_results['upper q'].unique()
```

Taking together the five targets, the two upper quantiles, and the six categorical independent variables, we can identify which group occured the most frequently as a variable of influence:


```python
chisq_results.loc[(chisq_results['feature'].str.contains('genre'))
                & (chisq_results['group'] != '-')]['group'].value_counts()
```

Using these value counts as a "rank" we can then groupby this rank and see how each group is influencing the propensity to be in the upper quadrant

Taking "Romantic" as an example, we see that it's multiplier effect is relatively consistent across the five targets and two quantiles:


```python
sort_key = {i: j for i,j in zip(chisq_results['group'].value_counts().index.values, range(chisq_results['group'].nunique()))}
chisq_results['rank'] = chisq_results['group'].apply(lambda x: sort_key[x])
chisq_results.sort_values('rank', inplace=True)
# chisq_results.drop('rank', axis=1, inplace=True)
chisq_results.loc[chisq_results['group'] != '-'][:20]
```


```python
chisq_results.loc[(chisq_results['group'] == 'Traditional')
                 & (chisq_results['target'] == 'monthly_stream30s')]
```

Let's use this idea of average multiplier effect, and average chi-square statistic to summarize by group.

Sorting by the test statistic, we see the top 5 most influential groups:


```python
chisq_results.groupby('group')[['chi', 'multiplier', 'rank']].mean().sort_values('chi', ascending=False)[:10]
```

Sorting instead by the multiplier, we can see which group has the _heaviest_ influence


```python
chisq_results.groupby('group')[['chi', 'multiplier', 'rank']].mean().sort_values('multiplier', ascending=False)[:10]
```

Sorting instead by rank, we see which groups show up most frequently


```python
chisq_results.groupby('group')[['chi', 'multiplier', 'rank']].mean().sort_values('rank', ascending=True)[:10]
```


```python
chisq_results.loc[chisq_results['target'] == 'monthly_stream30s']
```

It creates some fog to jumble together mood/genres this way. We can instead separate them and ask questions like:

##### What is the most influential primary genre on monthly streams over 30 seconds?

Answer: Children's followed by Latin

Reason: both genre's appear as influential in other guardrail metrics (high rank), have high test statistics, and are influential in both p99 and p90 with multiplier effects of [4.8, 2.6] and [3.1, 2.1], respectively.


```python
chisq_results.loc[(chisq_results['feature'] == 'genre_1')
                 & (chisq_results['target'] == 'monthly_stream30s')]
```

##### What is the most influential primary mood on monthly streams over 30 seconds?

Answer: Romantic and Lively

Reason: Romantic and Lively moods appear multiple times as highly influential (high rank) they have high multipliers. A contendent may be Tender, as it has a high multiplier effect as well at 3.75


```python
chisq_results.loc[(chisq_results['feature'] == 'mood_1')
                 & (chisq_results['target'] == 'monthly_stream30s')]
```

##### Which Categorical Feature is most influential overall?

Answer: genre_1, followed by genre_2 and mood_1

Reason: we see that these features appear multiple times across the 5 different targets and 2 different quantiles


```python
chisq_results['feature'].value_counts()
```

##### What are the shortcomings of this analysis?

We haven't taken into account confounding variables. For example, perhaps Latin genre is typically associated with Lively mood. Then which variable is it that actually contributes to a highly performing playlist? We have strategies for dealing with this. We can stratify the confounding variables by over or under sampling. We can also consider them together in a forward selection logistic model. We will take the latter approach later on in the analysis.

We haven't considered the categorical variables alongside the continuous variables, so we don't know how they fit overall in terms of relative improtance. We will approach this the same way as the confounding variables issue, and incorporate all variables in a logistic regression.

#### t-Test


```python
ttest_results = pd.read_csv("t_test_results.csv", index_col=0)
ttest_results.head()
```

#### Models


```python
log_results = pd.read_csv("../../scripts/fwd_selection_results.txt", header=None, index_col=0)
log_results.columns = ['feature', 'pseudo r2']
log_results.reset_index(inplace=True, drop=True)
log_results.drop(0, axis=0, inplace=True)
log_results
```


```python
target = "monthly_stream30s"
y = df[target].values
labels = y.copy()
names = []
weights = y.copy()
weights.dtype = 'float'
lim = 11
dom_class_weight = 1 / (lim - 1 - 1)
for idx, quant in zip(range(lim), np.linspace(0, 1, num=lim)):
    if idx < lim - 2:
        prev = quant
        continue
    elif idx == lim - 2:
        weights[y <= np.quantile(y, quant)] = dom_class_weight
        labels[labels <= np.quantile(y, quant)] = 0
        names += [f"less than {np.quantile(y, quant):.0f} listens"]
        
    else:
        labels[(labels > np.quantile(y, prev))
              & (labels <= np.quantile(y, quant))] = 1
        weights[(y > np.quantile(y, prev))
              & (y <= np.quantile(y, quant))] = 1.0
        names += [f"{np.quantile(y, prev):.0f} < listens <= {np.quantile(y, quant):.0f}"]
    prev = quant
y = labels

basemodel = pd.read_csv("../../scripts/basemodel.csv", index_col = 0)
X2 = basemodel.values
est = Logit(y, X2)
est2 = est.fit(disp=0)
summ = est2.summary()

res_table = summ.tables[1]
res_df = pd.DataFrame(res_table.data)
cols = res_df.iloc[0]
cols = [str(i) for i in cols]
res_df.drop(0, axis=0, inplace=True)
res_df.set_index(0, inplace=True)
res_df.columns = cols[1:]
res_df.index = basemodel.columns
res_df
```
