<a href="https://colab.research.google.com/github/wesleybeckner/data_science_foundations/blob/main/notebooks/S1_Regression_and_Analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Data Science Foundations Session 1: Regression and Analysis

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com

---

<br>

In this session we will look at fitting data to a curve using **regression**. We will also look at using regression to make **predictions** for new data points by dividing our data into a training and a testing set. Finally we will examine how much error we make in our fit and then in our predictions by computing the mean squared error.

<br>

---



<a name='x.0'></a>

## 1.0 Preparing Environment and Importing Data

[back to top](#top)

<a name='x.0.1'></a>

### 1.0.1 Import Packages

[back to top](#top)


```python
# Import pandas, pyplot, ipywidgets
import pandas as pd  
from matplotlib import pyplot as plt
from ipywidgets import interact

# Import Scikit-Learn library for the regression models
import sklearn         
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# for enrichment topics
import seaborn as sns
import numpy as np
```

### 1.0.2 Load Dataset

[back to top](#top)

For our discussion on regression and descriptive statistics today we will use a well known dataset of different wines and their quality ratings


```python
df = pd.read_csv("https://raw.githubusercontent.com/wesleybeckner/"\
                 "ds_for_engineers/main/data/wine_quality/winequalityN.csv")
df.shape
```




    (6497, 13)




```python
df.head()
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
      <th>type</th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>white</td>
      <td>7.0</td>
      <td>0.27</td>
      <td>0.36</td>
      <td>20.7</td>
      <td>0.045</td>
      <td>45.0</td>
      <td>170.0</td>
      <td>1.0010</td>
      <td>3.00</td>
      <td>0.45</td>
      <td>8.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>white</td>
      <td>6.3</td>
      <td>0.30</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.049</td>
      <td>14.0</td>
      <td>132.0</td>
      <td>0.9940</td>
      <td>3.30</td>
      <td>0.49</td>
      <td>9.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>white</td>
      <td>8.1</td>
      <td>0.28</td>
      <td>0.40</td>
      <td>6.9</td>
      <td>0.050</td>
      <td>30.0</td>
      <td>97.0</td>
      <td>0.9951</td>
      <td>3.26</td>
      <td>0.44</td>
      <td>10.1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>white</td>
      <td>7.2</td>
      <td>0.23</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.9956</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>white</td>
      <td>7.2</td>
      <td>0.23</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.9956</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



## 1.1 What is regression? 

It is the process of finding a relationship between **_dependent_** and **_independent_** variables to find trends in data. This abstract definition means that you have one variable (the dependent variable) which depends on one or more variables (the independent variables). One of the reasons for which we want to regress data is to understand whether there is a trend between two variables. 

**Housing Prices Example**

We can imagine this scenario with housing prices. Envision a **_mixed_** dataset of **_continuous_** and **_discrete_** independent variables. Some features could be continuous, floating point values like location ranking and housing condition. Others could be discrete like the number of rooms or bathrooms. We could take these features and use them to predict a house value. This would be a **_regression_** model.

<p align=center>
<img src="https://raw.githubusercontent.com/wesleybeckner/technology_explorers/main/assets/machine_learning/ML3.png" width=1000px></img>
</p>






## 1.2  Linear regression fitting with scikit-learn

#### 🏋️ Exercise 1: rudimentary EDA
What does the data look like? Recall how to visualize data in a pandas dataframe 

* for every column calculate the:
    * skew: `df.skew()`
    * kurtosis: `df.kurtosis()`
    * pearsons correlation with the dependent variable: `df.corr()`
    * number of missing entries `df.isnull()`
* and organize this into a new dataframe

_note:_ pearsons is just one type of correlation, another correlation available to us is **_spearman_** which differs from pearsons in that it depends on ranked values rather than their direct quantities, you can read more [here](https://support.minitab.com/en-us/minitab-express/1/help-and-how-to/modeling-statistics/regression/supporting-topics/basics/a-comparison-of-the-pearson-and-spearman-correlation-methods/)


```python
df.isnull().sum()
```




    type                     0
    fixed acidity           10
    volatile acidity         8
    citric acid              3
    residual sugar           2
    chlorides                2
    free sulfur dioxide      0
    total sulfur dioxide     0
    density                  0
    pH                       9
    sulphates                4
    alcohol                  0
    quality                  0
    dtype: int64




```python
# Cell for Exercise 1

# part A 
# using df.<method> define the following four variables with the results from
# skew(), kurtosis(), corr() (while selecting for quality), and isnull()
# for isnull() you'll notice the return is a dataframe of booleans. we would
# like to simply know the number of null values for each column. change the
# return of isnull() using the sum() method along the columns

# skew =
# kurt =
# pear =
# null =

# part B
# on line 13, put these results in a list using square brackets and call 
# pd.DataFrame on the list to make your new DataFrame! store it under the
# variable name dff



# part C
# take the transpose of this DataFrame using dff.T. reassign dff to this copy



# part D
# set the column names to 'skew', 'kurtosis', 'pearsons _quality', and 
# 'null count' using dff.columns



# Now return dff to the output to view your hand work
# dff # uncomment this line
```

I have gone ahead and repeated this exercise with the red vs white wine types:


```python
red = df.loc[df['type'] == 'red']
wht = df.loc[df['type'] == 'white']

def get_summary(df):
  skew = df.skew()
  kurt = df.kurtosis()
  pear = df.corr()['quality']
  null = df.isnull().sum()
  med = df.median()
  men = df.mean()

  dff = pd.DataFrame([skew, kurt, pear, null, med, men])
  dff = dff.T
  dff.columns = ['skew', 'kurtosis', 'pearsons _quality', 'null count', 'median',
                'mean']
  return dff

dffr = get_summary(red)
dffw = get_summary(wht)

desc = pd.concat([dffr, dffw], keys=['red', 'white'])
```

    /tmp/ipykernel_1419/2387423026.py:5: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      skew = df.skew()
    /tmp/ipykernel_1419/2387423026.py:6: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      kurt = df.kurtosis()
    /tmp/ipykernel_1419/2387423026.py:9: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      med = df.median()
    /tmp/ipykernel_1419/2387423026.py:10: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      men = df.mean()



```python
desc
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
      <th></th>
      <th>skew</th>
      <th>kurtosis</th>
      <th>pearsons _quality</th>
      <th>null count</th>
      <th>median</th>
      <th>mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="13" valign="top">red</th>
      <th>fixed acidity</th>
      <td>0.982192</td>
      <td>1.132624</td>
      <td>0.123834</td>
      <td>2.0</td>
      <td>7.90000</td>
      <td>8.322104</td>
    </tr>
    <tr>
      <th>volatile acidity</th>
      <td>0.672862</td>
      <td>1.226846</td>
      <td>-0.390858</td>
      <td>1.0</td>
      <td>0.52000</td>
      <td>0.527738</td>
    </tr>
    <tr>
      <th>citric acid</th>
      <td>0.317891</td>
      <td>-0.788476</td>
      <td>0.226917</td>
      <td>1.0</td>
      <td>0.26000</td>
      <td>0.271145</td>
    </tr>
    <tr>
      <th>residual sugar</th>
      <td>4.540655</td>
      <td>28.617595</td>
      <td>0.013732</td>
      <td>0.0</td>
      <td>2.20000</td>
      <td>2.538806</td>
    </tr>
    <tr>
      <th>chlorides</th>
      <td>5.680347</td>
      <td>41.715787</td>
      <td>-0.128907</td>
      <td>0.0</td>
      <td>0.07900</td>
      <td>0.087467</td>
    </tr>
    <tr>
      <th>free sulfur dioxide</th>
      <td>1.250567</td>
      <td>2.023562</td>
      <td>-0.050656</td>
      <td>0.0</td>
      <td>14.00000</td>
      <td>15.874922</td>
    </tr>
    <tr>
      <th>total sulfur dioxide</th>
      <td>1.515531</td>
      <td>3.809824</td>
      <td>-0.185100</td>
      <td>0.0</td>
      <td>38.00000</td>
      <td>46.467792</td>
    </tr>
    <tr>
      <th>density</th>
      <td>0.071288</td>
      <td>0.934079</td>
      <td>-0.174919</td>
      <td>0.0</td>
      <td>0.99675</td>
      <td>0.996747</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>0.194803</td>
      <td>0.814690</td>
      <td>-0.057094</td>
      <td>2.0</td>
      <td>3.31000</td>
      <td>3.310864</td>
    </tr>
    <tr>
      <th>sulphates</th>
      <td>2.429115</td>
      <td>11.712632</td>
      <td>0.251685</td>
      <td>2.0</td>
      <td>0.62000</td>
      <td>0.658078</td>
    </tr>
    <tr>
      <th>alcohol</th>
      <td>0.860829</td>
      <td>0.200029</td>
      <td>0.476166</td>
      <td>0.0</td>
      <td>10.20000</td>
      <td>10.422983</td>
    </tr>
    <tr>
      <th>quality</th>
      <td>0.217802</td>
      <td>0.296708</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>6.00000</td>
      <td>5.636023</td>
    </tr>
    <tr>
      <th>type</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="13" valign="top">white</th>
      <th>fixed acidity</th>
      <td>0.647981</td>
      <td>2.176560</td>
      <td>-0.114032</td>
      <td>8.0</td>
      <td>6.80000</td>
      <td>6.855532</td>
    </tr>
    <tr>
      <th>volatile acidity</th>
      <td>1.578595</td>
      <td>5.095526</td>
      <td>-0.194976</td>
      <td>7.0</td>
      <td>0.26000</td>
      <td>0.278252</td>
    </tr>
    <tr>
      <th>citric acid</th>
      <td>1.284217</td>
      <td>6.182036</td>
      <td>-0.009194</td>
      <td>2.0</td>
      <td>0.32000</td>
      <td>0.334250</td>
    </tr>
    <tr>
      <th>residual sugar</th>
      <td>1.076601</td>
      <td>3.469536</td>
      <td>-0.097373</td>
      <td>2.0</td>
      <td>5.20000</td>
      <td>6.393250</td>
    </tr>
    <tr>
      <th>chlorides</th>
      <td>5.023412</td>
      <td>37.560847</td>
      <td>-0.210181</td>
      <td>2.0</td>
      <td>0.04300</td>
      <td>0.045778</td>
    </tr>
    <tr>
      <th>free sulfur dioxide</th>
      <td>1.406745</td>
      <td>11.466342</td>
      <td>0.008158</td>
      <td>0.0</td>
      <td>34.00000</td>
      <td>35.308085</td>
    </tr>
    <tr>
      <th>total sulfur dioxide</th>
      <td>0.390710</td>
      <td>0.571853</td>
      <td>-0.174737</td>
      <td>0.0</td>
      <td>134.00000</td>
      <td>138.360657</td>
    </tr>
    <tr>
      <th>density</th>
      <td>0.977773</td>
      <td>9.793807</td>
      <td>-0.307123</td>
      <td>0.0</td>
      <td>0.99374</td>
      <td>0.994027</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>0.458402</td>
      <td>0.532552</td>
      <td>0.098858</td>
      <td>7.0</td>
      <td>3.18000</td>
      <td>3.188203</td>
    </tr>
    <tr>
      <th>sulphates</th>
      <td>0.977361</td>
      <td>1.589847</td>
      <td>0.053690</td>
      <td>2.0</td>
      <td>0.47000</td>
      <td>0.489835</td>
    </tr>
    <tr>
      <th>alcohol</th>
      <td>0.487342</td>
      <td>-0.698425</td>
      <td>0.435575</td>
      <td>0.0</td>
      <td>10.40000</td>
      <td>10.514267</td>
    </tr>
    <tr>
      <th>quality</th>
      <td>0.155796</td>
      <td>0.216526</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>6.00000</td>
      <td>5.877909</td>
    </tr>
    <tr>
      <th>type</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
def my_fig(metric=desc.columns):
  fig, ax = plt.subplots(1, 1, figsize=(10,10))
  pd.DataFrame(desc[metric]).unstack()[metric].T.plot(kind='barh', ax=ax)
```


```python
# interact(my_fig)
```

#### 🙋 Question 1: Discussion Around EDA Plot

What do we think of this plot?

> `metric = mean`, the cholrides values <br>
`metric = kurtosis`, residual sugar <br>
`metric = pearsons _quality`, _magnitudes_ and _directions_ <br>
How to improve the plot, what other plots would we like to see?

For instance, what if we were really curious about the high kurtosis for chlorides content? What more would we like to glean about the distribution of chloride content?


```python
# we can use df.describe() to take a look at the quantile values and min/max
df['chlorides'].describe()
```




    count    6495.000000
    mean        0.056042
    std         0.035036
    min         0.009000
    25%         0.038000
    50%         0.047000
    75%         0.065000
    max         0.611000
    Name: chlorides, dtype: float64




```python
# and see how these values appear in a KDE
fig, ax = plt.subplots(1,1,figsize=(10,10))
df['chlorides'].plot(kind='kde',ax=ax)
ax.set_xlim(0,.61)
```




    (0.0, 0.61)




    
![png](S1_Regression_and_Analysis_files/S1_Regression_and_Analysis_22_1.png)
    



```python
# lastly we may want to look at the raw values themselves. We can sort them 
# too view outliers
df['chlorides'].sort_values(ascending=False)[:50]
```




    5156    0.611
    5049    0.610
    5004    0.467
    4979    0.464
    5590    0.422
    6268    0.415
    6270    0.415
    5652    0.415
    6217    0.414
    5949    0.414
    5349    0.413
    6158    0.403
    4981    0.401
    5628    0.387
    6063    0.369
    4915    0.368
    5067    0.360
    5179    0.358
    484     0.346
    5189    0.343
    4917    0.341
    5124    0.337
    4940    0.332
    1217    0.301
    687     0.290
    4473    0.271
    5079    0.270
    6272    0.267
    5138    0.263
    1865    0.255
    5466    0.250
    1034    0.244
    5674    0.243
    5675    0.241
    683     0.240
    1638    0.239
    5045    0.236
    6456    0.235
    6468    0.230
    5465    0.226
    5464    0.226
    5564    0.222
    2186    0.217
    5996    0.216
    6333    0.214
    5206    0.214
    6332    0.214
    5205    0.213
    4497    0.212
    1835    0.211
    Name: chlorides, dtype: float64



### 1.2.2 Visualizing the data set - motivating regression analysis

In order to demonstrate simple linear regression with this dataset we will look at two particular features: `fixed acidity` and `density`.

We can create a scatter plot of `fixed acidity` vs `density` for the red wine in the dataset using `df.plot()` and see that there appears to be a general trend between the two features:


```python
fig, ax = plt.subplots(1, 1, figsize=(5,5))
df.loc[df['type'] == 'red'].plot(x='fixed acidity', y='density', ax=ax, 
                                 ls='', marker='.')
```




    <AxesSubplot:xlabel='fixed acidity'>




    
![png](S1_Regression_and_Analysis_files/S1_Regression_and_Analysis_26_1.png)
    


Now the question is: How do we quantify this trend?

### 1.2.3 Estimating the regression coefficients

It looks like density increases with fixed acidity following a line, maybe something like

$$y(x)= m \cdot x + b  \;\;\;\;\;\;\;\; \sf{eq. 1}$$ 

with \\( y=\sf density \\), \\(x=\sf fixed \space acidity\\), and \\(m\\) the slope and \\(b\\) the intercept. 

To solve the problem, we need to find the values of \\(b\\) and \\(m\\) in equation 1 to best fit the data. This is called **linear regression**.

In linear regression our goal is to minimize the error between computed values of positions \\(y^{\sf calc}(x_i)\equiv y^{\sf calc}_i\\) and known values \\(y^{\sf exact}(x_i)\equiv y^{\sf exact}_i\\), i.e. find \\(b\\) and \\(m\\) which lead to lowest value of

$$\epsilon (m,b) =SS_{\sf res}=\sum_{i=1}^{N}\left(y^{\sf exact}_i - y^{\sf calc}_i\right)^2 = \sum_{i=1}^{N}\left(y^{\sf exact}_i - m\cdot x_i - b \right)^2\;\;\;\;\;\;\;\;\;\;\;\sf{eq. 2}$$

Otherwise known as the **residual sum of squares**

To find out more see e.g. https://en.wikipedia.org/wiki/Simple_linear_regression


#### 🙋 Question 2: linear regression loss function
> Do we always want *m* and *b* to be large positive numbers so as to minimize eq. 2? 


Luckily [scikit-learn](https://scikit-learn.org/stable/) contains many functions related to regression including [linear regression](https://scikit-learn.org/stable/modules/linear_model.html). 

The function we will use is called <code> LinearRegression() </code>. 


```python
# Create linear regression object
model = linear_model.LinearRegression()

# Use model to fit to the data, the x values are densities and the y values are fixed acidity
# Note that we need to reshape the vectors to be of the shape x - (n_samples, n_features) and y (n_samples, n_targets)
x = red['density'].values.reshape(-1, 1)
y = red['fixed acidity'].values.reshape(-1, 1)
```


```python
print(red['density'].values.shape, red['fixed acidity'].values.shape)
print(x.shape, y.shape)
```

    (1599,) (1599,)
    (1599, 1) (1599, 1)


What happens when we try to fit the data as is?


```python
# Fit to the data
# model.fit(x, y)
```

#### 🏋️ Exercise 2: drop Null Values (and practice pandas operations)

Let's look back at our dataset description dataframe above, what do we notice, what contains null values?

There are several strategies for dealing with null values. For now let's take the simplest case, and drop rows in our dataframe that contain null


```python
# Cell for Exercise 2
# For this templated exercise you are going to complete everything in one line 
# of code, but we are going to break it up into steps. So for each part (A, B,
# etc.) paste your answer from the previous part to begin (your opertaions will
# read from left to right)

# step A
# select the 'density' and 'fixed acidity' columns of red. make sure the return
# is a dataframe

# step B
# now use the dropna() method on axis 0 (the rows) to drop any null values 

# step B
# select column 'density'

# step C
# select the values

# step D
# reshape the result with an empty second dimension using .reshape() and store
# the result under variable x

# repeat the same process with 'fixed acidity' and variable y
```

Now that we have our x and y arrays we can fit using ScikitLearn


```python
x = red[['density', 'fixed acidity']].dropna(axis=0)['density'].values.reshape(-1,1)
y = red[['density', 'fixed acidity']].dropna(axis=0)['fixed acidity'].values.reshape(-1,1)
```

#### 🙋 Question 3: why do we drop null values across both columns?

Notice in the above cell how we selected both `density` and `fixed acidity` before calling `dropna`? Why did we do that? Why didn't we just select `density` in the `x` variable case and `fixed acidity` in the `y` variable case?


```python
# Fit to the data
model.fit(x, y)

# Extract the values of interest
m = model.coef_[0][0]
b = model.intercept_[0]

# Print the slope m and intercept b
print('Scikit learn - Slope: ', m , 'Intercept: ', b )
```

    Scikit learn - Slope:  616.01314280661 Intercept:  -605.6880086750523


#### 🏋️ Exercise 3: calculating y_pred
Estimate the values of \\(y\\) by using your fitted parameters. Hint: Use your <code>model.coef_</code> and <code>model.intercept_</code> parameters to estimate y_pred following equation 1



```python
# define y_pred in terms of m, x, and b
# y_pred = 

# uncomment the following lines!
# fig, ax = plt.subplots(1,1, figsize=(10,10))
# ax.plot(x, y_pred, ls='', marker='*')
# ax.plot(x, y, ls='', marker='.')
```

We can also return predictions directly with the model object using the predict() method
> note: it is great to get in the habit of utilizing model outputs this way, as the API will be similar across all scikit-learn models (and sometimes models in other libraries as well!)


```python
# Another way to get this is using the model.predict function
y_pred = model.predict(x)
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.plot(x, y_pred, ls='', marker='*')
ax.plot(x, y, ls='', marker='.')
```




    [<matplotlib.lines.Line2D at 0x7fdd881875e0>]




    
![png](S1_Regression_and_Analysis_files/S1_Regression_and_Analysis_44_1.png)
    


## 1.3 Error and topics of model fitting (assessing model accuracy)

### 1.3.1 Measuring the quality of fit

#### 1.3.1.1 Mean Squared Error

The plot in Section 1.2.3 looks good, but numerically what is our error? What is the mean value of $\epsilon$, i.e. the **Mean Squared Error (MSE)**?

$${\sf MSE}=\epsilon_{\sf ave} = \frac{\sum_{i=1}^{N_{\sf times}}\left(y^{\sf exact}_i - m\cdot t_i - b \right)^2}{N_{\sf times}}\;\;\;\;\;\sf eq. 3$$


```python
# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y, y_pred))
```

    Mean squared error: 1.68


#### 1.3.1.2 R-square

Another way to measure error is the regression score, \\(R^2\\). \\(R^2\\) is generally defined as the ratio of the total sum of squares \\(SS_{\sf tot}\\) to the residual sum of squares \\(SS_{\sf res}\\):

$$SS_{\sf tot}=\sum_{i=1}^{N} \left(y^{\sf exact}_i-\bar{y}\right)^2\;\;\;\;\; \sf eq. 4$$
$$SS_{\sf res}=\sum_{i=1}^{N} \left(y^{\sf exact}_i - y^{\sf calc}_i\right)^2\;\;\;\;\; \sf eq. 5$$
$$R^2 = 1 - {SS_{\sf res}\over SS_{\sf tot}} \;\;\;\;\;\; \sf eq. 6$$

In eq. 4, \\(\bar{y}=\sum_i y^{\sf exact}_i/N\\) is the average value of y for \\(N\\) points. The best value of \\(R^2\\) is 1 but it can also take a negative value if the error is large.

See all the different regression metrics [here](https://scikit-learn.org/stable/modules/model_evaluation.html).

#### 🙋 Question 4: lets understand \\(R^2\\)
> Do we need a large value of \\(SS_{\sf tot}\\) to minimize \\(R^2\\) - is this something which we have the power to control?


```python
# Print the coefficient of determination - 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(y, y_pred))
```

    Coefficient of determination: 0.45


### 1.3.2 Corollaries with classification models

For classification tasks, we typically assess accuracy vs MSE or R-square, since we are dealing with categorical rather than numerical predictions.

What is accuracy? It is defined as the ratio of True assignments to all assignments. For a binary positive/negative classification task this can be written as the following:

$$ Acc = \frac{T_p + T_n}{F_p + F_n + T_p + T_n} $$

Where \\(T\\) is True, \\(F\\) is false, \\(p\\) is positive, \\(n\\) is negative

Just as a quick example, we can perform this type of task on our wine dataset by predicting on quality, which is a discrete 3-9 quality score:


```python
y_train = df['type'].values.reshape(-1,1)
x_train = df['quality'].values.reshape(-1,1)
```


```python
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression

# instantiate model
logreg = LogisticRegression()

# fit model
logreg.fit(x_train, y_train)
```

    /home/wbeckner/anaconda3/envs/py39/lib/python3.9/site-packages/sklearn/utils/validation.py:993: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)





    LogisticRegression()




```python
# make class predictions for the testing set
y_pred_class = logreg.predict(x_train)
```


```python
# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_train, y_pred_class))
```

    0.7538864091118977


### 1.3.3 Beyond a single input feature

(_also: quick appreciative beat for folding in domain area expertise into our models and features_)

The **acidity** of the wine (the dependent variable v) could depend on:

* potassium from the soil (increases alkalinity)
* unripe grapes (increases acidity)
* grapes grown in colder climates or reduced sunshine create less sugar (increases acidity)
* preprocessing such as adding tartaric acid to the grape juice before fermentation (increases acidity)
* malolactic fermentation (reduces acidity)
* \+ others

So in our lab today we will look at folding in additional variables in our dataset into the model
<hr style="border:1px solid grey"> </hr>


## 1.4 Multivariate regression

Let's now turn our attention to wine quality.

The value we aim to predict or evaluate is the quality of each wine in our dataset. This is our dependent variable. We will look at how this is related to the 12 other independent variables, also known as *input features*. We're going to do this with only the red wine data


```python
red.head()
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
      <th>type</th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4898</th>
      <td>red</td>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4899</th>
      <td>red</td>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4900</th>
      <td>red</td>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4901</th>
      <td>red</td>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4902</th>
      <td>red</td>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



### 1.4.1 Linear regression with all input fields

For this example, notice we have a categorical data variable in the 'type' column. We will ignore this for now, and only work with our red wines. In the future we will discuss how to deal with categorical variable such as this in a mathematical representation.


```python
# this is a list of all our features or independent variables
features = list(red.columns[1:])

# we're going to remove our target or dependent variable, density from this
# list
features.remove('density')

# now we define X and y according to these lists of names
X = red.dropna(axis=0)[features].values
y = red.dropna(axis=0)['density'].values

# we will talk about scaling/centering our data at a later time
X = (X - X.mean(axis=0)) / X.std(axis=0)
```


```python
red.isnull().sum(axis=0) # we are getting rid of some nasty nulls!
```




    type                    0
    fixed acidity           2
    volatile acidity        1
    citric acid             1
    residual sugar          0
    chlorides               0
    free sulfur dioxide     0
    total sulfur dioxide    0
    density                 0
    pH                      2
    sulphates               2
    alcohol                 0
    quality                 0
    dtype: int64




```python
# Create linear regression object - note that we are using all the input features
model = linear_model.LinearRegression()
model.fit(X, y)
y_calc = model.predict(X)
```

Let's see what the coefficients look like ... 


```python
print("Fit coefficients: \n", model.coef_, "\nNumber of coefficients:", len(model.coef_))
```

    Fit coefficients: 
     [ 1.64059336e-03  1.23999138e-04  1.16115898e-05  5.83002013e-04
      8.35961822e-05 -9.17472420e-05  8.61246026e-05  7.80966358e-04
      2.24558885e-04 -9.80600257e-04 -1.75587885e-05] 
    Number of coefficients: 11


We have 11 !!! That's because we are regressing respect to all **11 independent variables**!!!

So now, $$y_{\sf calc}= m_1x_1 +\, m_2x_2 \,+ \,m_3x_3 \,+\,... \,+ \,b =\sum_{i=1}^{13}m_i x_i + b\;\;\;\;\; \sf eq. 7$$


```python
print("We have 11 slopes / weights:\n\n", model.coef_)
print("\nAnd one intercept: ", model.intercept_)
```

    We have 11 slopes / weights:
    
     [ 1.64059336e-03  1.23999138e-04  1.16115898e-05  5.83002013e-04
      8.35961822e-05 -9.17472420e-05  8.61246026e-05  7.80966358e-04
      2.24558885e-04 -9.80600257e-04 -1.75587885e-05]
    
    And one intercept:  0.9967517451349656



```python
# This size should match the number of columns in X
if len(X[0]) == len(model.coef_):
    print("All good! The number of coefficients matches the number of input features.")
else:
    print("Hmm .. something strange is going on.")
```

    All good! The number of coefficients matches the number of input features.


### 🏋️ Exercise 4: evaluate the error

Let's **evaluate the error** by computing the MSE and \\(R^2\\) metrics (see eq. 3 and 6).


```python
# The mean squared error

# part A 
# calculate the MSE using mean_squared_error()
# mse = 

# part B
# calculate the R square using r2_score()
# r2 = 

# print('Mean squared error: {:.2f}'.format(mse))
# print('Coefficient of determination: {:.2f}'.format(r2))
```

### 🏋️ Exercise 5: make a plot of y actual vs y predicted

We can also look at how well the computed values match the true values graphically by generating a scatterplot.


```python
# generate a plot of y predicted vs y actual using plt.plot()
# remember you must set ls to an empty string and marker to some marker style

# plt.plot()
plt.title("Linear regression - computed values on entire data set", fontsize=16)
plt.xlabel("y$^{\sf calc}$")
plt.ylabel("y$^{\sf true}$")
plt.show()
```


    
![png](S1_Regression_and_Analysis_files/S1_Regression_and_Analysis_74_0.png)
    


### 🍒 1.4.2 **Enrichment**: Splitting into train and test sets

> note: more of this topic is covered in [**Model Selection and Validation**](https://wesleybeckner.github.io/data_science_foundations/S3_Model_Selection_and_Validation/)

To see whether we can predict, we will carry out our regression only on a part, 80%, of the full data set. This part is called the **training** data. We will then test the trained model to predict the rest of the data, 20% - the **test** data. The function which fits won't see the test data until it has to predict it. 

**We will motivate the use of train/test sets more explicitly in [Model Selection and Validation](https://wesleybeckner.github.io/data_science_foundations/S3_Model_Selection_and_Validation/)**

We start by splitting out data using scikit-learn's <code>train_test_split()</code> function:


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20,
                                                    random_state=42)
```

Now we check the size of <code> y_train </code> and <code> y_test </code>, the sum should be the size of y! If this works then we move on and carry out regression but we only use the training data!


```python
if len(y_test)+len(y_train) == len(y):
    
    print('All good, ready to to go and regress!\n')
    
    # Carry out linear regression
    print('Running linear regression algorithm on the training set\n')
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    print('Fit coefficients and intercept:\n\n', model.coef_, '\n\n', model.intercept_ )

    # Predict on the test set
    y_pred_test = model.predict(X_test)
```

    All good, ready to to go and regress!
    
    Running linear regression algorithm on the training set
    
    Fit coefficients and intercept:
    
     [ 1.62385613e-03  1.10578142e-04  7.75216492e-07  5.87755741e-04
      7.65190323e-05 -1.03490059e-04  8.87357873e-05  7.79083342e-04
      2.23534769e-04 -9.99858829e-04  5.85256438e-06] 
    
     0.9967531628434799


Now we can plot our predicted values to see how accurate we are in predicting. We will generate a scatterplot and computing the MSE and \\(R^2\\) metrics of error.


```python
sns.scatterplot(x=y_pred_test, y=y_test, color="mediumvioletred", s=50)

plt.title("Linear regression - predict test set", fontsize=16)
plt.xlabel("y$^{\sf calc}$")
plt.ylabel("y$^{\sf true}$")
plt.show()

print('Mean squared error: %.2e' % mean_squared_error(y_test, y_pred_test))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred_test))
```


    
![png](S1_Regression_and_Analysis_files/S1_Regression_and_Analysis_80_0.png)
    


    Mean squared error: 5.45e-07
    Coefficient of determination: 0.87


#### 1.4.2.1 Other data considerations

* Do we need all the independent variables? 
  * Topics of interential statistics covered in a couple sessions
* Can we output integer quality scores? 
  * Topics of non-binary classification tasks covered in week 4

### 🍒 1.4.3 **Enrichment**: Other regression algorithms

There are many other regression algorithms the two we want to highlight here are Ridge, LASSO, and Elastic Net. They differ by an added term to the loss function. Let's review. Eq. 2 expanded to multivariate form yields:

$$\sum_{i=1}^{N}(y_i - \sum_{j=1}^{P}x_{ij}\beta_{j})^2$$

for Ridge regression, we add a **_regularization_** term known as **_L2_** regularization:

$$\sum_{i=1}^{N}(y_i - \sum_{j=1}^{P}x_{ij}\beta_{j})^2 + \lambda \sum_{j=1}^{P}\beta_{j}^2$$

for **_LASSO_** (Least Absolute Shrinkage and Selection Operator) we add **_L1_** regularization:

$$\sum_{i=1}^{N}(y_i - \sum_{j=1}^{P}x_{ij}\beta_{j})^2 + \lambda \sum_{j=1}^{P}|\beta_{j}|$$

The key difference here is that LASSO will allow coefficients to shrink to 0 while Ridge regression will not. **_Elastic Net_** is a combination of these two regularization methods.


```python
model = linear_model.Ridge()
model.fit(X_train, y_train)
print('Fit coefficients and intercept:\n\n', model.coef_, '\n\n', model.intercept_ )

# Predict on the test set
y_calc_test = model.predict(X_test)
```

    Fit coefficients and intercept:
    
     [ 1.61930554e-03  1.11227142e-04  2.64709094e-06  5.87271456e-04
      7.58510569e-05 -1.02851782e-04  8.76686650e-05  7.75641517e-04
      2.23315063e-04 -9.98653815e-04  5.26839010e-06] 
    
     0.9967531358810221



```python
sns.scatterplot(x=y_calc_test, y=y_test, color="lightseagreen", s=50)
plt.title("Ridge regression - predict test set",fontsize=16)
plt.xlabel("y$^{\sf calc}$")
plt.ylabel("y$^{\sf true}$")
plt.show()

print('Mean squared error: %.2f' % mean_squared_error(y_test, y_calc_test))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_calc_test))
```


    
![png](S1_Regression_and_Analysis_files/S1_Regression_and_Analysis_85_0.png)
    


    Mean squared error: 0.00
    Coefficient of determination: 0.87


#### 🏋️ Exercise 6: Tune Hyperparameter for Ridge Regression

Use the docstring to peak into the hyperparameters for Ridge Regression. What is the optimal value of lambda?

Plot the \\(\beta\\) values vs \\(\lambda\\) from the results of your analysis


```python
# cell for exercise 3
out_lambdas = []
out_coefs = []
out_scores = []
for i in range(10):
  lambdas = []
  coefs = []
  scores = []
  X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20)
  for lamb in range(1,int(5e3),20):
    model = linear_model.Ridge(alpha=lamb)
    model.fit(X_train, y_train)
    lambdas.append(lamb)
    coefs.append(model.coef_)
    scores.append(r2_score(y_test, model.predict(X_test)))
    # print('MSE: %.4f' % mean_squared_error(y_test, model.predict(X_test)))
    # print('R2: %.4f' % r2_score(y_test, model.predict(X_test)))
  out_lambdas.append(lambdas)
  out_coefs.append(coefs)
  out_scores.append(scores)
```


```python
coef_means = np.array(out_coefs).mean(axis=0)
coef_stds = np.array(out_coefs).std(axis=0)
results_means = pd.DataFrame(coef_means,columns=features)
results_stds = pd.DataFrame(coef_stds,columns=features)
results_means['lambda'] = [i for i in lambdas]
```


```python
fig, ax = plt.subplots(1,1,figsize=(10,10))
for feat in features:
  ax.errorbar([i for i in lambdas], results_means[feat], yerr=results_stds[feat], label=feat)
# results.plot('lambda', 'scores', ax=ax[1])
ax.legend()
```




    <matplotlib.legend.Legend at 0x7fdd72270fd0>




    
![png](S1_Regression_and_Analysis_files/S1_Regression_and_Analysis_89_1.png)
    



```python
results = pd.DataFrame(coefs,columns=features)
results['lambda'] = [i for i in lambdas]
results['scores'] = scores
```


```python
fig, ax = plt.subplots(1,2,figsize=(10,5))
for feat in features:
  results.plot('lambda', feat, ax=ax[0])
results.plot('lambda', 'scores', ax=ax[1])
```




    <AxesSubplot:xlabel='lambda'>




    
![png](S1_Regression_and_Analysis_files/S1_Regression_and_Analysis_91_1.png)
    


## 🍒 1.5 **Enrichment**: Additional Regression Exercises


### Problem 1) Number and choice of input features

* Load the red wine dataset and evaluate how the linear regression predictions changes as you change the **number and choice of input features**. The total number of columns in X  is 11 and each column represents a specific input feature. 

* Estimate the MSE


```python
print(X_train.shape)
```

    (1274, 11)


If you want to use the first 5 features you could proceed as following:


```python
X_train_five = X_train[:,0:5]
X_test_five = X_test[:,0:5]
```

Check that the new variables have the shape your expect


```python
print(X_train_five.shape)
print(X_test_five.shape)
```

    (1274, 5)
    (319, 5)


Now you can use these to train your linear regression model and repeat for different numbers or sets of input features! Note that you do not need to change the output feature! It's size is independent from the number of input features, yet recall that its length is the same as the number of values per input feature.

Questions to think about while you work on this problem
- How many input feature variables does one need? Is there a maximum or minimum number? 
- Could one input feature variable be better than the rest?
- What if values are missing for one of the input feature variables - is it still worth using it?
- Can you use **_L1_** or **_L2_** to determine these optimum features more quickly?

### Problem 2) Type of regression algorithm


Try using other types of linear regression methods on the wine dataset: the LASSO model and the Elastic net model which are described by the 

<code > sklearn.linear_model.ElasticNet() </code> <br>
<code > sklearn.linear_model.Lasso() </code>

scikit-learn functions.

For more detail see [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet) and [Lasso](  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso).

Questions to think about while you work on this problem
- How does the error change with each model?
- Which model seems to perform best?
- How can you optimize the hyperparameter, \\(\lambda\\)
- Does one model do better than the other at determining which input features are more important?
- How about non linear regression / what if the data does not follow a line?
- How do the bias and variance change for each model


```python
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
```


```python
for model in [ElasticNet, Lasso, Ridge, LinearRegression]:
  model = model()
  model.fit(X_train, y_train)
  print(str(model))
  print('Mean squared error: %.ef' % mean_squared_error(y_test, model.predict(X_test)))
  print('Coefficient of determination: %.2f' % r2_score(y_test, model.predict(X_test)))
  print()
```

    ElasticNet()
    Mean squared error: 4e-06f
    Coefficient of determination: -0.00
    
    Lasso()
    Mean squared error: 4e-06f
    Coefficient of determination: -0.00
    
    Ridge()
    Mean squared error: 7e-07f
    Coefficient of determination: 0.83
    
    LinearRegression()
    Mean squared error: 7e-07f
    Coefficient of determination: 0.83
    


<hr style="border:1px solid grey"> </hr>

# References

* **Linear Regression**
To find out more see [simple linear regression](https://en.wikipedia.org/wiki/Simple_linear_regression)

* **scikit-learn**
    * [Scikit-learn](https://scikit-learn.org/stable/)
    * [Linear regression in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    * [Metrics of error](https://scikit-learn.org/stable/modules/model_evaluation.html)
    * [The Boston dataset](https://scikit-learn.org/stable/datasets/index.html#boston-dataset)

* **Pearson correlation**
To find out more see [pearson](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

* **Irreducible error, bias and variance**
    * Great Coursera videos [here](https://www.coursera.org/lecture/ml-regression/irreducible-error-and-bias-qlMrZ)
and [here](https://www.coursera.org/lecture/ml-regression/variance-and-the-bias-variance-tradeoff-ZvP40)

