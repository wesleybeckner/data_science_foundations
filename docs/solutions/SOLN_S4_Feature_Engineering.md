<a href="https://colab.research.google.com/github/wesleybeckner/data_science_foundations/blob/main/notebooks/solutions/SOLN_S4_Feature_Engineering.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Data Science Foundations, Session 4: Feature Engineering

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com

---

<br>

In the previous session we talked about model pipelines and conveniently began with a suitable set of input data. In the real world, this is hardly ever the case! What is constant is this: at the end of the day, our models need numbers. Not only this, but a suitable set of numbers. What does that mean? The answer to that question is the subject of our session today.

<br>

---

<br>

<a name='top'></a>

<a name='2.0'></a>

## 4.0 Preparing Environment and Importing Data

[back to top](#top)

<a name='x.0.1'></a>

### 4.0.1 Import Packages

[back to top](#top)


```python
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import random
import scipy.stats as stats
from scipy.stats import gamma
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns; sns.set()
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error, r2_score
```

<a name='x.0.2'></a>

### 4.0.2 Load Dataset

[back to top](#top)


```python
margin = pd.read_csv('https://raw.githubusercontent.com/wesleybeckner/'\
        'ds_for_engineers/main/data/truffle_margin/truffle_margin_customer.csv')

orders = pd.read_csv('https://raw.githubusercontent.com/wesleybeckner/'\
                 'ds_for_engineers/main/data/truffle_margin/truffle_orders.csv')
time_cols = [i for i in orders.columns if '/' in i]
```


```python
margin.head()
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
      <th>Base Cake</th>
      <th>Truffle Type</th>
      <th>Primary Flavor</th>
      <th>Secondary Flavor</th>
      <th>Color Group</th>
      <th>Customer</th>
      <th>Date</th>
      <th>KG</th>
      <th>EBITDA/KG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Butter</td>
      <td>Candy Outer</td>
      <td>Butter Pecan</td>
      <td>Toffee</td>
      <td>Taupe</td>
      <td>Slugworth</td>
      <td>1/2020</td>
      <td>53770.342593</td>
      <td>0.500424</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Butter</td>
      <td>Candy Outer</td>
      <td>Ginger Lime</td>
      <td>Banana</td>
      <td>Amethyst</td>
      <td>Slugworth</td>
      <td>1/2020</td>
      <td>466477.578125</td>
      <td>0.220395</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Butter</td>
      <td>Candy Outer</td>
      <td>Ginger Lime</td>
      <td>Banana</td>
      <td>Burgundy</td>
      <td>Perk-a-Cola</td>
      <td>1/2020</td>
      <td>80801.728070</td>
      <td>0.171014</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Butter</td>
      <td>Candy Outer</td>
      <td>Ginger Lime</td>
      <td>Banana</td>
      <td>White</td>
      <td>Fickelgruber</td>
      <td>1/2020</td>
      <td>18046.111111</td>
      <td>0.233025</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Butter</td>
      <td>Candy Outer</td>
      <td>Ginger Lime</td>
      <td>Rum</td>
      <td>Amethyst</td>
      <td>Fickelgruber</td>
      <td>1/2020</td>
      <td>19147.454268</td>
      <td>0.480689</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfcat = margin.columns[:-2]
dfcat
```




    Index(['Base Cake', 'Truffle Type', 'Primary Flavor', 'Secondary Flavor',
           'Color Group', 'Customer', 'Date'],
          dtype='object')




```python
orders.head()
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
      <th>Base Cake</th>
      <th>Truffle Type</th>
      <th>Primary Flavor</th>
      <th>Secondary Flavor</th>
      <th>Color Group</th>
      <th>Customer</th>
      <th>1/2020</th>
      <th>2/2020</th>
      <th>3/2020</th>
      <th>4/2020</th>
      <th>5/2020</th>
      <th>6/2020</th>
      <th>7/2020</th>
      <th>8/2020</th>
      <th>9/2020</th>
      <th>10/2020</th>
      <th>11/2020</th>
      <th>12/2020</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Butter</td>
      <td>Candy Outer</td>
      <td>Butter Pecan</td>
      <td>Toffee</td>
      <td>Taupe</td>
      <td>Slugworth</td>
      <td>53770.342593</td>
      <td>40735.108025</td>
      <td>40735.108025</td>
      <td>40735.108025</td>
      <td>53770.342593</td>
      <td>40735.108025</td>
      <td>40735.108025</td>
      <td>40735.108025</td>
      <td>53770.342593</td>
      <td>40735.108025</td>
      <td>40735.108025</td>
      <td>40735.108025</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Butter</td>
      <td>Candy Outer</td>
      <td>Ginger Lime</td>
      <td>Banana</td>
      <td>Amethyst</td>
      <td>Slugworth</td>
      <td>466477.578125</td>
      <td>299024.088542</td>
      <td>466477.578125</td>
      <td>299024.088542</td>
      <td>466477.578125</td>
      <td>299024.088542</td>
      <td>466477.578125</td>
      <td>299024.088542</td>
      <td>466477.578125</td>
      <td>299024.088542</td>
      <td>466477.578125</td>
      <td>299024.088542</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Butter</td>
      <td>Candy Outer</td>
      <td>Ginger Lime</td>
      <td>Banana</td>
      <td>Burgundy</td>
      <td>Perk-a-Cola</td>
      <td>80801.728070</td>
      <td>51795.979532</td>
      <td>51795.979532</td>
      <td>51795.979532</td>
      <td>80801.728070</td>
      <td>51795.979532</td>
      <td>51795.979532</td>
      <td>51795.979532</td>
      <td>80801.728070</td>
      <td>51795.979532</td>
      <td>51795.979532</td>
      <td>51795.979532</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Butter</td>
      <td>Candy Outer</td>
      <td>Ginger Lime</td>
      <td>Banana</td>
      <td>White</td>
      <td>Fickelgruber</td>
      <td>18046.111111</td>
      <td>13671.296296</td>
      <td>13671.296296</td>
      <td>13671.296296</td>
      <td>18046.111111</td>
      <td>13671.296296</td>
      <td>13671.296296</td>
      <td>13671.296296</td>
      <td>18046.111111</td>
      <td>13671.296296</td>
      <td>13671.296296</td>
      <td>13671.296296</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Butter</td>
      <td>Candy Outer</td>
      <td>Ginger Lime</td>
      <td>Rum</td>
      <td>Amethyst</td>
      <td>Fickelgruber</td>
      <td>19147.454268</td>
      <td>12274.009146</td>
      <td>12274.009146</td>
      <td>12274.009146</td>
      <td>12274.009146</td>
      <td>12274.009146</td>
      <td>19147.454268</td>
      <td>12274.009146</td>
      <td>12274.009146</td>
      <td>12274.009146</td>
      <td>12274.009146</td>
      <td>12274.009146</td>
    </tr>
  </tbody>
</table>
</div>



<a name='2.1'></a>

## 4.1 Categorical Features

[back to top](#top)

At the end of the day, our algorithms operate on numerical values. How do you get from a series of string values to numerical values? 


```python
margin['Customer'].unique()
```




    array(['Slugworth', 'Perk-a-Cola', 'Fickelgruber', 'Zebrabar',
           "Dandy's Candies"], dtype=object)



A naive way to do it would be to assign a number to every entry
```
'Slugworth' = 1
'Perk-a-Cola' = 2
'Dandy's Candies' = 3
```
but we would inadvertently end up with some weird mathematical relationships between these variables, e.g. `Dandy's Candies - Perk-a-Cola = Slugworth` (3 - 2 = 1). 

A work around for this is to think *multi-dimensionally* we express our categorical values as vectors in a hyperspace where they cannot be expressed in terms of one another, i.e. they are *orthogonal* 
```
'Slugworth' = [1,0,0]
'Perk-a-Cola' = [0,1,0]
'Dandy's Candies' = [0,0,1]
```
such a scheme, in machine learning vernacular, is termed one-hot encoding. 

<a name='2.1.1'></a>

### 4.1.1 One-Hot Encoding

[back to top](#top)

sklearn has a couple useful libraries for one-hot encoding. let's start with the `OneHotEncoder` class in its `preprocessing` library


```python
from sklearn.preprocessing import OneHotEncoder
```


```python
# create the encoder object
enc = OneHotEncoder()

# grab the columns we want to convert from strings
X_cat = margin['Customer'].values.reshape(-1,1)

# fit our encoder to this data
enc.fit(X_cat)
```




    OneHotEncoder()



After fitting our encoder, we can then use this object to create our training array.


```python
# as a reference here's our original data
display(X_cat[:10])
print(X_cat.shape, end='\n\n')

onehotlabels = enc.transform(X_cat).toarray()
print(onehotlabels.shape, end='\n\n')

# And here is our new data
onehotlabels[:10]
```


    array([['Slugworth'],
           ['Slugworth'],
           ['Perk-a-Cola'],
           ['Fickelgruber'],
           ['Fickelgruber'],
           ['Fickelgruber'],
           ['Slugworth'],
           ['Zebrabar'],
           ['Slugworth'],
           ['Zebrabar']], dtype=object)


    (1668, 1)
    
    (1668, 5)
    





    array([[0., 0., 0., 1., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 1., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]])



We have our customer information one-hot encoded, we need to do this for all our variables and concatenate them with our regular numerical variables in our original dataframe.


```python
# create the encoder object
enc = OneHotEncoder()

# grab the columns we want to convert from strings
X_cat = margin[dfcat].values

# fit our encoder to this data
enc.fit(X_cat)
onehotlabels = enc.transform(X_cat).toarray()
```


```python
X_num = margin["KG"]
print(X_num.shape)
X = np.concatenate((onehotlabels, X_num.values.reshape(-1,1)),axis=1)
X.shape
```

    (1668,)





    (1668, 119)



And now we grab our EBITDA (margin) data for prediction


```python
y = margin["EBITDA/KG"]
```

#### 🏋️ Exercise 1: Create a simple linear model

Using the `X` and `y` sets, use `train_test_split` and `LinearRegression` to make a baseline model based on what we've learned so far.

Assess your model performance visually by plottying `y_test` vs `y_test_pred`


```python
# Cell for Exercise 1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
model = LinearRegression()
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(y_test, y_test_pred, ls='', marker='.')
```




    [<matplotlib.lines.Line2D at 0x7fa4b01f90d0>]




    
![png](SOLN_S4_Feature_Engineering_files/SOLN_S4_Feature_Engineering_24_1.png)
    


#### 🙋 Question 1:

> How can we assess the relative feature importance of the features in our model?

We could be tempted to inspect the coefficients (`linear.coef_`) of our  model to evaluate the relative feature importance, but in order to do this our features need to be scaled (so that the relative coefficient sizes are meaningful). What other issues might there be (think categorical vs continuous variables). 

<a name='2.2'></a>

## 4.2 Derived Features

[back to top](#top)

Can we recall an example of where we've seen this previously? That's right earlier on in our session on model selection and validation we derived some polynomial features to create our polynomial model using the linear regression class in sklearn. 

We actually see this a lot in engineering, where we will describe log relationships or some other transformation of the original variable. Actually let me see if I can find an example in my handy BSL...

<img src="https://raw.githubusercontent.com/wesleybeckner/ds_for_engineers/main/assets/C2/bird_stewart_lightfoot.jpg" width=500px></img>

<small>concentration profiles in continous stirred tank vs plug flow reactors. Notice the y-axis is log scale. Thanks Bird, Stewart, Lightfoot!</small>

> Can we think of other examples where we would like to derive features from our input data?

<a name='2.2.1'></a>

### 4.2.1 Creating Polynomials

[back to top](#top)

Let's revisit our example from the previous session, right before we introduced Grid Search in sklearn


```python
# from Model Selection and Validation, 1.2.1
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))
```

in the above, we use sklearn's convenient tool, `make_pipeline` to join together the preprocessing tool `PolynomialFeatures` and the basic model `LinearRegression`. Let's take a look at what PolynomialFeatures does to some simple data


```python
x = np.arange(1,11)
y = x**3
print(x)
print(y)
```

    [ 1  2  3  4  5  6  7  8  9 10]
    [   1    8   27   64  125  216  343  512  729 1000]



```python
features = PolynomialFeatures(degree=3)
X2 = features.fit_transform(x.reshape(-1,1))
```

we see our new feature set contains our original features, plus new features up to the nth-degree polynomial we set when creating the features object from `PolynomialFeatures`


```python
print(X2)
```

    [[   1.    1.    1.    1.]
     [   1.    2.    4.    8.]
     [   1.    3.    9.   27.]
     [   1.    4.   16.   64.]
     [   1.    5.   25.  125.]
     [   1.    6.   36.  216.]
     [   1.    7.   49.  343.]
     [   1.    8.   64.  512.]
     [   1.    9.   81.  729.]
     [   1.   10.  100. 1000.]]



```python
model = LinearRegression().fit(X2, y)
yhat = model.predict(X2)
plt.scatter(x, y)
plt.plot(x, yhat);
```


    
![png](SOLN_S4_Feature_Engineering_files/SOLN_S4_Feature_Engineering_34_0.png)
    


<a name='2.2.2'></a>

### 4.2.2 Dealing with Time Series

[back to top](#top)

Often, we will be dealing with time series data, whether its data generated by machinery, reactors, or sales and customers. In the following we discuss some simple practices for dealing with time series data.

<a name='2.2.2.1'></a>

#### 🍒 4.2.2.1 **Enrichment**: Fast Fourier Transform

[back to top](#top)

Sometimes we'll want to create a more sophisticated transformation of our input data. As engineers, this can often have to do with some empirical knowledge we understand about our process.

When working with equipment and machinery, we will often want to convert a signal from the time to frequency domain. Let's cover how we can do that with numpy!

<img src="https://www.nti-audio.com/portals/0/pic/news/FFT-Time-Frequency-View-540.png" width=400px></img>

<small>[img src](https://www.nti-audio.com/en/support/know-how/fast-fourier-transform-fft#:~:text=The%20%22Fast%20Fourier%20Transform%22%20(,frequency%20information%20about%20the%20signal.)</small>

What I've drawn here in the following is called a [square-wave signal](https://en.wikipedia.org/wiki/Square_wave)


```python
t = np.linspace(0,1,501)
# FFT should be given an integer number of cycles so we leave out last sample
t = t[:-1] 
f = 5 # linear frequency in Hz
w = f * 2 * np.pi # radial frequency
h = 4 # height of square wave
amp = 4 * h / np.pi
s = amp * (np.sin(w*t) + np.sin(3*w*t)/3 + np.sin(5*w*t)/5)

# here is the call to numpy FFT
F = np.fft.fft(s)
freq = np.fft.fftfreq(t.shape[-1], d=t[1])

# reorder frequency spectrum and frequency bins with 0 Hz at the center
F = np.fft.fftshift(F)
freq = np.fft.fftshift(freq)
# scale frequency spectrum to correct amplitude
F = F / t.size

# amplitudes
amps = [max(np.sin(w*t)), max(np.sin(w*t*3)/3), max(np.sin(w*t*5)/5)]

fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].plot(t,s)
ax[0].plot(t,amp * np.sin(w*t), ls='--')
ax[0].plot(t,amp * np.sin(w*t*3)/3, ls='--')
ax[0].plot(t,amp * np.sin(w*t*5)/5, ls='--')
ax[0].set_title('Time Domain')
ax[0].set_xlim(0, 1)
ax[0].set_xlabel('Time (s)')

# tells us about the amplitude of the component at the
# corresponding frequency. Multiplied by two because the
# signal power is split between (-) and (+) frequency branches
# of FFT, but we're only visualizing the (+) branch
magnitude = 2 * np.sqrt(F.real**2 + F.imag**2)

ax[1].plot(freq, magnitude)
ax[1].set_xlim(0, 30)
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_title('Frequency Domain')
```




    Text(0.5, 1.0, 'Frequency Domain')




    
![png](SOLN_S4_Feature_Engineering_files/SOLN_S4_Feature_Engineering_38_1.png)
    


<a name='x.2.2.2'></a>

#### 4.2.2.2 Rolling Windows

[back to top](#top)

> to see an example of this dataset in action [visit this link](http://truffletopia.azurewebsites.net/forecast/)

One powerful technique for dealing with time series data, is to create a rolling window of features based on the historical data. The proper window size can usually be determined by trial and error, or constraints around access to the data itself.

<p align=center>
<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/11/3hotmk.gif"></img>
</p>

In the above gif, we have a window size of 7. What that means is for whatever time step units we are in (that could be minutes, days, months, etc.) we will have 7 of them included in a single instance or observation. This instance or observation is then interpreted by our model and used to assess the target value, typically the quantity in the very next time step after the window (the green bar in the gif).

Let's take an example with the orders data


```python
tidy_orders = orders.melt(id_vars=orders.columns[:6], var_name='Date', value_name='KG')
display(tidy_orders.head())
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
      <th>Base Cake</th>
      <th>Truffle Type</th>
      <th>Primary Flavor</th>
      <th>Secondary Flavor</th>
      <th>Color Group</th>
      <th>Customer</th>
      <th>Date</th>
      <th>KG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Butter</td>
      <td>Candy Outer</td>
      <td>Butter Pecan</td>
      <td>Toffee</td>
      <td>Taupe</td>
      <td>Slugworth</td>
      <td>1/2020</td>
      <td>53770.342593</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Butter</td>
      <td>Candy Outer</td>
      <td>Ginger Lime</td>
      <td>Banana</td>
      <td>Amethyst</td>
      <td>Slugworth</td>
      <td>1/2020</td>
      <td>466477.578125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Butter</td>
      <td>Candy Outer</td>
      <td>Ginger Lime</td>
      <td>Banana</td>
      <td>Burgundy</td>
      <td>Perk-a-Cola</td>
      <td>1/2020</td>
      <td>80801.728070</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Butter</td>
      <td>Candy Outer</td>
      <td>Ginger Lime</td>
      <td>Banana</td>
      <td>White</td>
      <td>Fickelgruber</td>
      <td>1/2020</td>
      <td>18046.111111</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Butter</td>
      <td>Candy Outer</td>
      <td>Ginger Lime</td>
      <td>Rum</td>
      <td>Amethyst</td>
      <td>Fickelgruber</td>
      <td>1/2020</td>
      <td>19147.454268</td>
    </tr>
  </tbody>
</table>
</div>


In the next exercise, we are going to attempt to predict an order amount, based on the previous order history. We will scrub all categorical labels and only use historical amounts to inform our models. In effect the data that the model will see will look like the following:


```python
fig, ax = plt.subplots(3,2,figsize=(10,20))
indices = np.argwhere(ax)
color_dict = {0: 'tab:blue',
              1: 'tab:green',
              2: 'tab:orange',
              3: 'tab:red',
              4: 'tab:pink',
              5: 'tab:brown'}
for index, customer in enumerate(tidy_orders.Customer.unique()):
  orders.loc[orders.Customer == customer].iloc[:,6:].reset_index().T.plot(c=color_dict[index], legend=False,
                                                                          ax=ax[indices[index][0], indices[index][1]])
  ax[indices[index][0], indices[index][1]].set_title(customer)
```


    
![png](SOLN_S4_Feature_Engineering_files/SOLN_S4_Feature_Engineering_43_0.png)
    


What we may notice is that there is very little noise or drift in our order history, but there is certainly some periodicity. The question is can we use a linear model to predict the next order amount based on these history orders?

##### 🏋️ Exercise 2: Optimize Rolling Window Size for Customer Forecasts

For this exercise, you will use the `process_data` function below to help you optimize the `window` size for predicting the order quantity in any given month. You will train a `LinearRegression` model.

* create a model using a window size of 3 and predict the order quantity for the month immediately following the window
* create a model for window sizes 1-11 and report the \\(R^2\\) for each model


```python
def process_data(Xy, time_cols=12, window=3, remove_null=False):
  """
  This function splits your time series data into the proper windows

  Parameters
  ----------
  Xy: array
    The input data. If there are non-time series columns, assumes they are on
    the left and time columns are on the right. 
  time_cols: int
    The number of time columns, default 12
  window: int
    The time window size, default 3

  Returns
  -------
  X_: array
    The independent variables, includes time and non-time series columns with
    the new window
  y_: array
    The dependent variable, selected from the time columns at the end of the 
    window
  labels:
    The time series labels, can be used in subsequent plot
  """
  # separate the non-time series columns
  X_cat = Xy[:,:-time_cols]

  # select the columns to apply the sweeping window
  X = Xy[:,-time_cols:]

  X_ = []
  y = []

  for i in range(X.shape[1]-window):
    # after attaching the current window to the non-time series 
    # columns, add it to a growing list
    X_.append(np.concatenate((X_cat, X[:, i:i+window]), axis=1))

    # add the next time delta after the window to the list of y
    # values
    y.append(X[:, i+window])

  # X_ is 3D: [number of replicates from sweeping window,
  #           length of input data, 
  #           size of new feature with categories and time]
  # we want to reshape X_ so that the replicates due to the sweeping window is 
  # a part of the same dimension as the instances of the input data
  X_ = np.array(X_).reshape(X.shape[0]*np.array(X_).shape[0],window+X_cat.shape[1])
  y = np.array(y).reshape(X.shape[0]*np.array(y).shape[0],)

  if remove_null:
    # remove training data where the target is 0 (may be unfair advantage)
    X_ = X_[np.where(~np.isnan(y.astype(float)))[0]]
    y = y[np.where(~np.isnan(y.astype(float)))[0]]

  # create labels that show the previous month values used to train the model
  labels = []
  for row in X_:
    labels.append("X: {}".format(np.array2string(row[-window:].astype(float).round())))
  return X_, y, labels
```


```python
# Code Cell for Exercise 2
kg_month_data = orders.values[:,6:]

# use kg_month_data and the function process_data to create your X, y arrays
# then use train_test_split to create train and test portions
# USE y_test and  y_pred for your actual and true test data
# change only window parameter in process_data()
print("window   R2")
for window in range(1,12):
  ######################
  ### YOUR CODE HERE ###
  ######################
  X, y, labels = process_data(kg_month_data, time_cols=12, window=3)
  features = PolynomialFeatures(degree=4)
  X2 = features.fit_transform(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)
  model = LinearRegression()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  print("{},\t {:.2f}".format(window, r2_score(y_test, y_pred)))
```

    window   R2
    1,	 0.97
    2,	 0.88
    3,	 0.80
    4,	 0.96
    5,	 0.90
    6,	 0.86
    7,	 0.61
    8,	 0.89
    9,	 0.76
    10,	 0.74
    11,	 0.87



```python
#### RUN AFTER EXERCISE 2.2.2.2.1 ####
fig = px.scatter(x=y_test, y=y_pred,
                 labels={
                     "y": "Prediction",
                     "x": "Actual"
                 })
fig.update_layout(
  autosize=False,
  width=800,
  height=500,
  title='R2: {:.3f}'.format(r2_score(y_test, y_pred))
  )
```


<div>                            <div id="239a6e0b-8482-49c8-9619-ebde564b91cd" class="plotly-graph-div" style="height:500px; width:800px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("239a6e0b-8482-49c8-9619-ebde564b91cd")) {                    Plotly.newPlot(                        "239a6e0b-8482-49c8-9619-ebde564b91cd",                        [{"hovertemplate":"Actual=%{x}<br>Prediction=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"","orientation":"v","showlegend":false,"x":[4528.380503144654,64.01384083044982,15617.489114658929,4528.380503144654,5698.392857142857,124348.89322916669,21939.111111111117,5712.33671988389,64983.42138364781,1483.5,93302.72435897436,1236.25,773.5849056603774,518.4642857142858,3798.928571428572,13671.296296296296,188.7692307692308,4057.832278481013,101149.25,14626.074074074077,12274.009146341465,141456.76315789475,4793.829296424452,74.78571428571429,10861.245674740485,86.24595469255664,19129.33333333333,39307.04113924051,74.78571428571429,46247.0,188.7692307692308,85.32258064516128,90677.41228070176,9306.611111111111,220.55016181229772,466477.57812500006,1158.125,220.55016181229772,1901.1290322580644,19129.33333333333,2637.8993710691825,1868.867924528302,3798.928571428572,2252.261437908497,5222.5,181.5552699228792,5038.387096774193,120.98615916955016,10146.898432174508,85.32258064516128,286.6666666666667,18613.222222222223,818.1891025641025,2499.201741654572,2010.9477124183009,6984.389273356401,42704.25889967639,11715.487421383648,39307.04113924051,440.859375,428.0,96.59546925566345,102.64285714285714,3569.0359477124184,13671.296296296296,773.5849056603774,2010.9477124183009,42704.25889967639,40608.41346153846,83347.75862068967,10796.31640058055,74.78571428571429,80801.72807017544,2540.0,5222.5,131.55526992287918,299024.0885416667,3882.680981595092,895.9082278481012,1180.5379746835442,15617.489114658929,85.32258064516128,6185.59509202454,16749.13157894737,47142.92857142857,3786.966463414634,936.280487804878,15244.632812500002,3272.80163599182,1271.904761904762,47142.92857142857,356.6666666666667,10861.245674740485,108.64285714285714,21438.1875,16918.132716049382,7892.142857142857,1901.1290322580644,2647.5,5038.387096774193,1677.5078125000002,39307.04113924051,51795.97953216374,916.3717948717948,41871.69230769231,12870.766853932584,1271.904761904762,181.5552699228792,211540.7575757576,3798.928571428572,2054.328561690525,3578.592162554427,2854.854368932039,51795.97953216374,11791.8,149.57142857142858,103.3653846153846,119015.43674698794,2499.201741654572,19286.94968553459,2185.3410740203194,19569.69230769231,12317.397660818711,6965.361445783132,86.24595469255664,3071.4691011235955,13867.125,50574.625,14626.074074074077,1005.4545454545454,16918.132716049382,2543.106741573034,10245.87593728698,8608.709677419354,150.3654970760234,3517.9333333333334,14626.074074074077,3177.0,2499.201741654572,345.6428571428572,10736.622807017544,13799.769319492503,12691.209677419354,11.175616835994196,2010.9477124183009,10736.622807017544,71138.51966292135,8104.40251572327,62.168396770472896,211540.7575757576,16569.261006289307,445.54838709677415,16918.132716049382,42.484177215189874,150.3654970760234,11.175616835994196,299024.0885416667,579.0625,10146.898432174508,181.5552699228792,494.57142857142856,411.1303344867359,2355.267295597484,494.57142857142856,79091.23475609756,10146.898432174508,1005.4545454545454,5038.387096774193,191.11111111111111,9304.668674698794,19569.69230769231,8250.491573033707,466477.57812500006,3048.0,1630.1966292134832,64.01384083044982,3569.0359477124184,1901.1290322580644,3272.80163599182,9772.200520833334,2185.3410740203194,28694.0,314.3203883495146,8104.40251572327,18557.572327044025,2231.547169811321,1432.2712418300653,773.5849056603774,40735.10802469136,314.3203883495146,471.91011235955057,42.484177215189874,9194.277108433736,8173.714285714285,837.8787878787879,1604.1437908496732,22.50980392156863,14.663461538461538,663176.1973875181,2854.854368932039,428.0,43.64516129032258,1526.2857142857142,1432.2712418300653,5456.578050443081,22331.935185185182,150.3654970760234,4057.832278481013,1868.867924528302,1630.1966292134832,4723.49129172714,5038.387096774193,19364.70552147239,117.22222222222224,15110.75,4057.832278481013,83347.75862068967,7892.142857142857,19129.33333333333,1968.8904494382025,207.39062500000003,62.168396770472896,3578.592162554427,1677.5078125000002,2499.201741654572,1254.4444444444443,1236.25,3578.592162554427,1992.4528301886792,14626.074074074077,1236.25,6965.361445783132,36.37096774193548,40735.10802469136,20.098039215686278,1432.2712418300653,538.8571428571429,101149.25,93302.72435897436,181.5552699228792,28694.0,2054.328561690525,5872.384615384615,31412.04644412192,5872.384615384615,854.952380952381,31412.04644412192,1253403.0130624091,63142.24137931035,20.098039215686278,3569.0359477124184,16569.261006289307,19286.94968553459,19286.94968553459,837.8787878787879,3665.809768637532,108.64285714285714,56.07911392405064,13671.296296296296,371.2903225806451,4057.832278481013,47142.92857142857,21438.1875,54.833333333333336,26081.56401384083,2540.0,115.76923076923076,16.423076923076923,8608.709677419354,4463.780120481927,8250.491573033707,15110.75,8173.714285714285,157100.37650602407,104499.0512820513,9076.930817610064,2611.25,428.0,14.663461538461538,2694.0,3569.0359477124184,86.24595469255664,678.7183544303797,494.57142857142856,712.4603174603175,663176.1973875181,10245.87593728698,5071.786163522012,1868.867924528302,26081.56401384083,3403.344867358708,4793.829296424452,64983.42138364781,3272.80163599182,5712.33671988389,9194.277108433736,608.4770114942529,42.484177215189874,466477.57812500006,3695.444059976932,3517.9333333333334,345.6428571428572,579.0625,7892.142857142857,5038.387096774193,45601.61516853933,1992.4528301886792,2647.5,5872.384615384615,6965.361445783132,64.01384083044982,45601.61516853933,23123.5,1992.4528301886792,2540.0,9060.337370242214,14.663461538461538,3882.680981595092,36.37096774193548,193984.2734375,2231.547169811321,108.64285714285714,329.7142857142857,117.49826989619376,773.5849056603774,36.37096774193548,8608.709677419354,371.2903225806451,45481.42307692308,10245.87593728698,63142.24137931035,678.7183544303797,11555.9375,4528.380503144654,2499.201741654572,10736.622807017544,5222.5,1901.1290322580644,93302.72435897436,1702.1929824561405,114208.8534107402,343.1394601542417,10245.87593728698,8173.714285714285,140637.14285714287,132.94270833333334,579.0625,663176.1973875181,1180.5379746835442,1236.25,26081.56401384083,854.952380952381,2010.9477124183009,30698.85714285714,11.175616835994196,90677.41228070176,38128.80258899677,663176.1973875181,71138.51966292135,1236.25,31412.04644412192,50574.625,83347.75862068967,12274.009146341465,90677.41228070176,808.2857142857143,11.175616835994196,11555.9375,86.24595469255664,41871.69230769231,19129.33333333333,329.7142857142857,1236.25,2637.8993710691825,579.7777777777778,30698.85714285714,329.7142857142857,2647.5,466477.57812500006,102.64285714285714,12691.209677419354,12317.397660818711,220.55016181229772,19147.454268292684,9304.668674698794,4057.832278481013,47142.92857142857,1630.1966292134832,4463.780120481927,2002.1844660194176,5071.786163522012,7035.866666666667,13504.20634920635,21601.383647798742,10245.87593728698,10861.245674740485,176.36477987421384,12691.209677419354,1432.2712418300653,608.4770114942529,10736.622807017544,3695.444059976932,157100.37650602407,1702.1929824561405,51795.97953216374,2386.449438202247,117.22222222222224,120.98615916955016,777.0363321799308,12274.009146341465,2611.25,2242.446601941748,168764.57142857145,627.2222222222222,40608.41346153846,2002.1844660194176,157.46855345911948,2647.5,119015.43674698794,579.0625,329.7142857142857,13121.345911949686,71138.51966292135,207.39062500000003,30698.85714285714,9060.337370242214,1529.7752808988764,3071.4691011235955,46247.0,538.8571428571429,16347.42857142857,23123.5,132.94270833333334,6984.389273356401,12691.209677419354,2499.201741654572,30221.5,15229.451612903224,191.11111111111111,428.0,3578.592162554427,11715.487421383648,19129.33333333333,30221.5,6432.321799307958,41871.69230769231,42.484177215189874,16918.132716049382,3695.444059976932,13504.20634920635,15617.489114658929,117.49826989619376,16569.261006289307,96.59546925566345,678.7183544303797,10146.898432174508,678.7183544303797,131.55526992287918,47142.92857142857,79091.23475609756,736.1797752808989,1180.5379746835442,29354.53846153846,45481.42307692308,21438.1875,6965.361445783132,31412.04644412192,10146.898432174508,11791.8,15617.489114658929,1677.5078125000002,5456.578050443081,14626.074074074077,94285.85714285714,248.6394601542416,5456.578050443081,26081.56401384083,63142.24137931035,12317.397660818711,12317.397660818711,608.4770114942529,11.175616835994196,10861.245674740485,2566.6935483870966,11555.9375,678.7183544303797,1529.7752808988764,1432.2712418300653,518.4642857142858,8104.40251572327,63142.24137931035,3798.928571428572,4057.832278481013,1526.2857142857142,2854.854368932039,3403.344867358708,9306.611111111111,538.8571428571429,13671.296296296296,117.22222222222224,343.1394601542417],"xaxis":"x","y":[4785.592155979585,378.9394302913829,15365.600809369838,4785.592155979585,4070.255357333951,134757.3771732384,14409.557558767568,5813.845282617682,61516.49587090381,1510.6755468797405,98738.27282520698,1684.2601507265988,1088.107600326137,647.864369876902,4263.755978756964,14168.403568365671,487.3481032940997,4287.792726026477,51779.415544128475,19934.734830917627,12141.409803201397,87747.52966460225,5818.94510670397,377.43130038813075,12797.394868451114,389.0359797814606,19263.507069512012,40163.79013647013,377.43130038813075,23840.068464074022,487.3481032940997,400.48488833402286,126112.56377290396,11223.080944977459,522.1062342859491,415175.9779430929,894.6750043489748,519.4101103091041,2138.6141864173633,19263.507069512012,2790.085070463646,2196.4290704108694,4263.755978756964,2244.514746383961,2963.0052419712265,602.4723068239653,5217.814575086902,410.08771833436947,16913.133647514336,390.2429375597753,489.6064298303322,16311.253798810561,1109.562627058514,2715.348806915594,2281.9990642021894,6353.789558047462,37073.75373050823,11896.354947646087,38882.42895370776,797.0601491084628,703.1493477716122,388.482661364839,481.8437277075316,3747.013377347842,16794.12337436424,1571.4700357476117,2257.4161561731,37073.75373050823,43146.57407027896,76460.52042984031,5813.845282617682,433.93372568151483,50253.31197444373,2781.8507793892813,2963.0052419712265,450.36262704088375,323624.6828685297,2286.347850033103,959.8167805278175,1463.9293549917084,25867.060321463283,388.50455558540244,3461.34813854915,10658.874464205835,48286.68591679114,4070.54807420052,1208.1895902424544,13863.416557439898,3913.805102706469,1545.439458867743,48286.68591679114,664.1495092797293,12797.394868451114,415.88888264864494,24308.715376885695,17460.789585310762,13878.585909613186,2138.6141864173633,3258.406190332729,5925.265679815453,1797.2371240684145,47713.131047358365,51804.0528093027,1099.5606427530397,40683.104312073505,8508.45844428347,1571.3535403980425,505.4911026847705,213132.54291644722,4263.755978756964,2570.353838719455,3926.506181336219,3317.1418553360736,54758.90192605476,20585.396531692135,377.43130038813075,406.91804655949966,120990.40817472423,4395.853637740449,19387.390319584494,2412.686086825897,19699.926384269966,12552.025861937103,7368.391717428835,390.09029678781303,2262.905415567944,11572.521113313724,56931.5110267007,15545.52362930884,1239.906233381196,17460.789585310762,1926.1552505875645,10673.158532726991,9907.699090223898,454.81639486308137,6355.618165841122,19934.734830917627,2858.35606741023,2834.2664976391197,665.469888483012,11592.823783640883,16177.143075924467,12543.725418313918,317.63569157648635,2257.4161561731,15201.471014315619,45645.23591676211,8323.62833082682,376.8168895165831,213132.54291644722,16698.569941711827,663.3571920970296,20710.099320853376,346.28222029939593,513.9332229141203,323.60536008687086,323624.6828685297,1301.212252741579,16913.133647514336,505.4911026847705,632.0783029798903,778.1760841513764,2635.5625663138067,632.0783029798903,76574.72476531091,11492.974658748008,1239.906233381196,5163.94089137517,494.71513791291954,11527.578799029196,19176.79696136619,8979.131225405608,415175.9779430929,2754.6914615362707,2567.0739167219367,378.9394302913829,3747.013377347842,2138.6141864173633,3913.805102706469,10871.491666793676,2516.669574686992,18752.149151877966,636.9170394143573,8323.62833082682,17785.632844249434,2407.321312452001,1686.4843440391876,1051.2986748894868,49435.50780135217,610.4363857799107,960.0492155948752,346.28222029939593,7141.329503589847,8187.405610699661,1148.288091129662,1686.4843440391876,324.6947768318741,320.7835266802878,639820.6545150073,3317.1418553360736,703.1493477716122,340.3871184326995,1531.8394329987157,1686.4843440391876,9236.305456033022,16619.83993793382,463.3944227377964,4287.792726026477,3364.1619784229933,2019.1544585081797,2715.348806915594,5163.94089137517,10185.63650295469,436.06660273854754,26293.480165383102,4218.370391131691,76460.52042984031,7915.880111460361,25978.487349635714,2375.22637673234,489.7605752448112,376.8168895165831,3756.2286447454576,1797.2371240684145,3060.8574440447824,1384.0395692783663,1535.8631900519042,6162.533552625073,2276.6020512892946,14800.536096877271,1535.8631900519042,7368.391717428835,340.3871184326995,40283.97684677175,325.06940680600604,1695.6732046475156,1028.5053850528614,51779.415544128475,92018.35013115287,505.4911026847705,18752.149151877966,3667.712553203018,5968.182975715722,34939.27854355379,6424.274585977019,992.354894936303,30596.638581578518,639820.6545150073,65413.624508456036,324.82371754886964,3747.013377347842,16698.569941711827,19387.390319584494,19387.390319584494,1148.288091129662,6305.285970941779,492.1628050669814,346.28222029939593,14168.403568365671,663.3571920970296,4218.370391131691,45766.272960899,37175.69967364701,369.63985077669076,22891.92825541111,2754.6914615362707,405.6544524204279,319.54816919005003,8698.931074864986,5689.033000226852,11752.173931839643,26293.480165383102,8187.405610699661,117110.65135820676,90877.76737128492,8855.335527589015,3229.016296504339,703.1493477716122,319.72742324233695,1604.2554168141014,3769.9108375996702,396.30197122916206,959.8167805278175,632.0783029798903,999.9729800621147,731503.1707028296,17075.133945943726,5082.686543693957,2107.5040932279385,22891.92825541111,4219.676962291559,5818.94510670397,61516.49587090381,3617.0754679045876,5813.845282617682,7141.329503589847,932.7370475860574,346.28222029939593,415175.9779430929,4555.635369302835,3885.8140646224183,769.1955699368395,953.6647702716481,8337.819513819635,5163.94089137517,48246.709882894014,2276.6020512892946,2886.664844581823,5968.182975715722,7368.391717428835,378.9394302913829,63573.70285359673,40074.17801737161,2276.6020512892946,3138.4978117622,8151.559019833033,319.72742324233695,2286.347850033103,340.3871184326995,172828.89934304,2407.321312452001,410.08048522816836,747.818087951304,407.0672088952674,1088.107600326137,341.517050774033,8698.931074864986,667.3272676429804,39725.42364104348,10673.158532726991,65413.624508456036,959.8167805278175,13195.11592262944,4785.592155979585,2834.2664976391197,15201.471014315619,2963.0052419712265,2425.884027039221,98738.27282520698,1997.7384398419902,115873.73047883328,489.0303345318461,10673.158532726991,9457.064447629586,137428.5757499219,449.0579108608325,1301.212252741579,1085751.3630379443,1502.4134405828336,1684.2601507265988,22891.92825541111,992.354894936303,2244.514746383961,31550.15438840883,316.0906893952995,90462.3574812228,37318.37237669782,639820.6545150073,45645.23591676211,1684.2601507265988,30596.638581578518,56931.5110267007,76460.52042984031,12141.409803201397,95635.30862544454,839.3500692330289,316.62245060704845,13195.11592262944,389.0359797814606,41802.402147140514,19263.507069512012,747.818087951304,1535.8631900519042,2790.085070463646,803.8781088470796,29908.894508749334,648.872490598366,2858.35606741023,415175.9779430929,409.7821718132848,12543.725418313918,17394.662595168105,522.1062342859491,12141.409803201397,9740.51689224938,5199.4234913728005,45766.272960899,2567.0739167219367,4831.715916465941,2417.5880406486763,5082.686543693957,3697.7340063828487,13472.111607949182,20652.753180263422,11602.10423834572,12797.394868451114,471.44073265789405,12543.725418313918,1695.6732046475156,932.7370475860574,10980.322568203586,4555.635369302835,117110.65135820676,2094.8448847085992,54758.90192605476,1826.31018819446,421.9954832228255,410.08771833436947,978.2266656454766,12141.409803201397,4796.262256309382,2236.064155834039,135924.7927322855,1041.1204286449113,40221.84276082843,2273.3851256392354,461.10966099955675,2858.35606741023,143848.60557702882,953.6647702716481,648.872490598366,12664.975331868296,45645.23591676211,489.7605752448112,29908.894508749334,8151.559019833033,2427.747778574022,2262.905415567944,23840.068464074022,866.7970691016989,8187.405610699661,26195.686125920434,449.0579108608325,6353.789558047462,12679.428015781728,2715.348806915594,15684.810589735318,12543.725418313918,504.4494927620649,703.1493477716122,4250.960414621229,11896.354947646087,25978.487349635714,15684.810589735318,5875.699548389586,41802.402147140514,347.0090494475603,16909.279272683176,4555.635369302835,15368.26716660901,16108.716384819742,407.0672088952674,16698.569941711827,388.482661364839,993.553834752781,10573.002781125038,971.4284514069652,450.36262704088375,45766.272960899,76574.72476531091,774.5157686911542,1443.732437768414,19176.79696136619,39725.42364104348,24308.715376885695,8706.164457317907,32091.295039989156,11492.974658748008,13508.076892121393,16108.716384819742,1797.2371240684145,5826.8479598607,15545.52362930884,45766.272960899,438.4351253031718,5826.8479598607,22891.92825541111,65413.624508456036,12552.025861937103,12552.025861937103,932.7370475860574,316.0906893952995,12797.394868451114,2860.1717851882872,11807.964454280216,1123.9090100996427,1913.5805743985902,1816.3382332975395,647.864369876902,8323.62833082682,65413.624508456036,5403.795319141717,4218.370391131691,1531.8394329987157,3111.527273406017,4219.676962291559,11223.080944977459,866.7970691016989,13722.736128528055,436.06660273854754,489.0303345318461],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Actual"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Prediction"}},"legend":{"tracegroupgap":0},"margin":{"t":60},"autosize":false,"width":800,"height":500,"title":{"text":"R2: 0.869"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('239a6e0b-8482-49c8-9619-ebde564b91cd');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


<a name='2.2.3'></a>

### 4.2.3 Image Preprocessing

[back to top](#top)

Image preprocessing is beyond the scope of this session. We cover this topic in [General Applications of Neural Networks](https://wesleybeckner.github.io/general_applications_of_neural_networks/S3_Computer_Vision_I/). For now, know that there is a wealth of considerations for how to handle images, and they all fit within the realm of feature engineering.

<a name='2.3'></a>

## 4.3 Transformed Features

[back to top](#top)

*Transformed* features, are features that we would like to augment based on their relationship within their own distribution or to other (allegedly) independent data within our training set. e.g. we're not *deriving* new features based on some empirical knowledge of the data, rather we are changing them due to statistical properties that we can assess based on the data itself.


<a name='2.3.1'></a>

### 4.3.1 Skewness

[back to top](#top)

<img src="https://i.pinimg.com/originals/d1/9f/7c/d19f7c7f5daaed737ab2516decea9874.png" width=400px></img>

Skewed data can lead to imbalances in our model prediction. Why? Skewed values in the distribution will bias the mean. When assigning weights to this input feature, therefore, the model will give preferential treatment to these values.

To demonstrate, I'm going to use scipy to create some skewed data.


```python
from scipy.stats import skewnorm
```


```python
a = 10
x = np.linspace(skewnorm.ppf(0.01, a),
                skewnorm.ppf(0.99, a), 100)
plt.plot(x, skewnorm.pdf(x, a),
       'r-', lw=5, alpha=0.6, label='skewnorm pdf')
```




    [<matplotlib.lines.Line2D at 0x7fa48d14a130>]




    
![png](SOLN_S4_Feature_Engineering_files/SOLN_S4_Feature_Engineering_53_1.png)
    


We can now generate a random population based on this distribution


```python
r = skewnorm.rvs(a, size=1000)
plt.hist(r)
```




    (array([113., 267., 225., 172., 116.,  62.,  26.,  13.,   2.,   4.]),
     array([-0.19733964,  0.15303313,  0.50340589,  0.85377866,  1.20415142,
             1.55452419,  1.90489696,  2.25526972,  2.60564249,  2.95601526,
             3.30638802]),
     <BarContainer object of 10 artists>)




    
![png](SOLN_S4_Feature_Engineering_files/SOLN_S4_Feature_Engineering_55_1.png)
    


Unskewed data will return something close to 0 from calling `df.skew()`. When dealing with actual data, we can use `df.skew()` to determine whether we should transform our data. 


```python
x = pd.DataFrame(r, columns=['Skewed Data'])
x['Skewed Data'].skew()
```




    0.9141902067398219



There are a handful of ways to deal with skewed data:

* log transform
* square root transform
* Box-Cox transform

Let's try the first two


```python
print('square root transformed skew: {:.4f}'.format(np.sqrt(x['Skewed Data']).skew()))
print('log transformed skew: {:.4f}'.format(np.log(x['Skewed Data']).skew()))
fig, ax = plt.subplots(1, 1, figsize=(10,10))
ax.hist(x['Skewed Data'], alpha=0.5, label='original: {:.2f}'.
        format((x['Skewed Data']).skew()))
ax.hist(np.sqrt(x['Skewed Data']), alpha=0.5, label='sqrt: {:.2f}'.
        format(np.sqrt(x['Skewed Data']).skew()))
ax.hist(np.log(x['Skewed Data']), alpha=0.5, label='log: {:.2f}'.
        format(np.log(x['Skewed Data']).skew()))
ax.legend()
```

    square root transformed skew: 0.0561
    log transformed skew: -1.6916


    /home/wbeckner/anaconda3/envs/py39/lib/python3.9/site-packages/pandas/core/arraylike.py:364: RuntimeWarning:
    
    invalid value encountered in sqrt
    
    /home/wbeckner/anaconda3/envs/py39/lib/python3.9/site-packages/pandas/core/arraylike.py:364: RuntimeWarning:
    
    invalid value encountered in log
    





    <matplotlib.legend.Legend at 0x7fa49157d340>




    
![png](SOLN_S4_Feature_Engineering_files/SOLN_S4_Feature_Engineering_59_3.png)
    


We see we didn't get much traction with the log transform, and the log transform will not be able to handle 0 values, and so we will sometimes have to code exceptions for those. 

Box-Cox is often a good route to go, but it has the added restriction that the data has to all be above 0.

Let's create a new distribution with this added restriction


```python
a = 6
r = skewnorm.rvs(a, size=1000)
r = [i for i in r if i > 0]
plt.hist(r)
```




    (array([220., 277., 182., 127.,  66.,  39.,  17.,   5.,   4.,   2.]),
     array([2.17150536e-03, 3.88613862e-01, 7.75056219e-01, 1.16149858e+00,
            1.54794093e+00, 1.93438329e+00, 2.32082565e+00, 2.70726800e+00,
            3.09371036e+00, 3.48015272e+00, 3.86659507e+00]),
     <BarContainer object of 10 artists>)




    
![png](SOLN_S4_Feature_Engineering_files/SOLN_S4_Feature_Engineering_61_1.png)
    



```python
from scipy import stats

x = pd.DataFrame(r, columns=['Skewed Data'])
fig, ax = plt.subplots(1, 1, figsize=(10,10))
ax.hist(x['Skewed Data'], alpha=0.5, label='original: {:.2f}'.
        format((x['Skewed Data']).skew()))
ax.hist(np.sqrt(x['Skewed Data']), alpha=0.5, label='sqrt: {:.2f}'.
        format(np.sqrt(x['Skewed Data']).skew()))
ax.hist(np.log(x['Skewed Data']), alpha=0.5, label='log: {:.2f}'.
        format(np.log(x['Skewed Data']).skew()))
ax.hist(stats.boxcox(x['Skewed Data'])[0], alpha=0.5, label='box-cox: {:.2f}'.
        format(pd.DataFrame(stats.boxcox(x['Skewed Data'])[0])[0].skew()))
ax.legend()

```




    <matplotlib.legend.Legend at 0x7fa492058f40>




    
![png](SOLN_S4_Feature_Engineering_files/SOLN_S4_Feature_Engineering_62_1.png)
    


#### 🏋️ Exercise 3: Transform data from a gamma distribution

Repeat section 2.3.1, this time synthesizing a gamma distribution and transforming it. Which transformation best reduces the skew? Do this for a dataset that does not contain values at or below 0.


```python
# code cell for exercise 3
from scipy.stats import gamma
a = 6
r = gamma.rvs(a, size=1000)
r = [i for i in r if i > 0]

x = pd.DataFrame(r, columns=['Skewed Data'])
fig, ax = plt.subplots(1, 1, figsize=(10,10))
ax.hist(x['Skewed Data'], alpha=0.5, label='original: {:.2f}'.
        format((x['Skewed Data']).skew()))
ax.hist(np.sqrt(x['Skewed Data']), alpha=0.5, label='sqrt: {:.2f}'.
        format(np.sqrt(x['Skewed Data']).skew()))
ax.hist(np.log(x['Skewed Data']), alpha=0.5, label='log: {:.2f}'.
        format(np.log(x['Skewed Data']).skew()))
ax.hist(stats.boxcox(x['Skewed Data'])[0], alpha=0.5, label='box-cox: {:.2f}'.
        format(pd.DataFrame(stats.boxcox(x['Skewed Data'])[0])[0].skew()))
ax.legend()
```




    <matplotlib.legend.Legend at 0x7fa49111dd30>




    
![png](SOLN_S4_Feature_Engineering_files/SOLN_S4_Feature_Engineering_64_1.png)
    


<a name='2.3.2'></a>

### 4.3.2 Colinearity

[back to top](#top)

Colinearity can also affect the performance of your machine learning model. In particular, if features are colinear, it can be easy for your model to overfit to your training dataset. This is often mitigated by regularization. If you're curious you can read more about it on [this discussion from StackExchange](https://stats.stackexchange.com/questions/168622/why-is-multicollinearity-not-checked-in-modern-statistics-machine-learning). We will still explore it explicitly here by calculating the [Variance Inflation Factor](https://www.statsmodels.org/stable/_modules/statsmodels/stats/outliers_influence.html#variance_inflation_factor) (VIF) on some hypothetical data.

$$ VIF = \frac{1}{1-R^2}$$

Usually we are concerned about data with a VIF above 10



<a name='x.3.2.1'></a>

#### 4.3.2.1 Detecting Colinearity

[back to top](#top)


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

**Step 1: Make some data**


```python
# we can throttle the error rate
random.seed(42)

# x2 will be sqrt of x1 plus some error
def func(x, err):
  return x**.5 + (err * random.randint(-1,1) * random.random() * x)
  
x0 = range(100)
x1 = [func(i, .05) for i in x0] # HIGH degree of colinearity with x0
x2 = [func(i, 1) for i in x0] # MED degree of colinearity with x0
x3 = [random.randint(0,100) for i in x0] # NO degree of colinearity with x0

# take a look
fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.plot(x0, x1, label='x1')
ax.plot(x0, x2, label='x2')
ax.plot(x0, x3, label='x3')
ax.legend()
```




    <matplotlib.legend.Legend at 0x7fa49121ca90>




    
![png](SOLN_S4_Feature_Engineering_files/SOLN_S4_Feature_Engineering_69_1.png)
    


To calculate the colinearities I'm going to aggregate these x's into a dataframe:


```python
colin = pd.DataFrame([x0,x1,x2,x3]).T
colin.columns = ['x0','x1','x2','x3']
colin.head()
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
      <th>x0</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.013751</td>
      <td>0.721523</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>1.400260</td>
      <td>1.414214</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>1.630546</td>
      <td>-0.438007</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>2.017388</td>
      <td>4.304847</td>
      <td>24.0</td>
    </tr>
  </tbody>
</table>
</div>



**Step 2: Calculate VIF factors**


```python
# calculate VIF factors
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(colin.values, i) for i in 
                     range(colin.shape[1])]
vif["features"] = colin.columns
```

**Step 3: Inspect VIF factors**


```python
# inspect VIF factors
display(vif)
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
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.555415</td>
      <td>x0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.823872</td>
      <td>x1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.030609</td>
      <td>x2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.559468</td>
      <td>x3</td>
    </tr>
  </tbody>
</table>
</div>


In this case, we may remove either `x0` or `x1` from the dataset.

<a name='x.3.2.2'></a>

#### 4.3.2.2 Fixing Colinearity

[back to top](#top)

It is good to aknowledge where colinearity exists as this will influence the interpretability of your model. In most cases, however, it won't have a heavy influence on the performance of your model. 

A simple method of dealing with colinearity, is to remove the highest VIF features from your model, iteratively, assessing the performance and determining whether to keep the variable or not. 

Another method is to create some linear combination of the correlated variables. This is encapsulated in the section on dimensionality reduction.

<a name='2.3.3'></a>

### 4.3.3 Normalization

[back to top](#top)

Many learning algorithms require zero mean and unit variance to behave optimally. Sklearn preprocessing library contains a very usefull class, `StandardScaler` for handling this automatically for us.



```python
from sklearn.preprocessing import StandardScaler
```


```python
scaler = StandardScaler()
normed = scaler.fit_transform(colin)
```


```python
colin[['x0','x1','x2','x3']].plot(kind='kde')
```




    <AxesSubplot:ylabel='Density'>




    
![png](SOLN_S4_Feature_Engineering_files/SOLN_S4_Feature_Engineering_81_1.png)
    



```python
pd.DataFrame(normed, columns = [['x0','x1','x2','x3']]).plot(kind='kde')
```




    <AxesSubplot:ylabel='Density'>




    
![png](SOLN_S4_Feature_Engineering_files/SOLN_S4_Feature_Engineering_82_1.png)
    


#### 🏋️ Exercise 4: Normalization affect on VIF 

In the above, we saw how to scale and center variables. How does this affect VIF? 

* Calculate the VIF for the scaled-centered data


```python
# Code Cell for Exercise 4
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(normed, i) for i in 
                     range(normed.shape[1])]
vif["features"] = colin.columns
display(vif)
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
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.286048</td>
      <td>x0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.296881</td>
      <td>x1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.015805</td>
      <td>x2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.035537</td>
      <td>x3</td>
    </tr>
  </tbody>
</table>
</div>


<a name='2.3.4'></a>

### 4.3.4 Dimensionality Reduction

[back to top](#top)

Dimensionality reduction is an awesome way to do feature engineering. It is very commonly used. Because it is also an unsupervised machine learning technique, we will visit this topic in that section.




<a name='2.4'></a>

## 4.4 Missing Data

[back to top](#top)

We will often have missing data in our datasets. How do we deal with this? Let's start by making some data with missing data. We'll use a numpy nan datatype to do this



```python
from numpy import nan
X = np.array([[ nan, 0,   3  ],
              [ 3,   7,   9  ],
              [ 3,   5,   2  ],
              [ 4,   nan, 6  ],
              [ 8,   8,   1  ]])
y = np.array([14, 16, -1,  8, -5])
```

<a name='2.4.1'></a>

### 4.4.1 Imputation

[back to top](#top)

A very common strategy is to impute or fill in the missing data, based on basic statistical descriptions of the feature column (mode, mean, and median)




```python
from sklearn.impute import SimpleImputer

# strategy = 'mean' will replace nan's with mean value
# of the column
# others are median and most_frequent (mode)
imp = SimpleImputer(strategy='mean')
X2 = imp.fit_transform(X)
X2
```




    array([[4.5, 0. , 3. ],
           [3. , 7. , 9. ],
           [3. , 5. , 2. ],
           [4. , 5. , 6. ],
           [8. , 8. , 1. ]])



<a name='2.4.2'></a>

### 4.4.2 Other Strategies

[back to top](#top)

Depending on the severity of missing data, you will sometimes opt to remove the whole column, or perhaps apply some simple learning to fill in the missing data. This is a great [article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3668100/) on more advanced strategies for handling missing data.



# References

[back to top](#top)
* [Box Cox](https://www.statisticshowto.com/box-cox-transformation/)
* [Multicolinearity](https://www.analyticsvidhya.com/blog/2020/03/what-is-multicollinearity/)
* [Missing Data](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3668100/)
