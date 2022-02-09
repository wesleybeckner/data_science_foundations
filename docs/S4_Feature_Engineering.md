<a href="https://colab.research.google.com/github/wesleybeckner/data_science_foundations/blob/main/notebooks/S4_Feature_Engineering.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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

    /usr/local/lib/python3.7/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm


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





  <div id="df-cfe07248-0454-4b72-a14c-daeb37bf271f">
    <div class="colab-df-container">
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
      <button class="colab-df-convert" onclick="convertToInteractive('df-cfe07248-0454-4b72-a14c-daeb37bf271f')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-cfe07248-0454-4b72-a14c-daeb37bf271f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-cfe07248-0454-4b72-a14c-daeb37bf271f');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
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




    OneHotEncoder(categories='auto', drop=None, dtype=<class 'numpy.float64'>,
                  handle_unknown='error', sparse=True)



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

Using the X and Y sets, use `train_test_split` and `LinearRegression` to make a baseline model based on what we've learned so far.

Assess your model performance visually by plottying `y_test` vs `y_test_pred`


```python
# Cell for Exercise 1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

```




    [<matplotlib.lines.Line2D at 0x7fb3d1bb0890>]




    
![png](S4_Feature_Engineering_files/S4_Feature_Engineering_24_1.png)
    


#### 🙋 Question 1:

> How can we assess the relative feature importance of the features in our model?

We could be tempted to inspect the coefficients (`linear.coef_`) of our  model to evaluate the relative feature importance, but in order to do this our features need to be scaled (so that the relative coefficient sizes are meaningful). What other issues might there be (think categorical vs continuous variables). 

<a name='2.2'></a>

## 4.2 Derived Features

[back to top](#top)

Can we recall an example of where we've seen this previously? That's right earlier on in our first session we derived some polynomial features to create our polynomial model using the linear regression class in sklearn. 

We actually see this a lot in engineering, where we will describe log relationships or some other transformation of the original variable. Actually let me see if I can find an example in my handy BSL...

<img src="https://raw.githubusercontent.com/wesleybeckner/ds_for_engineers/main/assets/C2/bird_stewart_lightfoot.jpg" width=500px></img>

<small>concentration profiles in continous stirred tank vs plug flow reactors. Notice the y-axis is log scale.Thank's Bird, Stewart, Lightfoot!</small>

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
```


```python
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


    
![png](S4_Feature_Engineering_files/S4_Feature_Engineering_35_0.png)
    


<a name='2.2.2'></a>

### 4.2.2 Dealing with Time Series

[back to top](#top)

Often, we will be dealing with time series data, whether its data generated by machinery, reactors, or sales and customers. In the following we discuss some simple practices for dealing with time series data.

<a name='2.2.2.1'></a>

#### **Enrichment**: 2.2.2.1 Fast Fourier Transform

[back to top](#top)

Sometimes we'll want to create a more sophisticated transformation of our input data. As engineers, this can often have to do with some empirical knowledge we understand about our process.

When working with equipment and machinery, we will often want to convert a signal from the time to frequency domain. Let's cover how we can do that with numpy!

<img src="https://www.nti-audio.com/portals/0/pic/news/FFT-Time-Frequency-View-540.png" width=400px></img>

<small>[img src](https://www.nti-audio.com/en/support/know-how/fast-fourier-transform-fft#:~:text=The%20%22Fast%20Fourier%20Transform%22%20(,frequency%20information%20about%20the%20signal.)</small>

What I've drawn here in the following is called a [square-wave signal](https://en.wikipedia.org/wiki/Square_wave)


```python
t = np.linspace(0,5,200)
w = 5
h = 4
s = 4 * h / np.pi * (np.sin(w*t) + np.sin(3*w*t)/3 + np.sin(5*w*t)/5)

# here is the call to numpy FFT
F = np.fft.fft(s)
freq = np.fft.fftfreq(t.shape[-1])

# amplitudes
amps = [max(np.sin(w*t)), max(np.sin(w*t*3)/3), max(np.sin(w*t*5)/5)]

fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].plot(t,s)
ax[0].plot(t,np.sin(w*t), ls='--')
ax[0].plot(t,np.sin(w*t*3)/3, ls='--')
ax[0].plot(t,np.sin(w*t*5)/5, ls='--')
ax[0].set_title('Time Domain')

# tells us about the amplitude of the component at the
# corresponding frequency
magnitude = np.sqrt(F.real**2 + F.imag**2)

ax[1].plot(freq, magnitude)
ax[1].set_xlim(0,.15)
ax[1].set_title('Frequency Domain')
```




    Text(0.5, 1.0, 'Frequency Domain')




    
![png](S4_Feature_Engineering_files/S4_Feature_Engineering_39_1.png)
    


We can construct a similar plot with 4 signals contributing to the square-wave:


```python
t = np.linspace(0,5,200)
w = 5
h = 4
s = 4 * h / np.pi * (np.sin(w*t) + np.sin(3*w*t)/3 + np.sin(5*w*t)/5
                      + np.sin(10*w*t)/10)

F = np.fft.fft(s)
freq = np.fft.fftfreq(t.shape[-1])

fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].plot(t,s)
ax[0].plot(t,np.sin(w*t), ls='--')
ax[0].plot(t,np.sin(w*t*3)/3, ls='--')
ax[0].plot(t,np.sin(w*t*5)/5, ls='--')
ax[0].plot(t,np.sin(10*w*t)/10, ls='--')
ax[0].set_title('Time Domain')

# tells us about the amplitude of the component at the
# corresponding frequency
magnitude = np.sqrt(F.real**2 + F.imag**2)

ax[1].plot(freq, magnitude)
ax[1].set_xlim(0)
ax[1].set_title('Frequency Domain')
```




    Text(0.5, 1.0, 'Frequency Domain')




    
![png](S4_Feature_Engineering_files/S4_Feature_Engineering_41_1.png)
    


<a name='x.2.2.2'></a>

#### 4.2.2.2 Rolling Windows

[back to top](#top)

One powerful technique for dealing with time series data, is to create a rolling window of features based on the historical data. The proper window size can usually be determined by trial and error, or constraints around access to the data itself.

<p align=center>
<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/11/3hotmk.gif"></img>
</p>

In the above gif, we have a window size of 7. What that means is for whatever time step units we are in (that could be minutes, days, months, etc.) we will have 7 of them included in a single instance or observation. This instance or observation is then interpreted by our model and used to assess the target value, typically the quantity in the very next time step after the window (the green bar in the gif).

##### 🏋️ Exercise 2: Optimize Rolling Window Size for Customer Forecasts

For this exercise, you will use the `process_data` function below to help you optimize the `window` size for predicting the order quantity in any given month. 

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
# use data and the function process_data to create your X, y arrays
# then use train_test_split to create train and test portions
data = orders.values[:,6:]

### YOUR CODE HERE ###
# USE y_test and  y_pred for your actual and true test data
# name your labels for the test set labels_test
# change only window parameter in process_data()

```

    1 0.762506752772391
    2 0.8895992010134899
    3 0.9413336982898548
    4 0.7532142077720143
    5 0.9675986952925033
    6 0.9900851583059013
    7 0.9994996102278398
    8 1.0
    9 1.0
    10 1.0
    11 1.0



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


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="d46fe430-b481-4c6b-8c71-3cb9cfb98a5e" class="plotly-graph-div" style="height:500px; width:800px;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("d46fe430-b481-4c6b-8c71-3cb9cfb98a5e")) {
                    Plotly.newPlot(
                        'd46fe430-b481-4c6b-8c71-3cb9cfb98a5e',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "Actual=%{x}<br>Prediction=%{y}", "legendgroup": "", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers", "name": "", "showlegend": false, "type": "scatter", "x": [108.64285714285714, 2355.267295597484, 1432.2712418300653, 19569.69230769231, 1702.1929824561405, 58020.911949685535, 1968.8904494382025, 16569.26100628931, 20.098039215686274, 13504.20634920635, 191.11111111111111, 8608.709677419354, 289.8888888888889, 3665.8097686375318, 440.859375, 712.4603174603176, 16918.132716049382, 8104.40251572327, 38128.80258899677, 36.37096774193548, 21438.1875, 19129.33333333333, 1529.7752808988766, 12274.009146341465, 14.663461538461537, 538.8571428571429, 4057.832278481013, 1630.1966292134832], "xaxis": "x", "y": [108.6428571428318, 2355.26729559746, 1432.2712418300403, 19569.692307692298, 1702.1929824561164, 58020.91194968554, 1968.8904494381782, 16569.261006289296, 20.098039215660815, 13504.206349206332, 191.11111111108582, 8608.709677419334, 289.88888888886396, 3665.809768637509, 440.8593749999747, 712.4603174602926, 16918.132716049364, 8104.402515723249, 38128.802588996754, 36.370967741910036, 21438.187499999993, 19129.33333333332, 1529.7752808988523, 12274.009146341446, 14.663461538436076, 538.8571428571178, 4057.8322784809898, 1630.196629213459], "yaxis": "y"}],
                        {"autosize": false, "height": 500, "legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "R2: 1.000"}, "width": 800, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Actual"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "Prediction"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('d46fe430-b481-4c6b-8c71-3cb9cfb98a5e');
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

                        })
                };

            </script>
        </div>
</body>
</html>


<a name='2.2.3'></a>

### 4.2.3 Image Preprocessing

[back to top](#top)

Image preprocessing is beyond the scope of this session. We will cover this in C4. For now, know that there is a wealth of considerations for how to handle images, and they all fit within the realm of feature engineering.

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




    [<matplotlib.lines.Line2D at 0x7fb3cb441910>]




    
![png](S4_Feature_Engineering_files/S4_Feature_Engineering_51_1.png)
    


We can now generate a random population based on this distribution


```python
r = skewnorm.rvs(a, size=1000)
plt.hist(r)
```




    (array([143., 290., 244., 160.,  96.,  43.,  13.,   7.,   3.,   1.]),
     array([-0.24457186,  0.18369502,  0.61196191,  1.04022879,  1.46849568,
             1.89676256,  2.32502945,  2.75329633,  3.18156322,  3.60983011,
             4.03809699]),
     <a list of 10 Patch objects>)




    
![png](S4_Feature_Engineering_files/S4_Feature_Engineering_53_1.png)
    


Unskewed data will return something close to 0 from calling `df.skew()`. When dealing with actual data, we can use `df.skew()` to determine whether we should transform our data. 


```python
x = pd.DataFrame(r, columns=['Skewed Data'])
x['Skewed Data'].skew()
```




    0.9914234810526167



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

    square root transformed skew: 0.1112
    log transformed skew: -1.9834


    /usr/local/lib/python3.7/dist-packages/pandas/core/series.py:726: RuntimeWarning:
    
    invalid value encountered in sqrt
    
    /usr/local/lib/python3.7/dist-packages/pandas/core/series.py:726: RuntimeWarning:
    
    invalid value encountered in log
    





    <matplotlib.legend.Legend at 0x7fb3cb3525d0>




    
![png](S4_Feature_Engineering_files/S4_Feature_Engineering_57_3.png)
    


We see we didn't get much traction with the log transform, and the log transform will not be able to handle 0 values, and so we will sometimes have to code exceptions for those. 

Boxplot is often a good route to go, but it has the added restriction that the data has to all be above 0.

Let's create a new distribution with this added restriction


```python
a = 6
r = skewnorm.rvs(a, size=1000)
r = [i for i in r if i > 0]
plt.hist(r)
```




    (array([186., 194., 186., 134., 107.,  70.,  41.,  18.,   7.,   6.]),
     array([4.86688792e-04, 2.93460209e-01, 5.86433729e-01, 8.79407250e-01,
            1.17238077e+00, 1.46535429e+00, 1.75832781e+00, 2.05130133e+00,
            2.34427485e+00, 2.63724837e+00, 2.93022189e+00]),
     <a list of 10 Patch objects>)




    
![png](S4_Feature_Engineering_files/S4_Feature_Engineering_59_1.png)
    



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




    <matplotlib.legend.Legend at 0x7fb3cb35e310>




    
![png](S4_Feature_Engineering_files/S4_Feature_Engineering_60_1.png)
    


#### 🏋️ Exercise 3: Transform data from a gamma distribution

Repeat section 2.3.1, this time synthesizing a gamma distribution and transforming it. Which transformation best reduces the skew? Do this for a dataset that does not contain values at or below 0.


```python
# code cell for exercise 3
from scipy.stats import gamma
```




    <matplotlib.legend.Legend at 0x7fb3cb31fe90>




    
![png](S4_Feature_Engineering_files/S4_Feature_Engineering_62_1.png)
    



    
![png](S4_Feature_Engineering_files/S4_Feature_Engineering_62_2.png)
    


<a name='2.3.2'></a>

### 4.3.2 Colinearity

[back to top](#top)

Colinearity can also affect the performance of your machine learning model. In particular, if features are colinear, it can be easy for your model to overfit to your training dataset. This is often mitigated by regularization. If you're curious you can read more about it on [this discussion from StackExchange](https://stats.stackexchange.com/questions/168622/why-is-multicollinearity-not-checked-in-modern-statistics-machine-learning). We will still explore it explicitly here by calculating the [Variance Inflation Factor](https://www.statsmodels.org/stable/_modules/statsmodels/stats/outliers_influence.html#variance_inflation_factor) (VIF) on some hypothetical data.

$$ VIF = \frac{1}{1-R^2}$$





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
x1 = [func(i, .05) for i in x0]
x2 = [func(i, 1) for i in x0]
x3 = [random.randint(0,100) for i in x0]

# take a look
fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.plot(x0, x1, label='x1')
ax.plot(x0, x2, label='x2')
ax.plot(x0, x3, label='x3')
ax.legend()
```




    <matplotlib.legend.Legend at 0x7fb3cbab8c90>




    
![png](S4_Feature_Engineering_files/S4_Feature_Engineering_67_1.png)
    


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




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb3cb102750>




    
![png](S4_Feature_Engineering_files/S4_Feature_Engineering_78_1.png)
    



```python
pd.DataFrame(normed, columns = [['x0','x1','x2','x3']]).plot(kind='kde')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb3cb102990>




    
![png](S4_Feature_Engineering_files/S4_Feature_Engineering_79_1.png)
    


#### 🏋️ Exercise 4: Normalization affect on VIF 

In the above, we saw how to scale and center variables. How does this affect VIF? 

* Calculate the VIF for the raw and scaled-centered data


```python
# Code Cell for Exercise 4
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
