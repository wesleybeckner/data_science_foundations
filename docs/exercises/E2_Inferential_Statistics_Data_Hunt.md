<a href="https://colab.research.google.com/github/wesleybeckner/data_science_foundations/blob/main/notebooks/exercises/E2_Inferential_Statistics_Data_Hunt.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Data Science Foundations <br> Lab 2: Data Hunt II

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com

---

<br>

<p align=center>
<img src="https://raw.githubusercontent.com/wesleybeckner/technology_fundamentals/main/assets/datahunt2.png" width=1000px></img>
</p>

<p align=center>
That's right you heard correctly. It's the data hunt part TWO.
</p>





<a name='x.0'></a>

## Preparing Environment and Importing Data

### Import Packages


```python
!pip install -U plotly
```

    Requirement already satisfied: plotly in /usr/local/lib/python3.7/dist-packages (4.4.1)
    Collecting plotly
      Downloading plotly-5.1.0-py2.py3-none-any.whl (20.6 MB)
    [K     |████████████████████████████████| 20.6 MB 1.3 MB/s 
    [?25hRequirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from plotly) (1.15.0)
    Collecting tenacity>=6.2.0
      Downloading tenacity-8.0.1-py3-none-any.whl (24 kB)
    Installing collected packages: tenacity, plotly
      Attempting uninstall: plotly
        Found existing installation: plotly 4.4.1
        Uninstalling plotly-4.4.1:
          Successfully uninstalled plotly-4.4.1
    Successfully installed plotly-5.1.0 tenacity-8.0.1



```python
# our standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from ipywidgets import interact

# our stats libraries
import random
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy

# our scikit-Learn library for the regression models
import sklearn         
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

### Import and Clean Data


```python
df = pd.read_csv("https://raw.githubusercontent.com/wesleybeckner/"\
                 "technology_fundamentals/main/assets/truffle_rates.csv")
df = df.loc[df['rate'] > 0]
```


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
      <th>base_cake</th>
      <th>truffle_type</th>
      <th>primary_flavor</th>
      <th>secondary_flavor</th>
      <th>color_group</th>
      <th>rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chiffon</td>
      <td>Candy Outer</td>
      <td>Cherry Cream Spice</td>
      <td>Ginger Beer</td>
      <td>Tiffany</td>
      <td>0.167097</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chiffon</td>
      <td>Candy Outer</td>
      <td>Cherry Cream Spice</td>
      <td>Ginger Beer</td>
      <td>Tiffany</td>
      <td>0.153827</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chiffon</td>
      <td>Candy Outer</td>
      <td>Cherry Cream Spice</td>
      <td>Ginger Beer</td>
      <td>Tiffany</td>
      <td>0.100299</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Chiffon</td>
      <td>Candy Outer</td>
      <td>Cherry Cream Spice</td>
      <td>Ginger Beer</td>
      <td>Tiffany</td>
      <td>0.333008</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Chiffon</td>
      <td>Candy Outer</td>
      <td>Cherry Cream Spice</td>
      <td>Ginger Beer</td>
      <td>Tiffany</td>
      <td>0.078108</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (9210, 6)



## Exploratory Data Analysis

### Q1 Finding Influential Features

Which of the five features (base_cake, truffle_type, primary_flavor, secondary_flavor, color_group) of the truffles is most influential on production rate?

Back your answer with both a visualization of the distributions (boxplot, kernel denisty estimate, histogram, violin plot) and a statistical test (moods median, ANOVA, t-test)

* Be sure: 
    * everything is labeled (can you improve your labels with additional descriptive statistical information e.g. indicate mean, std, etc.)
    * you meet the assumptions of your statistical test

#### Q1.1 Visualization

Use any number of visualizations. Here is an example to get you started:


```python
# Example: a KDE of the truffle_type and base_cake columns

fig, ax = plt.subplots(2, 1, figsize=(12,12))
sns.kdeplot(x=df['rate'], hue=df['truffle_type'], fill=True, ax=ax[0])
sns.kdeplot(x=df['rate'], hue=df['base_cake'], fill=True, ax=ax[1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f549eea03d0>




    
![png](E2_Inferential_Statistics_Data_Hunt_files/E2_Inferential_Statistics_Data_Hunt_13_1.png)
    



```python

```

    /usr/local/lib/python3.7/dist-packages/numpy/core/_asarray.py:83: VisibleDeprecationWarning:
    
    Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
    
    /usr/local/lib/python3.7/dist-packages/numpy/core/_asarray.py:83: VisibleDeprecationWarning:
    
    Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
    
    /usr/local/lib/python3.7/dist-packages/numpy/core/_asarray.py:83: VisibleDeprecationWarning:
    
    Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
    
    /usr/local/lib/python3.7/dist-packages/numpy/core/_asarray.py:83: VisibleDeprecationWarning:
    
    Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
    
    /usr/local/lib/python3.7/dist-packages/numpy/core/_asarray.py:83: VisibleDeprecationWarning:
    
    Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
    
    /usr/local/lib/python3.7/dist-packages/matplotlib/backends/backend_agg.py:214: RuntimeWarning:
    
    Glyph 9 missing from current font.
    
    /usr/local/lib/python3.7/dist-packages/matplotlib/backends/backend_agg.py:183: RuntimeWarning:
    
    Glyph 9 missing from current font.
    



    
![png](E2_Inferential_Statistics_Data_Hunt_files/E2_Inferential_Statistics_Data_Hunt_14_1.png)
    


#### Q1.2 Statistical Analysis

What statistical tests can you perform to evaluate your hypothesis from the visualizations (maybe you think one particular feature is significant). Here's an ANOVA on the `truffle_type` column to get you started:




```python
model = ols('rate ~ C({})'.format('truffle_type'), data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
display(anova_table)
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
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(truffle_type)</th>
      <td>36.383370</td>
      <td>2.0</td>
      <td>302.005</td>
      <td>9.199611e-128</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>554.596254</td>
      <td>9207.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


> Is this P value significant? What is the null hypothesis? How do we check the assumptions of ANOVA? 


```python

```

    base_cake



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
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(base_cake)</th>
      <td>331.373550</td>
      <td>5.0</td>
      <td>2349.684756</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>259.606073</td>
      <td>9204.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Shapiro:  0.9281061887741089 0.0
    Bartlett:  619.3727153356931 1.3175663824168166e-131
    
    truffle_type


    /usr/local/lib/python3.7/dist-packages/scipy/stats/morestats.py:1676: UserWarning:
    
    p-value may not be accurate for N > 5000.
    



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
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(truffle_type)</th>
      <td>36.383370</td>
      <td>2.0</td>
      <td>302.005</td>
      <td>9.199611e-128</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>554.596254</td>
      <td>9207.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Shapiro:  0.9645588994026184 1.3704698981096711e-42


    /usr/local/lib/python3.7/dist-packages/scipy/stats/morestats.py:1676: UserWarning:
    
    p-value may not be accurate for N > 5000.
    


    Bartlett:  533.0206680979852 1.8031528902362296e-116
    
    primary_flavor



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
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(primary_flavor)</th>
      <td>159.105452</td>
      <td>47.0</td>
      <td>71.815842</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>431.874171</td>
      <td>9162.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    /usr/local/lib/python3.7/dist-packages/scipy/stats/morestats.py:1676: UserWarning:
    
    p-value may not be accurate for N > 5000.
    


    Shapiro:  0.9738250970840454 6.485387538059916e-38
    Bartlett:  1609.0029005171464 1.848613457353585e-306
    
    secondary_flavor



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
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(secondary_flavor)</th>
      <td>115.773877</td>
      <td>28.0</td>
      <td>79.884192</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>475.205747</td>
      <td>9181.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Shapiro:  0.9717048406600952 4.3392384038527993e-39
    Bartlett:  1224.4882890761903 3.5546073028894766e-240
    
    color_group


    /usr/local/lib/python3.7/dist-packages/scipy/stats/morestats.py:1676: UserWarning:
    
    p-value may not be accurate for N > 5000.
    



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
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(color_group)</th>
      <td>33.878491</td>
      <td>11.0</td>
      <td>50.849974</td>
      <td>1.873235e-109</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>557.101132</td>
      <td>9198.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Shapiro:  0.9598756432533264 1.401298464324817e-44
    Bartlett:  298.6432027161358 1.6917844519244488e-57
    


    /usr/local/lib/python3.7/dist-packages/scipy/stats/morestats.py:1676: UserWarning:
    
    p-value may not be accurate for N > 5000.
    


### Q2 Finding Best and Worst Groups



#### Q2.1 Compare Every Group to the Whole

Of the primary flavors (feature), what 5 flavors (groups) would you recommend Truffletopia discontinue?

Iterate through every level (i.e. pound, cheese, sponge cakes) of every category (i.e. base cake, primary flavor, secondary flavor) and use moods median testing to compare the group distribution to the grand median rate.


```python

```

    (98, 10)


After you've computed a moods median test on every group, filter any data above a significance level of 0.05


```python

```

    (76, 10)


Return the groups with the lowest median performance (your table need not look exactly like the one I've created)


```python

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
      <th>descriptor</th>
      <th>group</th>
      <th>pearsons_chi_square</th>
      <th>p_value</th>
      <th>grand_median</th>
      <th>group_mean</th>
      <th>group_median</th>
      <th>size</th>
      <th>welch p</th>
      <th>table</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>primary_flavor</td>
      <td>Coconut</td>
      <td>56.8675</td>
      <td>4.66198e-14</td>
      <td>0.310345</td>
      <td>0.139998</td>
      <td>0.0856284</td>
      <td>100</td>
      <td>2.64572e-29</td>
      <td>[[12, 4593], [88, 4517]]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>secondary_flavor</td>
      <td>Wild Cherry Cream</td>
      <td>56.8675</td>
      <td>4.66198e-14</td>
      <td>0.310345</td>
      <td>0.139998</td>
      <td>0.0856284</td>
      <td>100</td>
      <td>2.64572e-29</td>
      <td>[[12, 4593], [88, 4517]]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>primary_flavor</td>
      <td>Pink Lemonade</td>
      <td>61.5563</td>
      <td>4.30253e-15</td>
      <td>0.310345</td>
      <td>0.129178</td>
      <td>0.0928782</td>
      <td>85</td>
      <td>2.05798e-28</td>
      <td>[[6, 4599], [79, 4526]]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>primary_flavor</td>
      <td>Chocolate</td>
      <td>51.3203</td>
      <td>7.84617e-13</td>
      <td>0.310345</td>
      <td>0.145727</td>
      <td>0.0957584</td>
      <td>91</td>
      <td>1.11719e-28</td>
      <td>[[11, 4594], [80, 4525]]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>primary_flavor</td>
      <td>Wild Cherry Cream</td>
      <td>43.5452</td>
      <td>4.14269e-11</td>
      <td>0.310345</td>
      <td>0.148964</td>
      <td>0.10588</td>
      <td>70</td>
      <td>2.59384e-20</td>
      <td>[[7, 4598], [63, 4542]]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>secondary_flavor</td>
      <td>Mixed Berry</td>
      <td>164.099</td>
      <td>1.43951e-37</td>
      <td>0.310345</td>
      <td>0.153713</td>
      <td>0.115202</td>
      <td>261</td>
      <td>6.73636e-75</td>
      <td>[[28, 4577], [233, 4372]]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>secondary_flavor</td>
      <td>Peppermint</td>
      <td>66.0235</td>
      <td>4.45582e-16</td>
      <td>0.310345</td>
      <td>0.129107</td>
      <td>0.12201</td>
      <td>86</td>
      <td>7.6449e-37</td>
      <td>[[5, 4600], [81, 4524]]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>base_cake</td>
      <td>Butter</td>
      <td>696.649</td>
      <td>1.60093e-153</td>
      <td>0.310345</td>
      <td>0.15951</td>
      <td>0.136231</td>
      <td>905</td>
      <td>0</td>
      <td>[[75, 4530], [830, 3775]]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>secondary_flavor</td>
      <td>Rum</td>
      <td>69.5192</td>
      <td>7.56747e-17</td>
      <td>0.310345</td>
      <td>0.157568</td>
      <td>0.139834</td>
      <td>93</td>
      <td>4.42643e-42</td>
      <td>[[6, 4599], [87, 4518]]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>secondary_flavor</td>
      <td>Cucumber</td>
      <td>175.061</td>
      <td>5.80604e-40</td>
      <td>0.310345</td>
      <td>0.170015</td>
      <td>0.14097</td>
      <td>288</td>
      <td>4.33234e-79</td>
      <td>[[33, 4572], [255, 4350]]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>primary_flavor</td>
      <td>Gingersnap</td>
      <td>131.114</td>
      <td>2.33844e-30</td>
      <td>0.310345</td>
      <td>0.159268</td>
      <td>0.143347</td>
      <td>192</td>
      <td>1.01371e-69</td>
      <td>[[17, 4588], [175, 4430]]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>primary_flavor</td>
      <td>Cherry Cream Spice</td>
      <td>66.3302</td>
      <td>3.81371e-16</td>
      <td>0.310345</td>
      <td>0.175751</td>
      <td>0.146272</td>
      <td>100</td>
      <td>1.03408e-36</td>
      <td>[[9, 4596], [91, 4514]]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>primary_flavor</td>
      <td>Orange Brandy</td>
      <td>97.0624</td>
      <td>6.71776e-23</td>
      <td>0.310345</td>
      <td>0.185908</td>
      <td>0.157804</td>
      <td>186</td>
      <td>1.86398e-49</td>
      <td>[[26, 4579], [160, 4445]]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>primary_flavor</td>
      <td>Irish Cream</td>
      <td>87.5008</td>
      <td>8.42448e-21</td>
      <td>0.310345</td>
      <td>0.184505</td>
      <td>0.176935</td>
      <td>151</td>
      <td>6.30631e-52</td>
      <td>[[18, 4587], [133, 4472]]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>base_cake</td>
      <td>Chiffon</td>
      <td>908.383</td>
      <td>1.47733e-199</td>
      <td>0.310345</td>
      <td>0.208286</td>
      <td>0.177773</td>
      <td>1821</td>
      <td>0</td>
      <td>[[334, 4271], [1487, 3118]]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>primary_flavor</td>
      <td>Ginger Lime</td>
      <td>40.1257</td>
      <td>2.38138e-10</td>
      <td>0.310345</td>
      <td>0.225157</td>
      <td>0.181094</td>
      <td>100</td>
      <td>2.63308e-18</td>
      <td>[[18, 4587], [82, 4523]]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>primary_flavor</td>
      <td>Doughnut</td>
      <td>98.4088</td>
      <td>3.40338e-23</td>
      <td>0.310345</td>
      <td>0.234113</td>
      <td>0.189888</td>
      <td>300</td>
      <td>1.03666e-40</td>
      <td>[[65, 4540], [235, 4370]]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>primary_flavor</td>
      <td>Butter Milk</td>
      <td>28.3983</td>
      <td>9.87498e-08</td>
      <td>0.310345</td>
      <td>0.237502</td>
      <td>0.190708</td>
      <td>100</td>
      <td>1.96333e-20</td>
      <td>[[23, 4582], [77, 4528]]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>primary_flavor</td>
      <td>Pecan</td>
      <td>40.8441</td>
      <td>1.64868e-10</td>
      <td>0.310345</td>
      <td>0.197561</td>
      <td>0.192372</td>
      <td>89</td>
      <td>2.86564e-25</td>
      <td>[[14, 4591], [75, 4530]]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>secondary_flavor</td>
      <td>Dill Pickle</td>
      <td>69.8101</td>
      <td>6.52964e-17</td>
      <td>0.310345</td>
      <td>0.228289</td>
      <td>0.19916</td>
      <td>241</td>
      <td>6.39241e-33</td>
      <td>[[56, 4549], [185, 4420]]</td>
    </tr>
  </tbody>
</table>
</div>



We would want to cut the following primary flavors. Check to see that you get a similar answer. rip wild cherry cream.

```
['Coconut', 'Pink Lemonade', 'Chocolate', 'Wild Cherry Cream', 'Gingersnap']
```


```python

```




    ['Coconut', 'Pink Lemonade', 'Chocolate', 'Wild Cherry Cream', 'Gingersnap']



#### Q2.2 Beyond Statistical Testing: Using Reasoning

Let's look at the total profile of the products associated with the five worst primary flavors. Given the number of different products made with any of these flavors, would you alter your answer at all?


```python
# 1. filter df for only bottom five flavors
# 2. groupby all columns besides rate
# 3. describe the rate column.

# by doing this we can evaluate just how much sampling variety we have for the
# worst performing flavors.

bottom_five = ['Coconut', 'Pink Lemonade', 'Chocolate', 'Wild Cherry Cream', 'Gingersnap']
df.loc[df['primary_flavor'].isin(bottom_five)].groupby(list(df.columns[:-1]))['rate'].describe()
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
      <th></th>
      <th></th>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>base_cake</th>
      <th>truffle_type</th>
      <th>primary_flavor</th>
      <th>secondary_flavor</th>
      <th>color_group</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Butter</th>
      <th>Jelly Filled</th>
      <th>Pink Lemonade</th>
      <th>Butter Rum</th>
      <th>Rose</th>
      <td>85.0</td>
      <td>0.129178</td>
      <td>0.137326</td>
      <td>0.000061</td>
      <td>0.032887</td>
      <td>0.092878</td>
      <td>0.171350</td>
      <td>0.860045</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Chiffon</th>
      <th>Candy Outer</th>
      <th>Wild Cherry Cream</th>
      <th>Rock and Rye</th>
      <th>Olive</th>
      <td>17.0</td>
      <td>0.094287</td>
      <td>0.059273</td>
      <td>0.010464</td>
      <td>0.053976</td>
      <td>0.077098</td>
      <td>0.120494</td>
      <td>0.229933</td>
    </tr>
    <tr>
      <th>Chocolate Outer</th>
      <th>Gingersnap</th>
      <th>Dill Pickle</th>
      <th>Burgundy</th>
      <td>59.0</td>
      <td>0.133272</td>
      <td>0.080414</td>
      <td>0.021099</td>
      <td>0.069133</td>
      <td>0.137972</td>
      <td>0.172066</td>
      <td>0.401387</td>
    </tr>
    <tr>
      <th>Jelly Filled</th>
      <th>Chocolate</th>
      <th>Tutti Frutti</th>
      <th>Burgundy</th>
      <td>91.0</td>
      <td>0.145727</td>
      <td>0.135230</td>
      <td>0.000033</td>
      <td>0.044847</td>
      <td>0.095758</td>
      <td>0.185891</td>
      <td>0.586570</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Pound</th>
      <th>Candy Outer</th>
      <th>Coconut</th>
      <th>Wild Cherry Cream</th>
      <th>Taupe</th>
      <td>100.0</td>
      <td>0.139998</td>
      <td>0.147723</td>
      <td>0.000705</td>
      <td>0.036004</td>
      <td>0.085628</td>
      <td>0.187318</td>
      <td>0.775210</td>
    </tr>
    <tr>
      <th>Chocolate Outer</th>
      <th>Gingersnap</th>
      <th>Rock and Rye</th>
      <th>Black</th>
      <td>67.0</td>
      <td>0.156160</td>
      <td>0.110666</td>
      <td>0.002846</td>
      <td>0.074615</td>
      <td>0.139572</td>
      <td>0.241114</td>
      <td>0.551898</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Jelly Filled</th>
      <th>Gingersnap</th>
      <th>Kiwi</th>
      <th>Taupe</th>
      <td>66.0</td>
      <td>0.185662</td>
      <td>0.132272</td>
      <td>0.000014</td>
      <td>0.086377</td>
      <td>0.166340</td>
      <td>0.247397</td>
      <td>0.593016</td>
    </tr>
    <tr>
      <th>Wild Cherry Cream</th>
      <th>Mango</th>
      <th>Taupe</th>
      <td>53.0</td>
      <td>0.166502</td>
      <td>0.160090</td>
      <td>0.001412</td>
      <td>0.056970</td>
      <td>0.108918</td>
      <td>0.207306</td>
      <td>0.787224</td>
    </tr>
  </tbody>
</table>
</div>



#### Q2.3 The Jelly Filled Conundrum

Your boss notices the Jelly filled truffles are being produced much faster than the candy outer truffles and suggests expanding into this product line. What is your response? Use the visualization tool below to help you think about this problem, then create any visualizations or analyses of your own.

[sunburst charts](https://plotly.com/python/sunburst-charts/)


```python
def sun(path=[['base_cake', 'truffle_type', 'primary_flavor', 'secondary_flavor', 'color_group'],
              ['truffle_type', 'base_cake', 'primary_flavor', 'secondary_flavor', 'color_group']]):
  fig = px.sunburst(df, path=path, 
                    color='rate', 
                    color_continuous_scale='viridis',
                    )

  fig.update_layout(
      margin=dict(l=20, r=20, t=20, b=20),
      height=650
  )
  fig.show()
```


```python
interact(sun)
```


    interactive(children=(Dropdown(description='path', options=(['base_cake', 'truffle_type', 'primary_flavor', 's…





    <function __main__.sun>


