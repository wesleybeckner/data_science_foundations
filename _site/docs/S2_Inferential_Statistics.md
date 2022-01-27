<a href="https://colab.research.google.com/github/wesleybeckner/data_science_foundations/blob/main/notebooks/S2_Inferential_Statistics.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Data Science Foundations, Session 2: Inferential Statistics

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com

---

<br>

In this session we will look at the utility of EDA combined with inferential statistics.

<br>

---


<a name='x.0'></a>

## 2.0 Preparing Environment and Importing Data

[back to top](#top)

<a name='x.0.1'></a>

### 2.0.1 Import Packages

[back to top](#top)


```python
# The modules we've seen before
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# our stats modules
import random
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy
```

<a name='x.0.2'></a>

### 2.0.2 Load Dataset

[back to top](#top)

For this session, we will use dummy datasets from sklearn.


```python
df = pd.read_csv('https://raw.githubusercontent.com/wesleybeckner/'\
                 'ds_for_engineers/main/data/truffle_margin/truffle_margin_customer.csv')
```


```python
df
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
    </tr>
    <tr>
      <th>1663</th>
      <td>Tiramisu</td>
      <td>Chocolate Outer</td>
      <td>Doughnut</td>
      <td>Pear</td>
      <td>Amethyst</td>
      <td>Fickelgruber</td>
      <td>12/2020</td>
      <td>38128.802589</td>
      <td>0.420111</td>
    </tr>
    <tr>
      <th>1664</th>
      <td>Tiramisu</td>
      <td>Chocolate Outer</td>
      <td>Doughnut</td>
      <td>Pear</td>
      <td>Burgundy</td>
      <td>Zebrabar</td>
      <td>12/2020</td>
      <td>108.642857</td>
      <td>0.248659</td>
    </tr>
    <tr>
      <th>1665</th>
      <td>Tiramisu</td>
      <td>Chocolate Outer</td>
      <td>Doughnut</td>
      <td>Pear</td>
      <td>Teal</td>
      <td>Zebrabar</td>
      <td>12/2020</td>
      <td>3517.933333</td>
      <td>0.378501</td>
    </tr>
    <tr>
      <th>1666</th>
      <td>Tiramisu</td>
      <td>Chocolate Outer</td>
      <td>Doughnut</td>
      <td>Rock and Rye</td>
      <td>Amethyst</td>
      <td>Slugworth</td>
      <td>12/2020</td>
      <td>10146.898432</td>
      <td>0.213149</td>
    </tr>
    <tr>
      <th>1667</th>
      <td>Tiramisu</td>
      <td>Chocolate Outer</td>
      <td>Doughnut</td>
      <td>Rock and Rye</td>
      <td>Burgundy</td>
      <td>Zebrabar</td>
      <td>12/2020</td>
      <td>1271.904762</td>
      <td>0.431813</td>
    </tr>
  </tbody>
</table>
<p>1668 rows × 9 columns</p>
</div>




```python
descriptors = df.columns[:-2]
```


```python
for col in df.columns[:-5]:
  print(col)
  print(df[col].unique())
  print()
```

    Base Cake
    ['Butter' 'Cheese' 'Chiffon' 'Pound' 'Sponge' 'Tiramisu']
    
    Truffle Type
    ['Candy Outer' 'Chocolate Outer' 'Jelly Filled']
    
    Primary Flavor
    ['Butter Pecan' 'Ginger Lime' 'Margarita' 'Pear' 'Pink Lemonade'
     'Raspberry Ginger Ale' 'Sassafras' 'Spice' 'Wild Cherry Cream'
     'Cream Soda' 'Horchata' 'Kettle Corn' 'Lemon Bar' 'Orange Pineapple\tP'
     'Plum' 'Orange' 'Butter Toffee' 'Lemon' 'Acai Berry' 'Apricot'
     'Birch Beer' 'Cherry Cream Spice' 'Creme de Menthe' 'Fruit Punch'
     'Ginger Ale' 'Grand Mariner' 'Orange Brandy' 'Pecan' 'Toasted Coconut'
     'Watermelon' 'Wintergreen' 'Vanilla' 'Bavarian Cream' 'Black Licorice'
     'Caramel Cream' 'Cheesecake' 'Cherry Cola' 'Coffee' 'Irish Cream'
     'Lemon Custard' 'Mango' 'Sour' 'Amaretto' 'Blueberry' 'Butter Milk'
     'Chocolate Mint' 'Coconut' 'Dill Pickle' 'Gingersnap' 'Chocolate'
     'Doughnut']
    
    Secondary Flavor
    ['Toffee' 'Banana' 'Rum' 'Tutti Frutti' 'Vanilla' 'Mixed Berry'
     'Whipped Cream' 'Apricot' 'Passion Fruit' 'Peppermint' 'Dill Pickle'
     'Black Cherry' 'Wild Cherry Cream' 'Papaya' 'Mango' 'Cucumber' 'Egg Nog'
     'Pear' 'Rock and Rye' 'Tangerine' 'Apple' 'Black Currant' 'Kiwi' 'Lemon'
     'Hazelnut' 'Butter Rum' 'Fuzzy Navel' 'Mojito' 'Ginger Beer']
    


## 2.1 Many Flavors of Statistical Tests

<p align="center">
<img src="https://luminousmen.com/media/descriptive-and-inferential-statistics.jpeg" width=400px></img>
<br>
<small> https://luminousmen.com/post/descriptive-and-inferential-statistics </small>
</p>

>Descriptive statistics describes data (for example, a chart or graph) and inferential statistics allows you to make predictions (“inferences”) from that data. With inferential statistics, you take data from samples and make generalizations about a population - [statshowto](https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/inferential-statistics/#:~:text=Descriptive%20statistics%20describes%20data%20(for,make%20generalizations%20about%20a%20population.)

* **Moods Median Test**
* [Kruskal-Wallis Test](https://sixsigmastudyguide.com/kruskal-wallis-non-parametric-hypothesis-test/) (Another comparison of Medians test)
* T-Test
* Analysis of Variance (ANOVA)
  * One Way ANOVA
  * Two Way ANOVA
  * MANOVA
  * Factorial ANOVA

When do I use each of these? We will talk about this as we proceed through the examples. [This page](https://support.minitab.com/en-us/minitab/20/help-and-how-to/statistics/nonparametrics/supporting-topics/which-test-should-i-use/) from minitab has good rules of thumb on the subject.



### 2.1.1 What is Mood's Median?

> You can use Chi-Square to test for a goodness of fit (whether a sample of data represents a distribution) or whether two variables are related (using a contingency table, which we will create below!)

**A special case of Pearon's Chi-Squared Test:** We create a table that counts the observations above and below the global median for two different groups. We then perform a *chi-squared test of significance* on this *contingency table* 

Null hypothesis: the Medians are all equal

The chi-square test statistic:

$$x^2 = \sum{\frac{(O-E)^2}{E}}$$

Where \\(O\\) is the observed frequency and \\(E\\) is the expected frequency.

**Let's take an example**, say we have three shifts with the following production rates:


```python
np.random.seed(42)
shift_one = [round(i) for i in np.random.normal(16, 3, 10)]
shift_two = [round(i) for i in np.random.normal(21, 3, 10)]
```


```python
print(shift_one)
print(shift_two)
```

    [17, 16, 18, 21, 15, 15, 21, 18, 15, 18]
    [20, 20, 22, 15, 16, 19, 18, 22, 18, 17]



```python
stat, p, m, table = scipy.stats.median_test(shift_one, shift_two, correction=False)
```

what is `median_test` returning?


```python
print("The perasons chi-square test statistic: {:.2f}".format(stat))
print("p-value of the test: {:.3f}".format(p))
print("the grand median: {}".format(m))
```

    The perasons chi-square test statistic: 1.98
    p-value of the test: 0.160
    the grand median: 18.0


Let's evaluate that test statistic ourselves by taking a look at the contingency table:


```python
table
```




    array([[2, 5],
           [8, 5]])



This is easier to make sense of if we order the shift times


```python
shift_one.sort()
shift_one
```




    [15, 15, 15, 16, 17, 18, 18, 18, 21, 21]



When we look at shift one, we see that 8 values are at or below the grand median.


```python
shift_two.sort()
shift_two
```




    [15, 16, 17, 18, 18, 19, 20, 20, 22, 22]



For shift two, only two are at or below the grand median.

Since the sample sizes are the same, the expected value for both groups is the same, 5 above and 5 below the grand median. The chi-square is then:

$$X^2 = \frac{(2-5)^2}{5} + \frac{(8-5)^2}{5} + \frac{(8-5)^2}{5} + \frac{(2-5)^2}{5}$$



```python
(3-5)**2/5 + (7-5)**2/5 + (7-5)**2/5 + (3-5)**2/5
```




    3.2



Our p-value, or the probability of observing the null-hypothsis, is under 0.05. We can conclude that these shift performances were drawn under seperate distributions.

For comparison, let's do this analysis again with shifts of equal performances


```python
np.random.seed(3)
shift_three = [round(i) for i in np.random.normal(16, 3, 10)]
shift_four = [round(i) for i in np.random.normal(16, 3, 10)]
stat, p, m, table = scipy.stats.median_test(shift_three, shift_four,
                                            correction=False)
print("The pearsons chi-square test statistic: {:.2f}".format(stat))
print("p-value of the test: {:.3f}".format(p))
print("the grand median: {}".format(m))
```

    The pearsons chi-square test statistic: 0.00
    p-value of the test: 1.000
    the grand median: 15.5


and the shift raw values:


```python
shift_three.sort()
shift_four.sort()
print(shift_three)
print(shift_four)
```

    [10, 14, 15, 15, 15, 16, 16, 16, 17, 21]
    [11, 12, 13, 14, 15, 16, 19, 19, 19, 21]



```python
table
```




    array([[5, 5],
           [5, 5]])



### 2.1.2 When to Use Mood's?

**Mood's Median Test is highly flexible** but has the following assumptions:

* Considers only one categorical factor
* Response variable is continuous (our shift rates)
* Data does not need to be normally distributed
  * But the distributions are similarly shaped
* Sample sizes can be unequal and small (less than 20 observations)

Other considerations:

* Not as powerful as Kruskal-Wallis Test but still useful for small sample sizes or when there are outliers

#### 🏋️ Exercise 1: Use Mood's Median Test


##### **Part A** Perform moods median test on Base Cake in Truffle data

We're also going to get some practice with pandas groupby.


```python
# what is returned by this groupby?
gp = df.groupby('Base Cake')
```

How do we find out? We could iterate through it:


```python
# seems to be a tuple of some sort
for i in gp:
  print(i)
  break
```

    ('Butter',      Base Cake     Truffle Type  ...             KG EBITDA/KG
    0       Butter      Candy Outer  ...   53770.342593  0.500424
    1       Butter      Candy Outer  ...  466477.578125  0.220395
    2       Butter      Candy Outer  ...   80801.728070  0.171014
    3       Butter      Candy Outer  ...   18046.111111  0.233025
    4       Butter      Candy Outer  ...   19147.454268  0.480689
    ...        ...              ...  ...            ...       ...
    1562    Butter  Chocolate Outer  ...    9772.200521  0.158279
    1563    Butter  Chocolate Outer  ...   10861.245675 -0.159275
    1564    Butter  Chocolate Outer  ...    3578.592163  0.431328
    1565    Butter     Jelly Filled  ...   21438.187500  0.105097
    1566    Butter     Jelly Filled  ...   15617.489115  0.185070
    
    [456 rows x 9 columns])



```python
# the first object appears to be the group
print(i[0])

# the second object appears to be the df belonging to that group
print(i[1])
```

    Butter
         Base Cake     Truffle Type  ...             KG EBITDA/KG
    0       Butter      Candy Outer  ...   53770.342593  0.500424
    1       Butter      Candy Outer  ...  466477.578125  0.220395
    2       Butter      Candy Outer  ...   80801.728070  0.171014
    3       Butter      Candy Outer  ...   18046.111111  0.233025
    4       Butter      Candy Outer  ...   19147.454268  0.480689
    ...        ...              ...  ...            ...       ...
    1562    Butter  Chocolate Outer  ...    9772.200521  0.158279
    1563    Butter  Chocolate Outer  ...   10861.245675 -0.159275
    1564    Butter  Chocolate Outer  ...    3578.592163  0.431328
    1565    Butter     Jelly Filled  ...   21438.187500  0.105097
    1566    Butter     Jelly Filled  ...   15617.489115  0.185070
    
    [456 rows x 9 columns]


going back to our diagram from our earlier pandas session. It looks like whenever we split in the groupby method, we create separate dataframes as well as their group label:

<img src="https://swcarpentry.github.io/r-novice-gapminder/fig/12-plyr-fig1.png" width=500></img>

Ok, so we know `gp` is separate dataframes. How do we turn them into arrays to then pass to `median_test`?


```python
# complete this for loop
for i, j in gp:
  # turn j into an array using the .values attribute
  # print this to the screen
```

After you've completed the previous step, turn this into a list comprehension and pass the result to a variable called `margins`


```python
# complete the code below
# margins = [# YOUR LIST COMPREHENSION HERE]
```

Remember the list unpacking we did for the tic tac toe project? We're going to do the same thing here. Unpack the margins list for `median_test` and run the cell below!


```python
# complete the following line
# stat, p, m, table = scipy.stats.median_test(<UNPACK MARGINS HERE>, correction=False)

print("The pearsons chi-square test statistic: {:.2f}".format(stat))
print("p-value of the test: {:.2e}".format(p))
print("the grand median: {:.2e}".format(m))
```

    The pearsons chi-square test statistic: 448.81
    p-value of the test: 8.85e-95
    the grand median: 2.16e-01


##### **Part B** View the distributions of the data using matplotlib and seaborn

What a fantastic statistical result we found! Can we affirm our result with some visualizations? I hope so! Create a boxplot below using pandas. In your call to `df.boxplot()` the `by` parameter should be set to `Base Cake` and the `column` parameter should be set to `EBITDA/KG`


```python
# YOUR BOXPLOT HERE
```

    /usr/local/lib/python3.7/dist-packages/numpy/core/_asarray.py:83: VisibleDeprecationWarning:
    
    Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
    





    <matplotlib.axes._subplots.AxesSubplot at 0x7fb512f782d0>




    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_44_2.png)
    


For comparison, I've shown the boxplot below using seaborn!


```python
fig, ax = plt.subplots(figsize=(10,7))
ax = sns.boxplot(x='Base Cake', y='EBITDA/KG', data=df, color='#A0cbe8')
```


    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_46_0.png)
    


##### **Part C** Perform Moods Median on all the other groups


```python
# Recall the other descriptors we have
descriptors
```




    Index(['Base Cake', 'Truffle Type', 'Primary Flavor', 'Secondary Flavor',
           'Color Group', 'Customer', 'Date'],
          dtype='object')




```python
for desc in descriptors:

  # YOUR CODE FORM MARGINS BELOW
  # margins = [<YOUR LIST COMPREHENSION>]

  # UNPACK MARGINS INTO MEDIAN_TEST
  # stat, p, m, table = scipy.stats.median_test(<YOUR UNPACKING METHOD>, correction=False)
  print(desc)
  print("The pearsons chi-square test statistic: {:.2f}".format(stat))
  print("p-value of the test: {:e}".format(p))
  print("the grand median: {}".format(m), end='\n\n')
```

    Base Cake
    The pearsons chi-square test statistic: 448.81
    p-value of the test: 8.851450e-95
    the grand median: 0.2160487288076019
    
    Truffle Type
    The pearsons chi-square test statistic: 22.86
    p-value of the test: 1.088396e-05
    the grand median: 0.2160487288076019
    
    Primary Flavor
    The pearsons chi-square test statistic: 638.99
    p-value of the test: 3.918933e-103
    the grand median: 0.2160487288076019
    
    Secondary Flavor
    The pearsons chi-square test statistic: 323.13
    p-value of the test: 6.083210e-52
    the grand median: 0.2160487288076019
    
    Color Group
    The pearsons chi-square test statistic: 175.18
    p-value of the test: 1.011412e-31
    the grand median: 0.2160487288076019
    
    Customer
    The pearsons chi-square test statistic: 5.66
    p-value of the test: 2.257760e-01
    the grand median: 0.2160487288076019
    
    Date
    The pearsons chi-square test statistic: 5.27
    p-value of the test: 9.175929e-01
    the grand median: 0.2160487288076019
    


##### **Part D** Many boxplots

And finally, we will confirm these visually. Complete the Boxplot for each group:


```python
for desc in descriptors:
  fig, ax = plt.subplots(figsize=(10,5))
  # sns.boxplot(x=<YOUR X VARIABLE HERE>, y='EBITDA/KG', data=df, color='#A0cbe8', ax=ax)
```

    /usr/local/lib/python3.7/dist-packages/matplotlib/backends/backend_agg.py:214: RuntimeWarning:
    
    Glyph 9 missing from current font.
    
    /usr/local/lib/python3.7/dist-packages/matplotlib/backends/backend_agg.py:183: RuntimeWarning:
    
    Glyph 9 missing from current font.
    



    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_51_1.png)
    



    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_51_2.png)
    



    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_51_3.png)
    



    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_51_4.png)
    



    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_51_5.png)
    



    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_51_6.png)
    



    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_51_7.png)
    


### 2.1.3 **Enrichment**: What is a T-test?

There are 1-sample and 2-sample T-tests 

_(note: we would use a 1-sample T-test just to determine if the sample mean is equal to a hypothesized population mean)_

Within 2-sample T-tests we have **_independent_** and **_dependent_** T-tests (uncorrelated or correlated samples)

For independent, two-sample T-tests:

* **_Equal variance_** (or pooled) T-test
  * `scipy.stats.ttest_ind(equal_var=True)`
* **_Unequal variance_** T-test
  * `scipy.stats.ttest_ind(equal_var=False)`
  * also called ***Welch's T-test***

<br>

For dependent T-tests:
* Paired (or correlated) T-test
  * `scipy.stats.ttest_rel`

A full discussion on T-tests is outside the scope of this session, but we can refer to wikipedia for more information, including formulas on how each statistic is computed:
* [student's T-test](https://en.wikipedia.org/wiki/Student%27s_t-test#Dependent_t-test_for_paired_samples)

### 2.1.4 **Enrichment**: Demonstration of T-tests

[back to top](#top)

We'll assume our shifts are of **_equal variance_** and proceed with the appropriate **_independent two-sample_** T-test...


```python
print(shift_one)
print(shift_two)
```

    [15, 15, 15, 16, 17, 18, 18, 18, 21, 21]
    [15, 16, 17, 18, 18, 19, 20, 20, 22, 22]


To calculate the T-test, we follow a slightly different statistical formula:

$T=\frac{\mu_1 - \mu_2}{s\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$

where $\mu$ are the means of the two groups, $n$ are the sample sizes and $s$ is the pooled standard deviation, also known as the cummulative variance (depending on if you square it or not):

$s= \sqrt{\frac{(n_1-1)\sigma_1^2 + (n_2-1)\sigma_2^2}{n_1 + n_2 - 2}}$

where $\sigma$ are the standard deviations. What you'll notice here is we are combining the two variances, we can only do this if we assume the variances are somewhat equal, this is known as the *equal variances* t-test.


```python
mean_shift_one = np.mean(shift_one)
mean_shift_two = np.mean(shift_two)

print(mean_shift_one, mean_shift_two)
```

    17.4 18.7



```python
com_var = ((np.sum([(i - mean_shift_one)**2 for i in shift_one]) + 
            np.sum([(i - mean_shift_two)**2 for i in shift_two])) /
            (len(shift_one) + len(shift_two)-2))
print(com_var)
```

    5.361111111111111



```python
T = (np.abs(mean_shift_one - mean_shift_two) / (
     np.sqrt(com_var/len(shift_one) +
     com_var/len(shift_two))))
```


```python
T
```




    1.2554544209603191



We see that this hand-computed result matches that of the `scipy` module:


```python
scipy.stats.ttest_ind(shift_two, shift_one, equal_var=True)
```




    Ttest_indResult(statistic=1.2554544209603191, pvalue=0.22536782778843117)



### **Enrichment**: 6.1.5 What are F-statistics and the F-test?

The F-statistic is simply a ratio of two variances, or the ratio of _mean squares_

_mean squares_ is the estimate of population variance that accounts for the degrees of freedom to compute that estimate. 

We will explore this in the context of ANOVA

### 2.1.6 **Enrichment**: What is Analysis of Variance? 

ANOVA uses the F-test to determine whether the variability between group means is larger than the variability within the groups. If that statistic is large enough, you can conclude that the means of the groups are not equal.

**The caveat is that ANOVA tells us whether there is a difference in means but it does not tell us where the difference is.** To find where the difference is between the groups, we have to conduct post-hoc tests.

There are two main types:
* One-way (one factor) and
* Two-way (two factor) where factor is an indipendent variable

<br>

| Ind A | Ind B | Dep |
|-------|-------|-----|
| X     | H     | 10  |
| X     | I     | 12  |
| Y     | I     | 11  |
| Y     | H     | 20  |

<br>

#### ANOVA Hypotheses

* _Null hypothesis_: group means are equal
* _Alternative hypothesis_: at least one group mean is different form the other groups

### ANOVA Assumptions

* Residuals (experimental error) are normally distributed (test with Shapiro-Wilk)
* Homogeneity of variances (variances are equal between groups) (test with Bartlett's)
* Observations are sampled independently from each other
* _Note: ANOVA assumptions can be checked using test statistics (e.g. Shapiro-Wilk, Bartlett’s, Levene’s test) and the visual approaches such as residual plots (e.g. QQ-plots) and histograms._

### Steps for ANOVA

* Check sample sizes: equal observations must be in each group
* Calculate Sum of Square between groups and within groups (\\(SS_B, SS_E\\))
* Calculate Mean Square between groups and within groups (\\(MS_B, MS_E\\))
* Calculate F value (\\(MS_B/MS_E\\))

<br>

This might be easier to see in a table:

<br>

| Source of Variation         | degree of freedom (Df) | Sum of squares (SS) | Mean square (MS)   | F value     |
|-----------------------------|------------------------|---------------------|--------------------|-------------|
| Between Groups             | Df_b = P-1             | SS_B                | MS_B = SS_B / Df_B | MS_B / MS_E |
| Within Groups | Df_E = P(N-1)          | SS_E                | MS_E = SS_E / Df_E |             |
| total                       | Df_T = PN-1            | SS_T           |                    |             |

Where:
$$ SS_B = \sum_{i}^{P}{(\bar{y}_i-\bar{y})^2} $$
<br>
$$ SS_E = \sum_{ik}^{PN}{(\bar{y}_{ik}-\bar{y}_i)^2} $$
<br>
$$ SS_T = SS_B + SS_E $$

Let's go  back to our shift data to take an example:


```python
shifts = pd.DataFrame([shift_one, shift_two, shift_three, shift_four]).T
shifts.columns = ['A', 'B', 'C', 'D']
shifts.boxplot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb5128c6690>




    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_65_1.png)
    


#### 2.1.6.0 **Enrichment**: SNS Boxplot

this is another great way to view boxplot data. Notice how sns also shows us the raw data alongside the box and whiskers using a _swarmplot_.


```python
shift_melt = pd.melt(shifts.reset_index(), id_vars=['index'], 
                     value_vars=['A', 'B', 'C', 'D'])
shift_melt.columns = ['index', 'shift', 'rate']
ax = sns.boxplot(x='shift', y='rate', data=shift_melt, color='#A0cbe8')
ax = sns.swarmplot(x="shift", y="rate", data=shift_melt, color='#79706e')
```


    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_67_0.png)
    


Anyway back to ANOVA...


```python
fvalue, pvalue = stats.f_oneway(shifts['A'], 
                                shifts['B'],
                                shifts['C'],
                                shifts['D'])
print(fvalue, pvalue)
```

    2.8666172656539457 0.04998066540684099


We can get this in the format of the table we saw above:


```python
# get ANOVA table 
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Ordinary Least Squares (OLS) model
model = ols('rate ~ C(shift)', data=shift_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table
# output (ANOVA F and p value)
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
      <th>C(shift)</th>
      <td>64.475</td>
      <td>3.0</td>
      <td>2.866617</td>
      <td>0.049981</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>269.900</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



The **_Shapiro-Wilk_** test can be used to check the _normal distribution of residuals_. Null hypothesis: data is drawn from normal distribution.


```python
w, pvalue = stats.shapiro(model.resid)
print(w, pvalue)
```

    0.9800916314125061 0.6929556727409363


We can use **_Bartlett’s_** test to check the _Homogeneity of variances_. Null hypothesis: samples from populations have equal variances.


```python
w, pvalue = stats.bartlett(shifts['A'], 
                           shifts['B'], 
                           shifts['C'], 
                           shifts['D'])
print(w, pvalue)
```

    1.9492677462621584 0.5830028540285896


#### 2.1.6.1 ANOVA Interpretation

The _p_ value form ANOVA analysis is significant (_p_ < 0.05) and we can conclude there are significant difference between the shifts. But we do not know which shift(s) are different. For this we need to perform a post hoc test. There are a multitude of these that are beyond the scope of this discussion ([Tukey-kramer](https://www.real-statistics.com/one-way-analysis-of-variance-anova/unplanned-comparisons/tukey-kramer-test/) is one such test) 

<p align=center>
<img src="https://media.tenor.com/images/4da4d46c8df02570a9a1219cac42bf27/tenor.gif"></img>
</p>

### 2.1.7 Putting it all together

In summary, there are many statistical tests at our disposal when performing inferential statistical analysis. In times like these, a simple decision tree can be extraordinarily useful!

<img src="https://cdn.scribbr.com/wp-content/uploads//2020/01/flowchart-for-choosing-a-statistical-test.png" width=800px></img>

<small>source: [scribbr](https://www.scribbr.com/statistics/statistical-tests/)</small>

## 2.2 Evaluate statistical significance of product margin: a snake in the garden

### 2.2.1 Mood's Median on product descriptors

The first issue we run into with moods is... what? 

We can only perform moods on two groups at a time. How can we get around this?

Let's take a look at the category with the fewest descriptors. If we remember, this was the Truffle Types.


```python
df.columns
```




    Index(['Base Cake', 'Truffle Type', 'Primary Flavor', 'Secondary Flavor',
           'Color Group', 'Customer', 'Date', 'KG', 'EBITDA/KG'],
          dtype='object')




```python
df['Truffle Type'].unique()
```




    array(['Candy Outer', 'Chocolate Outer', 'Jelly Filled'], dtype=object)




```python
col = 'Truffle Type'
moodsdf = pd.DataFrame()
for truff in df[col].unique():
  
  # for each 
  group = df.loc[df[col] == truff]['EBITDA/KG']
  pop = df.loc[~(df[col] == truff)]['EBITDA/KG']
  stat, p, m, table = scipy.stats.median_test(group, pop)
  median = np.median(group)
  mean = np.mean(group)
  size = len(group)
  print("{}: N={}".format(truff, size))
  print("Welch's T-Test for Unequal Variances")
  print(scipy.stats.ttest_ind(group, pop, equal_var=False))
  welchp = scipy.stats.ttest_ind(group, pop, equal_var=False).pvalue
  print()
  moodsdf = pd.concat([moodsdf, 
                            pd.DataFrame([truff, 
                                          stat, p, m, mean, median, size,
                                          welchp, table]).T])
moodsdf.columns = [col, 'pearsons_chi_square', 'p_value', 
              'grand_median', 'group_mean', 'group_median', 'size', 'welch p',
              'table']
```

    Candy Outer: N=288
    Welch's T-Test for Unequal Variances
    Ttest_indResult(statistic=-2.7615297773427527, pvalue=0.005911048922657976)
    
    Chocolate Outer: N=1356
    Welch's T-Test for Unequal Variances
    Ttest_indResult(statistic=4.409449025092911, pvalue=1.1932685612874952e-05)
    
    Jelly Filled: N=24
    Welch's T-Test for Unequal Variances
    Ttest_indResult(statistic=-8.4142523067935, pvalue=7.929912531660173e-09)
    


### Question 1: Moods Results on Truffle Type

> What do we notice about the resultant table?

* **_p-values_** Most are quite small (really low probability of achieving these table results under a single distribution)
* group sizes: our Jelly Filled group is relatively small


```python
moodsdf.sort_values('p_value')
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
      <th>Truffle Type</th>
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
      <td>Jelly Filled</td>
      <td>18.6432</td>
      <td>1.57604e-05</td>
      <td>0.216049</td>
      <td>0.0513823</td>
      <td>0.0179334</td>
      <td>24</td>
      <td>7.92991e-09</td>
      <td>[[1, 833], [23, 811]]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Chocolate Outer</td>
      <td>6.6275</td>
      <td>0.0100416</td>
      <td>0.216049</td>
      <td>0.262601</td>
      <td>0.225562</td>
      <td>1356</td>
      <td>1.19327e-05</td>
      <td>[[699, 135], [657, 177]]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Candy Outer</td>
      <td>1.51507</td>
      <td>0.218368</td>
      <td>0.216049</td>
      <td>0.230075</td>
      <td>0.204264</td>
      <td>288</td>
      <td>0.00591105</td>
      <td>[[134, 700], [154, 680]]</td>
    </tr>
  </tbody>
</table>
</div>



We can go ahead and repeat this analysis for all of our product categories:


```python
df.columns[:5]
```




    Index(['Base Cake', 'Truffle Type', 'Primary Flavor', 'Secondary Flavor',
           'Color Group'],
          dtype='object')




```python
moodsdf = pd.DataFrame()
for col in df.columns[:5]:
  for truff in df[col].unique():
    group = df.loc[df[col] == truff]['EBITDA/KG']
    pop = df.loc[~(df[col] == truff)]['EBITDA/KG']
    stat, p, m, table = scipy.stats.median_test(group, pop)
    median = np.median(group)
    mean = np.mean(group)
    size = len(group)
    welchp = scipy.stats.ttest_ind(group, pop, equal_var=False).pvalue
    moodsdf = pd.concat([moodsdf, 
                              pd.DataFrame([col, truff, 
                                            stat, p, m, mean, median, size,
                                            welchp, table]).T])
moodsdf.columns = ['descriptor', 'group', 'pearsons_chi_square', 'p_value', 
                'grand_median', 'group_mean', 'group_median', 'size', 'welch p',
                'table']
print(moodsdf.shape)
```

    (101, 10)



```python
moodsdf = moodsdf.loc[(moodsdf['welch p'] < 0.005) &
            (moodsdf['p_value'] < 0.005)].sort_values('group_median')

moodsdf = moodsdf.sort_values('group_median').reset_index(drop=True)
print(moodsdf.shape)
```

    (51, 10)



```python
moodsdf
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
      <td>Secondary Flavor</td>
      <td>Papaya</td>
      <td>18.6432</td>
      <td>1.57604e-05</td>
      <td>0.216049</td>
      <td>0.0167466</td>
      <td>0.00245839</td>
      <td>24</td>
      <td>1.04879e-10</td>
      <td>[[1, 833], [23, 811]]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Primary Flavor</td>
      <td>Orange Pineapple\tP</td>
      <td>18.6432</td>
      <td>1.57604e-05</td>
      <td>0.216049</td>
      <td>0.0167466</td>
      <td>0.00245839</td>
      <td>24</td>
      <td>1.04879e-10</td>
      <td>[[1, 833], [23, 811]]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Primary Flavor</td>
      <td>Cherry Cream Spice</td>
      <td>10.1564</td>
      <td>0.00143801</td>
      <td>0.216049</td>
      <td>0.0187023</td>
      <td>0.00970093</td>
      <td>12</td>
      <td>7.15389e-07</td>
      <td>[[0, 834], [12, 822]]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Secondary Flavor</td>
      <td>Cucumber</td>
      <td>18.6432</td>
      <td>1.57604e-05</td>
      <td>0.216049</td>
      <td>0.0513823</td>
      <td>0.0179334</td>
      <td>24</td>
      <td>7.92991e-09</td>
      <td>[[1, 833], [23, 811]]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Truffle Type</td>
      <td>Jelly Filled</td>
      <td>18.6432</td>
      <td>1.57604e-05</td>
      <td>0.216049</td>
      <td>0.0513823</td>
      <td>0.0179334</td>
      <td>24</td>
      <td>7.92991e-09</td>
      <td>[[1, 833], [23, 811]]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Primary Flavor</td>
      <td>Orange</td>
      <td>18.6432</td>
      <td>1.57604e-05</td>
      <td>0.216049</td>
      <td>0.0513823</td>
      <td>0.0179334</td>
      <td>24</td>
      <td>7.92991e-09</td>
      <td>[[1, 833], [23, 811]]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Primary Flavor</td>
      <td>Toasted Coconut</td>
      <td>15.2613</td>
      <td>9.36173e-05</td>
      <td>0.216049</td>
      <td>0.0370021</td>
      <td>0.0283916</td>
      <td>24</td>
      <td>3.13722e-08</td>
      <td>[[2, 832], [22, 812]]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Secondary Flavor</td>
      <td>Apricot</td>
      <td>15.2613</td>
      <td>9.36173e-05</td>
      <td>0.216049</td>
      <td>0.0603122</td>
      <td>0.0374225</td>
      <td>24</td>
      <td>4.5952e-08</td>
      <td>[[2, 832], [22, 812]]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Primary Flavor</td>
      <td>Kettle Corn</td>
      <td>29.0621</td>
      <td>7.00962e-08</td>
      <td>0.216049</td>
      <td>0.0554518</td>
      <td>0.045891</td>
      <td>60</td>
      <td>6.3013e-18</td>
      <td>[[9, 825], [51, 783]]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Primary Flavor</td>
      <td>Acai Berry</td>
      <td>18.6432</td>
      <td>1.57604e-05</td>
      <td>0.216049</td>
      <td>0.0365051</td>
      <td>0.0494656</td>
      <td>24</td>
      <td>1.49539e-10</td>
      <td>[[1, 833], [23, 811]]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Primary Flavor</td>
      <td>Pink Lemonade</td>
      <td>10.1564</td>
      <td>0.00143801</td>
      <td>0.216049</td>
      <td>0.0398622</td>
      <td>0.0563492</td>
      <td>12</td>
      <td>1.06677e-05</td>
      <td>[[0, 834], [12, 822]]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Secondary Flavor</td>
      <td>Black Cherry</td>
      <td>58.9004</td>
      <td>1.65861e-14</td>
      <td>0.216049</td>
      <td>0.0559745</td>
      <td>0.0628979</td>
      <td>96</td>
      <td>6.65441e-31</td>
      <td>[[11, 823], [85, 749]]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Primary Flavor</td>
      <td>Watermelon</td>
      <td>15.2613</td>
      <td>9.36173e-05</td>
      <td>0.216049</td>
      <td>0.0440497</td>
      <td>0.0678958</td>
      <td>24</td>
      <td>5.20636e-08</td>
      <td>[[2, 832], [22, 812]]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Primary Flavor</td>
      <td>Plum</td>
      <td>34.8516</td>
      <td>3.55816e-09</td>
      <td>0.216049</td>
      <td>0.0849632</td>
      <td>0.0799934</td>
      <td>72</td>
      <td>1.20739e-16</td>
      <td>[[11, 823], [61, 773]]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Secondary Flavor</td>
      <td>Dill Pickle</td>
      <td>10.1564</td>
      <td>0.00143801</td>
      <td>0.216049</td>
      <td>0.0370421</td>
      <td>0.0824944</td>
      <td>12</td>
      <td>7.45639e-06</td>
      <td>[[0, 834], [12, 822]]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Primary Flavor</td>
      <td>Horchata</td>
      <td>10.1564</td>
      <td>0.00143801</td>
      <td>0.216049</td>
      <td>0.0370421</td>
      <td>0.0824944</td>
      <td>12</td>
      <td>7.45639e-06</td>
      <td>[[0, 834], [12, 822]]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Primary Flavor</td>
      <td>Lemon Custard</td>
      <td>12.2175</td>
      <td>0.000473444</td>
      <td>0.216049</td>
      <td>0.0793894</td>
      <td>0.087969</td>
      <td>24</td>
      <td>6.19493e-06</td>
      <td>[[3, 831], [21, 813]]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Primary Flavor</td>
      <td>Fruit Punch</td>
      <td>10.1564</td>
      <td>0.00143801</td>
      <td>0.216049</td>
      <td>0.0789353</td>
      <td>0.0903256</td>
      <td>12</td>
      <td>7.61061e-05</td>
      <td>[[0, 834], [12, 822]]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Base Cake</td>
      <td>Chiffon</td>
      <td>117.046</td>
      <td>2.80454e-27</td>
      <td>0.216049</td>
      <td>0.127851</td>
      <td>0.125775</td>
      <td>288</td>
      <td>6.79655e-43</td>
      <td>[[60, 774], [228, 606]]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Base Cake</td>
      <td>Butter</td>
      <td>134.367</td>
      <td>4.54085e-31</td>
      <td>0.216049</td>
      <td>0.142082</td>
      <td>0.139756</td>
      <td>456</td>
      <td>9.8457e-52</td>
      <td>[[122, 712], [334, 500]]</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Secondary Flavor</td>
      <td>Banana</td>
      <td>10.8053</td>
      <td>0.00101207</td>
      <td>0.216049</td>
      <td>0.163442</td>
      <td>0.15537</td>
      <td>60</td>
      <td>7.23966e-09</td>
      <td>[[17, 817], [43, 791]]</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Primary Flavor</td>
      <td>Cream Soda</td>
      <td>9.51186</td>
      <td>0.00204148</td>
      <td>0.216049</td>
      <td>0.150265</td>
      <td>0.163455</td>
      <td>24</td>
      <td>2.17977e-06</td>
      <td>[[4, 830], [20, 814]]</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Secondary Flavor</td>
      <td>Peppermint</td>
      <td>9.51186</td>
      <td>0.00204148</td>
      <td>0.216049</td>
      <td>0.150265</td>
      <td>0.163455</td>
      <td>24</td>
      <td>2.17977e-06</td>
      <td>[[4, 830], [20, 814]]</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Primary Flavor</td>
      <td>Grand Mariner</td>
      <td>10.5818</td>
      <td>0.00114208</td>
      <td>0.216049</td>
      <td>0.197463</td>
      <td>0.165529</td>
      <td>72</td>
      <td>0.000828508</td>
      <td>[[22, 812], [50, 784]]</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Color Group</td>
      <td>Amethyst</td>
      <td>20.4883</td>
      <td>5.99977e-06</td>
      <td>0.216049</td>
      <td>0.195681</td>
      <td>0.167321</td>
      <td>300</td>
      <td>4.0427e-07</td>
      <td>[[114, 720], [186, 648]]</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Color Group</td>
      <td>Burgundy</td>
      <td>10.9997</td>
      <td>0.000911278</td>
      <td>0.216049</td>
      <td>0.193048</td>
      <td>0.171465</td>
      <td>120</td>
      <td>0.000406312</td>
      <td>[[42, 792], [78, 756]]</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Color Group</td>
      <td>White</td>
      <td>35.7653</td>
      <td>2.22582e-09</td>
      <td>0.216049</td>
      <td>0.19</td>
      <td>0.177264</td>
      <td>432</td>
      <td>1.56475e-16</td>
      <td>[[162, 672], [270, 564]]</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Color Group</td>
      <td>Opal</td>
      <td>11.5872</td>
      <td>0.000664086</td>
      <td>0.216049</td>
      <td>0.317878</td>
      <td>0.259304</td>
      <td>324</td>
      <td>3.94098e-07</td>
      <td>[[190, 644], [134, 700]]</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Secondary Flavor</td>
      <td>Apple</td>
      <td>27.2833</td>
      <td>1.75723e-07</td>
      <td>0.216049</td>
      <td>0.326167</td>
      <td>0.293876</td>
      <td>36</td>
      <td>0.00117635</td>
      <td>[[34, 800], [2, 832]]</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Secondary Flavor</td>
      <td>Tangerine</td>
      <td>32.6264</td>
      <td>1.11688e-08</td>
      <td>0.216049</td>
      <td>0.342314</td>
      <td>0.319273</td>
      <td>48</td>
      <td>0.000112572</td>
      <td>[[44, 790], [4, 830]]</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Secondary Flavor</td>
      <td>Black Currant</td>
      <td>34.7784</td>
      <td>3.69452e-09</td>
      <td>0.216049</td>
      <td>0.357916</td>
      <td>0.332449</td>
      <td>36</td>
      <td>9.48594e-08</td>
      <td>[[36, 798], [0, 834]]</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Secondary Flavor</td>
      <td>Pear</td>
      <td>16.6143</td>
      <td>4.58043e-05</td>
      <td>0.216049</td>
      <td>0.373034</td>
      <td>0.33831</td>
      <td>60</td>
      <td>3.05948e-05</td>
      <td>[[46, 788], [14, 820]]</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Primary Flavor</td>
      <td>Vanilla</td>
      <td>34.7784</td>
      <td>3.69452e-09</td>
      <td>0.216049</td>
      <td>0.378053</td>
      <td>0.341626</td>
      <td>36</td>
      <td>1.00231e-06</td>
      <td>[[36, 798], [0, 834]]</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Color Group</td>
      <td>Citrine</td>
      <td>10.1564</td>
      <td>0.00143801</td>
      <td>0.216049</td>
      <td>0.390728</td>
      <td>0.342512</td>
      <td>12</td>
      <td>0.00192513</td>
      <td>[[12, 822], [0, 834]]</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Color Group</td>
      <td>Teal</td>
      <td>13.5397</td>
      <td>0.000233572</td>
      <td>0.216049</td>
      <td>0.323955</td>
      <td>0.3446</td>
      <td>96</td>
      <td>0.00120954</td>
      <td>[[66, 768], [30, 804]]</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Base Cake</td>
      <td>Tiramisu</td>
      <td>52.3606</td>
      <td>4.61894e-13</td>
      <td>0.216049</td>
      <td>0.388267</td>
      <td>0.362102</td>
      <td>144</td>
      <td>7.98485e-12</td>
      <td>[[114, 720], [30, 804]]</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Primary Flavor</td>
      <td>Doughnut</td>
      <td>74.9353</td>
      <td>4.86406e-18</td>
      <td>0.216049</td>
      <td>0.439721</td>
      <td>0.379361</td>
      <td>108</td>
      <td>2.47855e-15</td>
      <td>[[98, 736], [10, 824]]</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Secondary Flavor</td>
      <td>Ginger Beer</td>
      <td>22.3634</td>
      <td>2.25628e-06</td>
      <td>0.216049</td>
      <td>0.444895</td>
      <td>0.382283</td>
      <td>24</td>
      <td>0.000480693</td>
      <td>[[24, 810], [0, 834]]</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Color Group</td>
      <td>Rose</td>
      <td>18.6432</td>
      <td>1.57604e-05</td>
      <td>0.216049</td>
      <td>0.42301</td>
      <td>0.407061</td>
      <td>24</td>
      <td>6.16356e-05</td>
      <td>[[23, 811], [1, 833]]</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Base Cake</td>
      <td>Cheese</td>
      <td>66.8047</td>
      <td>2.99776e-16</td>
      <td>0.216049</td>
      <td>0.450934</td>
      <td>0.435638</td>
      <td>84</td>
      <td>4.5139e-18</td>
      <td>[[79, 755], [5, 829]]</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Primary Flavor</td>
      <td>Butter Toffee</td>
      <td>60.1815</td>
      <td>8.65028e-15</td>
      <td>0.216049</td>
      <td>0.50366</td>
      <td>0.456343</td>
      <td>60</td>
      <td>2.09848e-19</td>
      <td>[[60, 774], [0, 834]]</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Color Group</td>
      <td>Slate</td>
      <td>10.1564</td>
      <td>0.00143801</td>
      <td>0.216049</td>
      <td>0.540214</td>
      <td>0.483138</td>
      <td>12</td>
      <td>1.69239e-05</td>
      <td>[[12, 822], [0, 834]]</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Primary Flavor</td>
      <td>Gingersnap</td>
      <td>22.3634</td>
      <td>2.25628e-06</td>
      <td>0.216049</td>
      <td>0.643218</td>
      <td>0.623627</td>
      <td>24</td>
      <td>9.02005e-16</td>
      <td>[[24, 810], [0, 834]]</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Primary Flavor</td>
      <td>Dill Pickle</td>
      <td>22.3634</td>
      <td>2.25628e-06</td>
      <td>0.216049</td>
      <td>0.642239</td>
      <td>0.655779</td>
      <td>24</td>
      <td>5.14558e-16</td>
      <td>[[24, 810], [0, 834]]</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Color Group</td>
      <td>Olive</td>
      <td>44.9675</td>
      <td>2.00328e-11</td>
      <td>0.216049</td>
      <td>0.637627</td>
      <td>0.670186</td>
      <td>60</td>
      <td>3.71909e-20</td>
      <td>[[56, 778], [4, 830]]</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Primary Flavor</td>
      <td>Butter Milk</td>
      <td>10.1564</td>
      <td>0.00143801</td>
      <td>0.216049</td>
      <td>0.699284</td>
      <td>0.688601</td>
      <td>12</td>
      <td>2.76773e-07</td>
      <td>[[12, 822], [0, 834]]</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Base Cake</td>
      <td>Sponge</td>
      <td>127.156</td>
      <td>1.71707e-29</td>
      <td>0.216049</td>
      <td>0.698996</td>
      <td>0.699355</td>
      <td>120</td>
      <td>2.5584e-80</td>
      <td>[[120, 714], [0, 834]]</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Primary Flavor</td>
      <td>Chocolate Mint</td>
      <td>10.1564</td>
      <td>0.00143801</td>
      <td>0.216049</td>
      <td>0.685546</td>
      <td>0.699666</td>
      <td>12</td>
      <td>1.48794e-07</td>
      <td>[[12, 822], [0, 834]]</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Primary Flavor</td>
      <td>Coconut</td>
      <td>10.1564</td>
      <td>0.00143801</td>
      <td>0.216049</td>
      <td>0.732777</td>
      <td>0.717641</td>
      <td>12</td>
      <td>3.06716e-14</td>
      <td>[[12, 822], [0, 834]]</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Primary Flavor</td>
      <td>Blueberry</td>
      <td>22.3634</td>
      <td>2.25628e-06</td>
      <td>0.216049</td>
      <td>0.759643</td>
      <td>0.72536</td>
      <td>24</td>
      <td>9.11605e-14</td>
      <td>[[24, 810], [0, 834]]</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Primary Flavor</td>
      <td>Amaretto</td>
      <td>10.1564</td>
      <td>0.00143801</td>
      <td>0.216049</td>
      <td>0.782156</td>
      <td>0.764845</td>
      <td>12</td>
      <td>8.53642e-10</td>
      <td>[[12, 822], [0, 834]]</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2.2 **Enrichment**: Broad Analysis of Categories: ANOVA



Recall our "melted" shift data. It will be useful to think of getting our Truffle data in this format:


```python
shift_melt.head()
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
      <th>index</th>
      <th>shift</th>
      <th>rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>A</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>A</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>A</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>A</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>A</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('/', '_')
```


```python
# get ANOVA table 
# Ordinary Least Squares (OLS) model
model = ols('EBITDA_KG ~ C(Truffle_Type)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table
# output (ANOVA F and p value)
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
      <th>C(Truffle_Type)</th>
      <td>1.250464</td>
      <td>2.0</td>
      <td>12.882509</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>80.808138</td>
      <td>1665.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Recall the **_Shapiro-Wilk_** test can be used to check the _normal distribution of residuals_. Null hypothesis: data is drawn from normal distribution.


```python
w, pvalue = stats.shapiro(model.resid)
print(w, pvalue)
```

    0.9576056599617004 1.2598073820281984e-21


And the **_Bartlett’s_** test to check the _Homogeneity of variances_. Null hypothesis: samples from populations have equal variances.


```python
gb = df.groupby('Truffle_Type')['EBITDA_KG']
gb
```




    <pandas.core.groupby.generic.SeriesGroupBy object at 0x7fb5127814d0>




```python
w, pvalue = stats.bartlett(*[gb.get_group(x) for x in gb.groups])
print(w, pvalue)
```

    109.93252546442552 1.344173733366234e-24


Wow it looks like our data is not drawn from a normal distribution! Let's check this for other categories...

We can wrap these in a for loop:


```python
for col in df.columns[:5]:
  print(col)
  model = ols('EBITDA_KG ~ C({})'.format(col), data=df).fit()
  anova_table = sm.stats.anova_lm(model, typ=2)
  display(anova_table)
  w, pvalue = stats.shapiro(model.resid)
  print("Shapiro: ", w, pvalue)
  gb = df.groupby(col)['EBITDA_KG']
  w, pvalue = stats.bartlett(*[gb.get_group(x) for x in gb.groups])
  print("Bartlett: ", w, pvalue)
  print()
```

    Base_Cake



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
      <th>C(Base_Cake)</th>
      <td>39.918103</td>
      <td>5.0</td>
      <td>314.869955</td>
      <td>1.889884e-237</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>42.140500</td>
      <td>1662.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Shapiro:  0.9634131193161011 4.1681337029688696e-20
    Bartlett:  69.83288886114195 1.1102218566053728e-13
    
    Truffle_Type



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
      <th>C(Truffle_Type)</th>
      <td>1.250464</td>
      <td>2.0</td>
      <td>12.882509</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>80.808138</td>
      <td>1665.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Shapiro:  0.9576056599617004 1.2598073820281984e-21
    Bartlett:  109.93252546442552 1.344173733366234e-24
    
    Primary_Flavor



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
      <th>C(Primary_Flavor)</th>
      <td>50.270639</td>
      <td>50.0</td>
      <td>51.143649</td>
      <td>1.153434e-292</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>31.787964</td>
      <td>1617.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Shapiro:  0.948470413684845 9.90281706784179e-24
    Bartlett:  210.15130419114894 1.5872504991231547e-21
    
    Secondary_Flavor



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
      <th>C(Secondary_Flavor)</th>
      <td>15.088382</td>
      <td>28.0</td>
      <td>13.188089</td>
      <td>1.929302e-54</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>66.970220</td>
      <td>1639.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Shapiro:  0.9548103213310242 2.649492974953278e-22
    Bartlett:  420.6274502894803 1.23730070350945e-71
    
    Color_Group



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
      <th>C(Color_Group)</th>
      <td>16.079685</td>
      <td>11.0</td>
      <td>36.689347</td>
      <td>6.544980e-71</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>65.978918</td>
      <td>1656.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Shapiro:  0.969061017036438 1.8926407335144587e-18
    Bartlett:  136.55525281340468 8.164787784033709e-24
    


### 2.2.3 **Enrichment**: Visual Analysis of Residuals: QQ-Plots

This can be distressing and is often why we want visual methods to see what is going on with our data!


```python
model = ols('EBITDA_KG ~ C(Truffle_Type)', data=df).fit()

#create instance of influence
influence = model.get_influence()

#obtain standardized residuals
standardized_residuals = influence.resid_studentized_internal

# res.anova_std_residuals are standardized residuals obtained from ANOVA (check above)
sm.qqplot(standardized_residuals, line='45')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()

# histogram
plt.hist(model.resid, bins='auto', histtype='bar', ec='k') 
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()
```


    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_103_0.png)
    



    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_103_1.png)
    


We see that a lot of our data is swayed by extremely high and low values, so what can we conclude? 

> You need the right test statistic for the right job, in this case, we are littered with unequal variance in our groupings so we use the moods median and welch (unequal variance t-test) to make conclusions about our data


# References

* [Renesh Bedre ANOVA](https://www.reneshbedre.com/blog/anova.html)
* [Minitab ANOVA](https://blog.minitab.com/en/adventures-in-statistics-2/understanding-analysis-of-variance-anova-and-the-f-test)
* [Analytics Vidhya ANOVA](https://www.analyticsvidhya.com/blog/2020/06/introduction-anova-statistics-data-science-covid-python/)
* [Renesh Bedre Hypothesis Testing](https://www.reneshbedre.com/blog/hypothesis-testing.html)
* [Real Statistics Turkey-kramer](https://www.real-statistics.com/one-way-analysis-of-variance-anova/unplanned-comparisons/tukey-kramer-test/)
* [Mutual Information](https://www.kaggle.com/ryanholbrook/mutual-information)