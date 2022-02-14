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
for col in df.columns[:-2]:
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
    
    Color Group
    ['Taupe' 'Amethyst' 'Burgundy' 'White' 'Black' 'Opal' 'Citrine' 'Rose'
     'Slate' 'Teal' 'Tiffany' 'Olive']
    
    Customer
    ['Slugworth' 'Perk-a-Cola' 'Fickelgruber' 'Zebrabar' "Dandy's Candies"]
    
    Date
    ['1/2020' '2/2020' '3/2020' '4/2020' '5/2020' '6/2020' '7/2020' '8/2020'
     '9/2020' '10/2020' '11/2020' '12/2020']
    


## 2.1 Many Flavors of Statistical Tests

<p align="center">
<img src="https://luminousmen.com/media/descriptive-and-inferential-statistics.jpeg" width=400px></img>
</p>

<small>[img src](https://luminousmen.com/post/descriptive-and-inferential-statistics)</small>


> Descriptive statistics describes data (for example, a chart or graph) and inferential statistics allows you to make predictions (“inferences”) from that data. With inferential statistics, you take data from samples and make generalizations about a population 

[statshowto](https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/inferential-statistics/)

* Non-parametric tests
    * [Moods Median](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_test.html) (Comparison of Medians)
    * [Kruskal-Wallis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html#scipy.stats.kruskal)  (Comparison of Medians, compare to _ANOVA_)
    * [Mann Whitney](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#scipy.stats.mannwhitneyu) (Rank order test, compare to _T-test_)
* Comparison of means
    * [T-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html#scipy.stats.ttest_ind)
        * Independent
            * Equal Variances (students T-test)
            * Unequal Variances (Welch's T-test)
        * Dependent
* Comparison of variances
    * Analysis of Variance (ANOVA)
        * One Way ANOVA
        * Two Way ANOVA
        * MANOVA
        * Factorial ANOVA

When do I use each of these? We will talk about this as we proceed through the examples. [This page](https://support.minitab.com/en-us/minitab/20/help-and-how-to/statistics/nonparametrics/supporting-topics/which-test-should-i-use/) from minitab has good rules of thumb on the subject.



### 2.1.1 What is Mood's Median?

> You can use Chi-Square to test for a goodness of fit (whether a sample of data represents a distribution) or whether two variables are related (using a contingency table, which we will create below!)

**A special case of Pearon's Chi-Squared Test:** We create a table that counts the observations above and below the global median for **two or more groups**. We then perform a *chi-squared test of significance* on this *contingency table* 

Null hypothesis: the Medians are all equal

The chi-square test statistic:

$$x^2 = \sum{\frac{(O-E)^2}{E}}$$

Where \\(O\\) is the observed frequency and \\(E\\) is the expected frequency.

**Let's take an example**, say we have three shifts with the following production rates:


```python
np.random.seed(7)
shift_one = [round(i) for i in np.random.normal(16, 3, 10)]
shift_two = [round(i) for i in np.random.normal(21, 3, 10)]
```


```python
print(shift_one)
print(shift_two)
```

    [21, 15, 16, 17, 14, 16, 16, 11, 19, 18]
    [19, 20, 23, 20, 20, 17, 23, 21, 22, 16]



```python
stat, p, m, table = scipy.stats.median_test(shift_one, shift_two, correction=False)
```

what is `median_test` returning?


```python
print("The pearsons chi-square test statistic: {:.2f}".format(stat))
print("p-value of the test: {:.3f}".format(p))
print("the grand median: {}".format(m))
```

    The pearsons chi-square test statistic: 7.20
    p-value of the test: 0.007
    the grand median: 18.5


Let's evaluate that test statistic ourselves by taking a look at the contingency table:


```python
table
```




    array([[2, 8],
           [8, 2]])



This is easier to make sense of if we order the shift times


```python
shift_one.sort()
shift_one
```




    [11, 14, 15, 16, 16, 16, 17, 18, 19, 21]



When we look at shift one, we see that 8 values are at or below the grand median.


```python
shift_two.sort()
shift_two
```




    [16, 17, 19, 20, 20, 20, 21, 22, 23, 23]



For shift two, only two are at or below the grand median.

Since the sample sizes are the same, the expected value for both groups is the same, 5 above and 5 below the grand median. The chi-square is then:

$$X^2 = \frac{(2-5)^2}{5} + \frac{(8-5)^2}{5} + \frac{(8-5)^2}{5} + \frac{(2-5)^2}{5}$$



```python
(2-5)**2/5 + (8-5)**2/5 + (8-5)**2/5 + (2-5)**2/5
```




    7.2



Our p-value, or the probability of observing the null-hypothsis, is under 0.05 (at 0.007). We can conclude that these shift performances were drawn under seperate distributions.

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


##### **Part A** Perform moods median test on Base Cake (Categorical Variable) and EBITDA/KG (Continuous Variable) in Truffle data

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

    ('Butter',      Base Cake     Truffle Type Primary Flavor Secondary Flavor Color Group  \
    0       Butter      Candy Outer   Butter Pecan           Toffee       Taupe   
    1       Butter      Candy Outer    Ginger Lime           Banana    Amethyst   
    2       Butter      Candy Outer    Ginger Lime           Banana    Burgundy   
    3       Butter      Candy Outer    Ginger Lime           Banana       White   
    4       Butter      Candy Outer    Ginger Lime              Rum    Amethyst   
    ...        ...              ...            ...              ...         ...   
    1562    Butter  Chocolate Outer           Plum     Black Cherry        Opal   
    1563    Butter  Chocolate Outer           Plum     Black Cherry       White   
    1564    Butter  Chocolate Outer           Plum            Mango       Black   
    1565    Butter     Jelly Filled         Orange         Cucumber    Amethyst   
    1566    Butter     Jelly Filled         Orange         Cucumber    Burgundy   
    
                 Customer     Date             KG  EBITDA/KG  
    0           Slugworth   1/2020   53770.342593   0.500424  
    1           Slugworth   1/2020  466477.578125   0.220395  
    2         Perk-a-Cola   1/2020   80801.728070   0.171014  
    3        Fickelgruber   1/2020   18046.111111   0.233025  
    4        Fickelgruber   1/2020   19147.454268   0.480689  
    ...               ...      ...            ...        ...  
    1562     Fickelgruber  12/2020    9772.200521   0.158279  
    1563      Perk-a-Cola  12/2020   10861.245675  -0.159275  
    1564        Slugworth  12/2020    3578.592163   0.431328  
    1565        Slugworth  12/2020   21438.187500   0.105097  
    1566  Dandy's Candies  12/2020   15617.489115   0.185070  
    
    [456 rows x 9 columns])



```python
# the first object appears to be the group
print(i[0])

# the second object appears to be the df belonging to that group
print(i[1])
```

    Butter
         Base Cake     Truffle Type Primary Flavor Secondary Flavor Color Group  \
    0       Butter      Candy Outer   Butter Pecan           Toffee       Taupe   
    1       Butter      Candy Outer    Ginger Lime           Banana    Amethyst   
    2       Butter      Candy Outer    Ginger Lime           Banana    Burgundy   
    3       Butter      Candy Outer    Ginger Lime           Banana       White   
    4       Butter      Candy Outer    Ginger Lime              Rum    Amethyst   
    ...        ...              ...            ...              ...         ...   
    1562    Butter  Chocolate Outer           Plum     Black Cherry        Opal   
    1563    Butter  Chocolate Outer           Plum     Black Cherry       White   
    1564    Butter  Chocolate Outer           Plum            Mango       Black   
    1565    Butter     Jelly Filled         Orange         Cucumber    Amethyst   
    1566    Butter     Jelly Filled         Orange         Cucumber    Burgundy   
    
                 Customer     Date             KG  EBITDA/KG  
    0           Slugworth   1/2020   53770.342593   0.500424  
    1           Slugworth   1/2020  466477.578125   0.220395  
    2         Perk-a-Cola   1/2020   80801.728070   0.171014  
    3        Fickelgruber   1/2020   18046.111111   0.233025  
    4        Fickelgruber   1/2020   19147.454268   0.480689  
    ...               ...      ...            ...        ...  
    1562     Fickelgruber  12/2020    9772.200521   0.158279  
    1563      Perk-a-Cola  12/2020   10861.245675  -0.159275  
    1564        Slugworth  12/2020    3578.592163   0.431328  
    1565        Slugworth  12/2020   21438.187500   0.105097  
    1566  Dandy's Candies  12/2020   15617.489115   0.185070  
    
    [456 rows x 9 columns]


going back to our diagram from our earlier pandas session. It looks like whenever we split in the groupby method, we create separate dataframes as well as their group label:

<img src="https://swcarpentry.github.io/r-novice-gapminder/fig/12-plyr-fig1.png" width=500></img>

Ok, so we know `gp` is separate dataframes. How do we turn them into arrays to then pass to `median_test`?


```python
# complete this for loop
for i, j in gp:
  pass
  # turn 'EBITDA/KG' of j into an array using the .values attribute
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

    The pearsons chi-square test statistic: 0.00
    p-value of the test: 1.00e+00
    the grand median: 1.55e+01


##### **Part B** View the distributions of the data using matplotlib and seaborn

What a fantastic statistical result we found! Can we affirm our result with some visualizations? I hope so! Create a boxplot below using pandas. In your call to `df.boxplot()` the `by` parameter should be set to `Base Cake` and the `column` parameter should be set to `EBITDA/KG`


```python
# YOUR BOXPLOT HERE
```

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
    The pearsons chi-square test statistic: 0.00
    p-value of the test: 1.000000e+00
    the grand median: 15.5
    
    Truffle Type
    The pearsons chi-square test statistic: 0.00
    p-value of the test: 1.000000e+00
    the grand median: 15.5
    
    Primary Flavor
    The pearsons chi-square test statistic: 0.00
    p-value of the test: 1.000000e+00
    the grand median: 15.5
    
    Secondary Flavor
    The pearsons chi-square test statistic: 0.00
    p-value of the test: 1.000000e+00
    the grand median: 15.5
    
    Color Group
    The pearsons chi-square test statistic: 0.00
    p-value of the test: 1.000000e+00
    the grand median: 15.5
    
    Customer
    The pearsons chi-square test statistic: 0.00
    p-value of the test: 1.000000e+00
    the grand median: 15.5
    
    Date
    The pearsons chi-square test statistic: 0.00
    p-value of the test: 1.000000e+00
    the grand median: 15.5
    


##### **Part D** Many boxplots

And finally, we will confirm these visually. Complete the Boxplot for each group:


```python
for desc in descriptors:
  fig, ax = plt.subplots(figsize=(10,5))
  # sns.boxplot(x=<YOUR X VARIABLE HERE>, y='EBITDA/KG', data=df, color='#A0cbe8', ax=ax)
```


    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_51_0.png)
    



    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_51_1.png)
    



    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_51_2.png)
    



    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_51_3.png)
    



    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_51_4.png)
    



    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_51_5.png)
    



    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_51_6.png)
    


### 2.1.3 What is a T-test?

There are 1-sample and 2-sample T-tests 

_(note: we would use a 1-sample T-test just to determine if the sample mean is equal to a hypothesized population mean)_

Within 2-sample T-tests we have **_independent_** and **_dependent_** T-tests (uncorrelated or correlated samples)

For independent, two-sample T-tests:

* **_Equal variance_** (or pooled) T-test
    * `scipy.stats.ttest_ind(equal_var=True)`
    * also called ***Student's T-test***
* **_Unequal variance_** T-test
    * `scipy.stats.ttest_ind(equal_var=False)`
    * also called ***Welch's T-test***

<br>

For dependent T-tests:

* Paired (or correlated) T-test
    * `scipy.stats.ttest_rel`
    * _ex_: patient symptoms before and after treatment

A full discussion on T-tests is outside the scope of this session, but we can refer to wikipedia for more information, including formulas on how each statistic is computed:

* [student's T-test](https://en.wikipedia.org/wiki/Student%27s_t-test#Dependent_t-test_for_paired_samples)

#### 2.1.3.1 Demonstration of T-tests

[back to top](#top)

We'll assume our shifts are of **_equal variance_** and proceed with the appropriate **_independent two-sample_** T-test...


```python
print(shift_one)
print(shift_two)
```

    [11, 14, 15, 16, 16, 16, 17, 18, 19, 21]
    [16, 17, 19, 20, 20, 20, 21, 22, 23, 23]


To calculate the T-test, we follow a slightly different statistical formula:

$$T=\frac{\mu_1 - \mu_2}{s\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$$

where \\(\mu\\) are the means of the two groups, \\(n\\) are the sample sizes and \\(s\\) is the pooled standard deviation, also known as the cummulative variance (depending on if you square it or not):

$$s= \sqrt{\frac{(n_1-1)\sigma_1^2 + (n_2-1)\sigma_2^2}{n_1 + n_2 - 2}}$$

where \\(\sigma\\) are the standard deviations. What you'll notice here is we are combining the two variances, we can only do this if we assume the variances are somewhat equal, this is known as the *equal variances* t-test.


```python
mean_shift_one = np.mean(shift_one)
mean_shift_two = np.mean(shift_two)

print(mean_shift_one, mean_shift_two)
```

    16.3 20.1



```python
com_var = ((np.sum([(i - mean_shift_one)**2 for i in shift_one]) + 
            np.sum([(i - mean_shift_two)**2 for i in shift_two])) /
            (len(shift_one) + len(shift_two)-2))
print(com_var)
```

    6.5



```python
T = (np.abs(mean_shift_one - mean_shift_two) / (
     np.sqrt(com_var/len(shift_one) +
     com_var/len(shift_two))))
```


```python
T
```




    3.3328204733667115



We see that this hand-computed result matches that of the `scipy` module:


```python
scipy.stats.ttest_ind(shift_two, shift_one, equal_var=True)
```




    Ttest_indResult(statistic=3.3328204733667115, pvalue=0.0037029158660758575)



### 2.1.4 What are F-statistics and the F-test?

The F-statistic is simply a ratio of two variances, or the ratio of _mean squares_

_mean squares_ is the estimate of population variance that accounts for the degrees of freedom to compute that estimate. 

We will explore this in the context of ANOVA

#### 2.1.4.1 What is Analysis of Variance? 

ANOVA uses the F-test to determine whether the variability between group means is larger than the variability within the groups. If that statistic is large enough, you can conclude that the means of the groups are not equal.

**The caveat is that ANOVA tells us whether there is a difference in means but it does not tell us where the difference is.** To find where the difference is between the groups, we have to conduct post-hoc tests.

There are two main types:

* One-way (one factor) and
* Two-way (two factor) where factor is an independent variable

<br>

| Ind A | Ind B | Dep |
|-------|-------|-----|
| X     | H     | 10  |
| X     | I     | 12  |
| Y     | I     | 11  |
| Y     | H     | 20  |

<br>

##### ANOVA Hypotheses

* _Null hypothesis_: group means are equal
* _Alternative hypothesis_: at least one group mean is different from the other groups

##### ANOVA Assumptions

* Residuals (experimental error) are normally distributed (test with Shapiro-Wilk)
* Homogeneity of variances (variances are equal between groups) (test with Bartlett's)
* Observations are sampled independently from each other
* _Note: ANOVA assumptions can be checked using test statistics (e.g. Shapiro-Wilk, Bartlett’s, Levene’s test) and the visual approaches such as residual plots (e.g. QQ-plots) and histograms._

##### Steps for ANOVA

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




    <AxesSubplot:>




    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_65_1.png)
    


#### 2.1.4.2 SNS Boxplot

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

    5.599173553719008 0.0029473487978665873


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
      <td>135.5</td>
      <td>3.0</td>
      <td>5.599174</td>
      <td>0.002947</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>290.4</td>
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

    0.9750654697418213 0.5121709108352661


We can use **_Bartlett’s_** test to check the _Homogeneity of variances_. Null hypothesis: samples from populations have equal variances.


```python
w, pvalue = stats.bartlett(shifts['A'], 
                           shifts['B'], 
                           shifts['C'], 
                           shifts['D'])
print(w, pvalue)
```

    1.3763632854696672 0.711084540821183


#### 2.1.4.3 ANOVA Interpretation

The _p_ value form ANOVA analysis is significant (_p_ < 0.05) and we can conclude there are significant difference between the shifts. But we do not know which shift(s) are different. For this we need to perform a post hoc test. There are a multitude of these that are beyond the scope of this discussion ([Tukey-kramer](https://www.real-statistics.com/one-way-analysis-of-variance-anova/unplanned-comparisons/tukey-kramer-test/) is one such test) 

<p align=center>
<img src="https://media.tenor.com/images/4da4d46c8df02570a9a1219cac42bf27/tenor.gif"></img>
</p>

### 2.1.5 Putting it all together

In summary, there are many statistical tests at our disposal when performing inferential statistical analysis. In times like these, a simple decision tree can be extraordinarily useful!

<img src="https://cdn.scribbr.com/wp-content/uploads//2020/01/flowchart-for-choosing-a-statistical-test.png" width=800px></img>

<small>source: [scribbr](https://www.scribbr.com/statistics/statistical-tests/)</small>

## 🍒 2.2 Enrichment: Evaluate statistical significance of product margin: a snake in the garden

### 2.2.1 Mood's Median on product descriptors

The first issue we run into with moods is... what? 

Since Mood's is nonparametric, we can easily become overconfident in our results. Let's take an example, continuing with the `Truffle Type` column. Recall that there are 3 unique Truffle Types:


```python
df['Truffle Type'].unique()
```




    array(['Candy Outer', 'Chocolate Outer', 'Jelly Filled'], dtype=object)



We can loop through each group and compute the:

* Moods test (comparison of medians)
* Welch's T-test (unequal variances, comparison of means) 
* Shapiro-Wilk test for normality


```python
col = 'Truffle Type'
moodsdf = pd.DataFrame()
confidence_level = 0.01
welch_rej = mood_rej = shapiro_rej = False
for truff in df[col].unique():
  
  # for each 
  group = df.loc[df[col] == truff]['EBITDA/KG']
  pop = df.loc[~(df[col] == truff)]['EBITDA/KG']
  stat, p, m, table = scipy.stats.median_test(group, pop)
  if p < confidence_level:
        mood_rej = True
  median = np.median(group)
  mean = np.mean(group)
  size = len(group)
  print("{}: N={}".format(truff, size))
  print("Moods Median Test")
  print("\tstatistic={:.2f}, pvalue={:.2e}".format(stat, p), end="\n")
  print(f"\treject: {mood_rej}")
  print("Welch's T-Test")
  print("\tstatistic={:.2f}, pvalue={:.2e}".format(*scipy.stats.ttest_ind(group, pop, equal_var=False)))
  welchp = scipy.stats.ttest_ind(group, pop, equal_var=False).pvalue
  if welchp < confidence_level:
        welch_rej = True
  print(f"\treject: {welch_rej}")
  print("Shapiro-Wilk")
  print("\tstatistic={:.2f}, pvalue={:.2e}".format(*stats.shapiro(group)))
  if stats.shapiro(group).pvalue < confidence_level:
    shapiro_rej = True
  print(f"\treject: {shapiro_rej}")
  print(end="\n\n")
  moodsdf = pd.concat([moodsdf, 
                            pd.DataFrame([truff, 
                                          stat, p, m, mean, median, size,
                                          welchp, table]).T])
moodsdf.columns = [col, 'pearsons_chi_square', 'p_value', 
              'grand_median', 'group_mean', 'group_median', 'size', 'welch p',
              'table']
```

    Candy Outer: N=288
    Moods Median Test
    	statistic=1.52, pvalue=2.18e-01
    	reject: False
    Welch's T-Test
    	statistic=-2.76, pvalue=5.91e-03
    	reject: True
    Shapiro-Wilk
    	statistic=0.96, pvalue=2.74e-07
    	reject: True
    
    
    Chocolate Outer: N=1356
    Moods Median Test
    	statistic=6.63, pvalue=1.00e-02
    	reject: False
    Welch's T-Test
    	statistic=4.41, pvalue=1.19e-05
    	reject: True
    Shapiro-Wilk
    	statistic=0.96, pvalue=6.30e-19
    	reject: True
    
    
    Jelly Filled: N=24
    Moods Median Test
    	statistic=18.64, pvalue=1.58e-05
    	reject: True
    Welch's T-Test
    	statistic=-8.41, pvalue=7.93e-09
    	reject: True
    Shapiro-Wilk
    	statistic=0.87, pvalue=6.56e-03
    	reject: True
    
    


### 🙋‍♀️ Question 1: Moods Results on Truffle Type

What can we say about these results? 

Recall that:

* Shapiro-Wilk Null Hypothesis: the distribution is normal
* Welch's T-test: requires that the distributions be normal
* Moods test: does not require normality in distributions

> main conclusions: all the groups are non-normal and therefore welch's test is invalid. We saw that the Welch's test had much lower p-values than the Moods median test. This is good news! It means that our Moods test, while allowing for non-normality, is much more conservative in its test-statistic, and therefore was unable to reject the null hypothesis in the cases of the chocolate outer and candy outer groups

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
moodsdf = moodsdf.loc[(moodsdf['p_value'] < confidence_level)].sort_values('group_median')

moodsdf = moodsdf.sort_values('group_median').reset_index(drop=True)
print(moodsdf.shape)
```

    (57, 10)



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
      <td>Wild Cherry Cream</td>
      <td>6.798913</td>
      <td>0.009121</td>
      <td>0.216049</td>
      <td>-0.122491</td>
      <td>-0.15791</td>
      <td>12</td>
      <td>0.00001</td>
      <td>[[1, 833], [11, 823]]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Primary Flavor</td>
      <td>Lemon Bar</td>
      <td>6.798913</td>
      <td>0.009121</td>
      <td>0.216049</td>
      <td>-0.122491</td>
      <td>-0.15791</td>
      <td>12</td>
      <td>0.00001</td>
      <td>[[1, 833], [11, 823]]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Primary Flavor</td>
      <td>Orange Pineapple\tP</td>
      <td>18.643248</td>
      <td>0.000016</td>
      <td>0.216049</td>
      <td>0.016747</td>
      <td>0.002458</td>
      <td>24</td>
      <td>0.0</td>
      <td>[[1, 833], [23, 811]]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Secondary Flavor</td>
      <td>Papaya</td>
      <td>18.643248</td>
      <td>0.000016</td>
      <td>0.216049</td>
      <td>0.016747</td>
      <td>0.002458</td>
      <td>24</td>
      <td>0.0</td>
      <td>[[1, 833], [23, 811]]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Primary Flavor</td>
      <td>Cherry Cream Spice</td>
      <td>10.156401</td>
      <td>0.001438</td>
      <td>0.216049</td>
      <td>0.018702</td>
      <td>0.009701</td>
      <td>12</td>
      <td>0.000001</td>
      <td>[[0, 834], [12, 822]]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Secondary Flavor</td>
      <td>Cucumber</td>
      <td>18.643248</td>
      <td>0.000016</td>
      <td>0.216049</td>
      <td>0.051382</td>
      <td>0.017933</td>
      <td>24</td>
      <td>0.0</td>
      <td>[[1, 833], [23, 811]]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Truffle Type</td>
      <td>Jelly Filled</td>
      <td>18.643248</td>
      <td>0.000016</td>
      <td>0.216049</td>
      <td>0.051382</td>
      <td>0.017933</td>
      <td>24</td>
      <td>0.0</td>
      <td>[[1, 833], [23, 811]]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Primary Flavor</td>
      <td>Orange</td>
      <td>18.643248</td>
      <td>0.000016</td>
      <td>0.216049</td>
      <td>0.051382</td>
      <td>0.017933</td>
      <td>24</td>
      <td>0.0</td>
      <td>[[1, 833], [23, 811]]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Primary Flavor</td>
      <td>Toasted Coconut</td>
      <td>15.261253</td>
      <td>0.000094</td>
      <td>0.216049</td>
      <td>0.037002</td>
      <td>0.028392</td>
      <td>24</td>
      <td>0.0</td>
      <td>[[2, 832], [22, 812]]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Primary Flavor</td>
      <td>Wild Cherry Cream</td>
      <td>6.798913</td>
      <td>0.009121</td>
      <td>0.216049</td>
      <td>0.047647</td>
      <td>0.028695</td>
      <td>12</td>
      <td>0.00038</td>
      <td>[[1, 833], [11, 823]]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Secondary Flavor</td>
      <td>Apricot</td>
      <td>15.261253</td>
      <td>0.000094</td>
      <td>0.216049</td>
      <td>0.060312</td>
      <td>0.037422</td>
      <td>24</td>
      <td>0.0</td>
      <td>[[2, 832], [22, 812]]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Primary Flavor</td>
      <td>Kettle Corn</td>
      <td>29.062065</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.055452</td>
      <td>0.045891</td>
      <td>60</td>
      <td>0.0</td>
      <td>[[9, 825], [51, 783]]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Primary Flavor</td>
      <td>Acai Berry</td>
      <td>18.643248</td>
      <td>0.000016</td>
      <td>0.216049</td>
      <td>0.036505</td>
      <td>0.049466</td>
      <td>24</td>
      <td>0.0</td>
      <td>[[1, 833], [23, 811]]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Primary Flavor</td>
      <td>Pink Lemonade</td>
      <td>10.156401</td>
      <td>0.001438</td>
      <td>0.216049</td>
      <td>0.039862</td>
      <td>0.056349</td>
      <td>12</td>
      <td>0.000011</td>
      <td>[[0, 834], [12, 822]]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Secondary Flavor</td>
      <td>Black Cherry</td>
      <td>58.900366</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.055975</td>
      <td>0.062898</td>
      <td>96</td>
      <td>0.0</td>
      <td>[[11, 823], [85, 749]]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Primary Flavor</td>
      <td>Watermelon</td>
      <td>15.261253</td>
      <td>0.000094</td>
      <td>0.216049</td>
      <td>0.04405</td>
      <td>0.067896</td>
      <td>24</td>
      <td>0.0</td>
      <td>[[2, 832], [22, 812]]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Primary Flavor</td>
      <td>Sassafras</td>
      <td>6.798913</td>
      <td>0.009121</td>
      <td>0.216049</td>
      <td>0.072978</td>
      <td>0.074112</td>
      <td>12</td>
      <td>0.000059</td>
      <td>[[1, 833], [11, 823]]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Primary Flavor</td>
      <td>Plum</td>
      <td>34.851608</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.084963</td>
      <td>0.079993</td>
      <td>72</td>
      <td>0.0</td>
      <td>[[11, 823], [61, 773]]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Secondary Flavor</td>
      <td>Dill Pickle</td>
      <td>10.156401</td>
      <td>0.001438</td>
      <td>0.216049</td>
      <td>0.037042</td>
      <td>0.082494</td>
      <td>12</td>
      <td>0.000007</td>
      <td>[[0, 834], [12, 822]]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Primary Flavor</td>
      <td>Horchata</td>
      <td>10.156401</td>
      <td>0.001438</td>
      <td>0.216049</td>
      <td>0.037042</td>
      <td>0.082494</td>
      <td>12</td>
      <td>0.000007</td>
      <td>[[0, 834], [12, 822]]</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Primary Flavor</td>
      <td>Lemon Custard</td>
      <td>12.217457</td>
      <td>0.000473</td>
      <td>0.216049</td>
      <td>0.079389</td>
      <td>0.087969</td>
      <td>24</td>
      <td>0.000006</td>
      <td>[[3, 831], [21, 813]]</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Primary Flavor</td>
      <td>Fruit Punch</td>
      <td>10.156401</td>
      <td>0.001438</td>
      <td>0.216049</td>
      <td>0.078935</td>
      <td>0.090326</td>
      <td>12</td>
      <td>0.000076</td>
      <td>[[0, 834], [12, 822]]</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Base Cake</td>
      <td>Chiffon</td>
      <td>117.046226</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.127851</td>
      <td>0.125775</td>
      <td>288</td>
      <td>0.0</td>
      <td>[[60, 774], [228, 606]]</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Base Cake</td>
      <td>Butter</td>
      <td>134.36727</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.142082</td>
      <td>0.139756</td>
      <td>456</td>
      <td>0.0</td>
      <td>[[122, 712], [334, 500]]</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Secondary Flavor</td>
      <td>Banana</td>
      <td>10.805348</td>
      <td>0.001012</td>
      <td>0.216049</td>
      <td>0.163442</td>
      <td>0.15537</td>
      <td>60</td>
      <td>0.0</td>
      <td>[[17, 817], [43, 791]]</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Primary Flavor</td>
      <td>Cream Soda</td>
      <td>9.511861</td>
      <td>0.002041</td>
      <td>0.216049</td>
      <td>0.150265</td>
      <td>0.163455</td>
      <td>24</td>
      <td>0.000002</td>
      <td>[[4, 830], [20, 814]]</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Secondary Flavor</td>
      <td>Peppermint</td>
      <td>9.511861</td>
      <td>0.002041</td>
      <td>0.216049</td>
      <td>0.150265</td>
      <td>0.163455</td>
      <td>24</td>
      <td>0.000002</td>
      <td>[[4, 830], [20, 814]]</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Primary Flavor</td>
      <td>Grand Mariner</td>
      <td>10.581767</td>
      <td>0.001142</td>
      <td>0.216049</td>
      <td>0.197463</td>
      <td>0.165529</td>
      <td>72</td>
      <td>0.000829</td>
      <td>[[22, 812], [50, 784]]</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Color Group</td>
      <td>Amethyst</td>
      <td>20.488275</td>
      <td>0.000006</td>
      <td>0.216049</td>
      <td>0.195681</td>
      <td>0.167321</td>
      <td>300</td>
      <td>0.0</td>
      <td>[[114, 720], [186, 648]]</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Color Group</td>
      <td>Burgundy</td>
      <td>10.999677</td>
      <td>0.000911</td>
      <td>0.216049</td>
      <td>0.193048</td>
      <td>0.171465</td>
      <td>120</td>
      <td>0.000406</td>
      <td>[[42, 792], [78, 756]]</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Color Group</td>
      <td>White</td>
      <td>35.76526</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.19</td>
      <td>0.177264</td>
      <td>432</td>
      <td>0.0</td>
      <td>[[162, 672], [270, 564]]</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Primary Flavor</td>
      <td>Ginger Lime</td>
      <td>7.835047</td>
      <td>0.005124</td>
      <td>0.216049</td>
      <td>0.21435</td>
      <td>0.183543</td>
      <td>84</td>
      <td>0.001323</td>
      <td>[[29, 805], [55, 779]]</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Primary Flavor</td>
      <td>Mango</td>
      <td>11.262488</td>
      <td>0.000791</td>
      <td>0.216049</td>
      <td>0.28803</td>
      <td>0.245049</td>
      <td>132</td>
      <td>0.009688</td>
      <td>[[85, 749], [47, 787]]</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Color Group</td>
      <td>Opal</td>
      <td>11.587164</td>
      <td>0.000664</td>
      <td>0.216049</td>
      <td>0.317878</td>
      <td>0.259304</td>
      <td>324</td>
      <td>0.0</td>
      <td>[[190, 644], [134, 700]]</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Secondary Flavor</td>
      <td>Apple</td>
      <td>27.283292</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.326167</td>
      <td>0.293876</td>
      <td>36</td>
      <td>0.001176</td>
      <td>[[34, 800], [2, 832]]</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Secondary Flavor</td>
      <td>Tangerine</td>
      <td>32.626389</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.342314</td>
      <td>0.319273</td>
      <td>48</td>
      <td>0.000113</td>
      <td>[[44, 790], [4, 830]]</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Secondary Flavor</td>
      <td>Black Currant</td>
      <td>34.778391</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.357916</td>
      <td>0.332449</td>
      <td>36</td>
      <td>0.0</td>
      <td>[[36, 798], [0, 834]]</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Secondary Flavor</td>
      <td>Pear</td>
      <td>16.614303</td>
      <td>0.000046</td>
      <td>0.216049</td>
      <td>0.373034</td>
      <td>0.33831</td>
      <td>60</td>
      <td>0.000031</td>
      <td>[[46, 788], [14, 820]]</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Primary Flavor</td>
      <td>Vanilla</td>
      <td>34.778391</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.378053</td>
      <td>0.341626</td>
      <td>36</td>
      <td>0.000001</td>
      <td>[[36, 798], [0, 834]]</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Color Group</td>
      <td>Citrine</td>
      <td>10.156401</td>
      <td>0.001438</td>
      <td>0.216049</td>
      <td>0.390728</td>
      <td>0.342512</td>
      <td>12</td>
      <td>0.001925</td>
      <td>[[12, 822], [0, 834]]</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Color Group</td>
      <td>Teal</td>
      <td>13.539679</td>
      <td>0.000234</td>
      <td>0.216049</td>
      <td>0.323955</td>
      <td>0.3446</td>
      <td>96</td>
      <td>0.00121</td>
      <td>[[66, 768], [30, 804]]</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Base Cake</td>
      <td>Tiramisu</td>
      <td>52.360619</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.388267</td>
      <td>0.362102</td>
      <td>144</td>
      <td>0.0</td>
      <td>[[114, 720], [30, 804]]</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Primary Flavor</td>
      <td>Doughnut</td>
      <td>74.935256</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.439721</td>
      <td>0.379361</td>
      <td>108</td>
      <td>0.0</td>
      <td>[[98, 736], [10, 824]]</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Secondary Flavor</td>
      <td>Ginger Beer</td>
      <td>22.363443</td>
      <td>0.000002</td>
      <td>0.216049</td>
      <td>0.444895</td>
      <td>0.382283</td>
      <td>24</td>
      <td>0.000481</td>
      <td>[[24, 810], [0, 834]]</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Color Group</td>
      <td>Rose</td>
      <td>18.643248</td>
      <td>0.000016</td>
      <td>0.216049</td>
      <td>0.42301</td>
      <td>0.407061</td>
      <td>24</td>
      <td>0.000062</td>
      <td>[[23, 811], [1, 833]]</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Base Cake</td>
      <td>Cheese</td>
      <td>66.804744</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.450934</td>
      <td>0.435638</td>
      <td>84</td>
      <td>0.0</td>
      <td>[[79, 755], [5, 829]]</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Primary Flavor</td>
      <td>Butter Toffee</td>
      <td>60.181468</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.50366</td>
      <td>0.456343</td>
      <td>60</td>
      <td>0.0</td>
      <td>[[60, 774], [0, 834]]</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Color Group</td>
      <td>Slate</td>
      <td>10.156401</td>
      <td>0.001438</td>
      <td>0.216049</td>
      <td>0.540214</td>
      <td>0.483138</td>
      <td>12</td>
      <td>0.000017</td>
      <td>[[12, 822], [0, 834]]</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Primary Flavor</td>
      <td>Gingersnap</td>
      <td>22.363443</td>
      <td>0.000002</td>
      <td>0.216049</td>
      <td>0.643218</td>
      <td>0.623627</td>
      <td>24</td>
      <td>0.0</td>
      <td>[[24, 810], [0, 834]]</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Primary Flavor</td>
      <td>Dill Pickle</td>
      <td>22.363443</td>
      <td>0.000002</td>
      <td>0.216049</td>
      <td>0.642239</td>
      <td>0.655779</td>
      <td>24</td>
      <td>0.0</td>
      <td>[[24, 810], [0, 834]]</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Color Group</td>
      <td>Olive</td>
      <td>44.967537</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.637627</td>
      <td>0.670186</td>
      <td>60</td>
      <td>0.0</td>
      <td>[[56, 778], [4, 830]]</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Primary Flavor</td>
      <td>Butter Milk</td>
      <td>10.156401</td>
      <td>0.001438</td>
      <td>0.216049</td>
      <td>0.699284</td>
      <td>0.688601</td>
      <td>12</td>
      <td>0.0</td>
      <td>[[12, 822], [0, 834]]</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Base Cake</td>
      <td>Sponge</td>
      <td>127.156266</td>
      <td>0.0</td>
      <td>0.216049</td>
      <td>0.698996</td>
      <td>0.699355</td>
      <td>120</td>
      <td>0.0</td>
      <td>[[120, 714], [0, 834]]</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Primary Flavor</td>
      <td>Chocolate Mint</td>
      <td>10.156401</td>
      <td>0.001438</td>
      <td>0.216049</td>
      <td>0.685546</td>
      <td>0.699666</td>
      <td>12</td>
      <td>0.0</td>
      <td>[[12, 822], [0, 834]]</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Primary Flavor</td>
      <td>Coconut</td>
      <td>10.156401</td>
      <td>0.001438</td>
      <td>0.216049</td>
      <td>0.732777</td>
      <td>0.717641</td>
      <td>12</td>
      <td>0.0</td>
      <td>[[12, 822], [0, 834]]</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Primary Flavor</td>
      <td>Blueberry</td>
      <td>22.363443</td>
      <td>0.000002</td>
      <td>0.216049</td>
      <td>0.759643</td>
      <td>0.72536</td>
      <td>24</td>
      <td>0.0</td>
      <td>[[24, 810], [0, 834]]</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Primary Flavor</td>
      <td>Amaretto</td>
      <td>10.156401</td>
      <td>0.001438</td>
      <td>0.216049</td>
      <td>0.782156</td>
      <td>0.764845</td>
      <td>12</td>
      <td>0.0</td>
      <td>[[12, 822], [0, 834]]</td>
    </tr>
  </tbody>
</table>
</div>



### 🍒🍒 2.2.2 **Enrichment**: Broad Analysis of Categories: ANOVA



Recall our "melted" shift data. It will be useful to think of getting our Truffle data in this format:


```python
shift_melt.head()
```





  <div id="df-d48631b7-b109-4627-aa30-b85e20e0f05c">
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
    </div>
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





  <div id="df-6482d5af-de98-431a-99f6-9afefa83eae3">
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
    </div>
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




    <pandas.core.groupby.generic.SeriesGroupBy object at 0x7fafac7cfd10>




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




  <div id="df-7ab530db-b767-4222-a1eb-b8d9494d63c2">
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
    </div>
  </div>



    Shapiro:  0.9634131193161011 4.1681337029688696e-20
    Bartlett:  69.83288886114195 1.1102218566053728e-13
    
    Truffle_Type




  <div id="df-925cf159-35c7-459a-b8ef-99d535050c04">
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
    </div>
  </div>



    Shapiro:  0.9576056599617004 1.2598073820281984e-21
    Bartlett:  109.93252546442552 1.344173733366234e-24
    
    Primary_Flavor




  <div id="df-940e5512-3a8a-463f-a5a8-a221bee351c5">
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
    </div>
  </div>



    Shapiro:  0.948470413684845 9.90281706784179e-24
    Bartlett:  210.15130419114894 1.5872504991231547e-21
    
    Secondary_Flavor




  <div id="df-c4875fd1-47ab-4223-92f4-82173a73e7e4">
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
    </div>
  </div>



    Shapiro:  0.9548103213310242 2.649492974953278e-22
    Bartlett:  420.6274502894803 1.23730070350945e-71
    
    Color_Group




  <div id="df-b3ea2010-fdc1-49d7-b22e-6e115f99a217">
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
    </div>
  </div>



    Shapiro:  0.969061017036438 1.8926407335144587e-18
    Bartlett:  136.55525281340468 8.164787784033709e-24
    


### 🍒🍒 2.2.3 **Enrichment**: Visual Analysis of Residuals: QQ-Plots

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


    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_102_0.png)
    



    
![png](S2_Inferential_Statistics_files/S2_Inferential_Statistics_102_1.png)
    


We see that a lot of our data is swayed by extremely high and low values, so what can we conclude? 

> You need the right test statistic for the right job, in this case, we are littered with unequal variance in our groupings so we use the moods median and welch (unequal variance t-test) to make conclusions about our data


# References

* [Renesh Bedre ANOVA](https://www.reneshbedre.com/blog/anova.html)
* [Minitab ANOVA](https://blog.minitab.com/en/adventures-in-statistics-2/understanding-analysis-of-variance-anova-and-the-f-test)
* [Analytics Vidhya ANOVA](https://www.analyticsvidhya.com/blog/2020/06/introduction-anova-statistics-data-science-covid-python/)
* [Renesh Bedre Hypothesis Testing](https://www.reneshbedre.com/blog/hypothesis-testing.html)
* [Real Statistics Turkey-kramer](https://www.real-statistics.com/one-way-analysis-of-variance-anova/unplanned-comparisons/tukey-kramer-test/)
* [Mutual Information](https://www.kaggle.com/ryanholbrook/mutual-information)
