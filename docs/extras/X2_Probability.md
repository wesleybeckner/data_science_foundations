## Measuring Uncertainty

One strategy is to just count occurance of outcomes

The probability of either of two mutually exclusive events occuring is the sum of their probabilities

$$ P(A \space or \space B) = P(A \cup B) = P(A) + P(B) $$

The probability of two mutually exclusive events occuring together is the product of their probabilities

$$ P(A \space and \space B) = P(A \cap B) = P(A) * P(B) $$

For non-mutually exclusive events:

$$ P (A \cup B) = P(A) + P(B) - P(A \cap B) $$

### Q1

In a single toss of 2 fair (evenly-weighted) six-sided dice, find the probability that their sum will be at most 9


```python
tot_outcomes = 6**2
sum_less_than_9 = 6 + 6 + 6 + 5 + 4 + 3
sum_less_than_9/tot_outcomes
```




    0.8333333333333334



### Q2

In a single toss of 2 fair (evenly-weighted) six-sided dice, find the probability that the values rolled by each die will be different and the two dice have a sum of 6.


```python
# only 5 outcomes will sum to 6
# one of those has equal numbers
# so there are 4/36 chances or 1/9 probability
```

### Q3

There are 3 urns labeled X, Y, and Z.


* Urn X contains 4 red balls and 3 black balls.
* Urn Y contains 5 red balls and 4 black balls.
* Urn Z contains 4 red balls and 4 black balls.

One ball is drawn from each of the 3 urns. What is the probability that, of the 3 balls drawn, 2 are red and 1 is black?


```python
# multiply and sum probabilities 
# RRB 4/7 * 5/9 * 4/8
# RBR 4/7 * 4/9 * 4/8
# BRR 3/7 * 5/9 * 4/8
(4/7 * 5/9 * 1/2) +\
(4/7 * 4/9 * 1/2) +\
(3/7 * 5/9 * 1/2)
```




    0.40476190476190477



## Conditional Probability

The flagship expression here is Bayes Rule or Bayesian Inference:

$$ P(A|B) = \frac{P(B|A) * P(A)}{P(B)} = \frac{P(A \cap B)}{P(B)}$$

Where \\(\cap\\) is the intersection of \\(A\\) and \\(B\\).

### Q1

Suppose a family has 2 children, one of which is a boy. What is the probability that both children are boys?


```python
# child1 child2
# a boy; a girl
# a girl; a boy
# a boy; a boy
```

### Q2

You draw 2 cards from a standard 52-card deck without replacing them. What is the probability that both cards are of the same suit?


```python
# suites
# 13 13 13 13
# hearts
(13-1)/51 #12/51
# spades 4/17
# clubs 4/17
# diamonds 4/17
# (4*4)/(4*17)
# multiplying out still yields
# 12/51
```




    0.9411764705882353



### Q3

If the probability of student A passing an exam is 2/7 and the probability of student B failing the exam is 3/7, then find the probability that at least 1 of the 2 students will pass the exam


```python
# P(A) = 2/7
# P(B) = 4/7

# All outcomes - sum to 1
# A pass B pass 2/7 * 4/7
# A fail B fail 5/7 * 3/7
# A pass B fail 2/7 * 3/7
# A fail B pass 5/7 * 4/7

# all outcomes
(2/7)*(4/7)+\
(5/7)*(3/7)+\
(2/7)*(3/7)+\
(5/7)*(4/7)

# outcomes we care about
(2/7)*(4/7)+\
(2/7)*(3/7)+\
(5/7)*(4/7)

34/49
```




    0.6938775510204082



### Q4

Historical data shows that it has only rained 5 days per year in some desert region (assuming a 365 day year). A meteorologist predicts that it will rain today. When it actually rains, the meteorologist correctly predicts rain 90% of the time. When it doesn't rain, the meteorologist incorrectly predicts rain 10% of the time. Find the probability that it will rain today.


```python
# P(A|B) = probability that it will rain today given that the meteorologist has predicted it will rain
# P(B|A) = probability that the meteoroligist will say it will rain when it rains; 90%
# P(A) = probability that it will rain; 5/365
# P(B) = probability that meteoroligist will say it will rain

# what is P(B) then?

# P(B) = (5/365*.90) + ((365-5)/365*.1)

P_B = (5/365*.90) + ((365-5)/365*.1)
P_A = 5/365
P_BA = 0.9

P_AB = P_BA * P_A / P_B

print(f"P(B|A): {P_BA}")
print(f"P(B): {P_B}")
print(f"P(A): {P_A}")
print(f"P(A|B): {P_AB}")
```

    P(B|A): 0.9
    P(B): 0.11095890410958904
    P(A): 0.0136986301369863
    P(A|B): 0.1111111111111111


## Binomial Probabilities

> Operates on PMF (Probability Mass Functions) for discrete values

[answer key](https://nostarch.com/download/resources/Bayes_exercise_solutions_new.pdf)

$$ B(K;n,p) = \binom{n}{k} \times p^k \times (1 - p)^{n-k} $$

We can calculate the total number of outcomes we care about from a total number of trials  using the binomial coefficient (this field of study is called combinatorics): 

$$ \binom{n}{k} = \frac{n!}{k! \times (n - k)!}$$

This allows us to calculate the probability of an event:

$$ B(K;n,p) = \binom{n}{k} \times P(desired \space outcome) $$


```python
def fact(x):
  """
  return the factorial of a number using recursion 
  """
  if x == 1 or x == 0:
    return 1
  else:
    return fact(x-1) * x

def n_choose_k(n, k):
  """
  Returns the number of outcomes we care about of all possible outcomes
  """
  return fact(n) / (fact(k) * fact(n - k))

def binom(n, k, p):
  """
  Returns the probability of an event occuring K times in a total number of n
  trials having a probability of p
  """
  return n_choose_k(n, k) * p**k * (1-p) ** (n-k)

def k_or_more(n, k, p):
  """
  we can solve the K or more problem recursively
  """
  if k == n:
    return binom(n, k, p)
  else:
    return k_or_more(n, k+1, p) + binom(n, k, p)
```

### Q1

When you're searching for a new job, it's always helpful to have more than one offer on the table so you can use it in negotiations. If you have 1/5 probability of receiving a job offer when you interview, and you interview iwth seven companies in a month, what is the probability you'll have at least two competing offers by the end of that month?


```python
p = 1/5
n = 7
k = 2

offers1 = k_or_more(n, k, p)
print(offers1)
```

    0.4232832000000002


### Q2

You get a bunch of recruiter emails and find out you have 25 interviews lined up in the next month. Unfortunately, you know this will leave you exhausted, and the probability of getting an offer will drop to 1/10 if you're tired. You really don't want to go on this many interviews unless you are at least twice as likely to get a least two competing offers. Are you more likely to get at least two offers if you go for 25 interviews, or stick to just 7?


```python
p = 1/10
n = 25
k = 2

offers2 = k_or_more(n, k, p)
print(offers2)

print(offers2/offers1)
```

    0.7287940935386341
    1.7217647512082543


The ratio of boys to girls for babies born in Russia is 1.09:1. If there is 1 child born per birth, what proportion of Russian families with exactly 6 children will have at least 3 boys?


```python
br, gr = 1.09, 1
p = br / (br + gr)
n = 6
k = 3

k_or_more(n, k, p)
```




    0.6957033161509107



## The Beta Distribution

> Operates on PDF (Probability Density Function) for continuous values

Think: Probability of probabilities

$$ Beta(\rho; \alpha, \beta) = \frac{\rho^{\alpha - 1} \times (1-\rho)^{\beta - 1}}{beta(\alpha, \beta)} $$

where \\(\rho\\) is the probability of an event. This corresponds to the different hypotheses for the possible probabilities that could be generating our observed data; \\(\alpha\\) represents how many times we observe an event we care about such as winning a coin toss; \\(\beta\\) represents how many times the event we care about _didn't_ happen, such as losing a coin toss. The total number of trials is \\(\alpha + \beta\\) (contrast this with \\(n\\) and \\(k\\) in the binomial distribution).

The beta (lowercase) distribution:

$$ \int_0^1{\rho^{\alpha - 1} \times (1-\rho)^{\beta - 1}} $$

Putting this all together. The probability that an event occurs in a specific range:

$$ Beta(\rho; \alpha, \beta) = \int_{lower \space bound}^{upper \space bound}{\frac{\rho^{\alpha - 1} \times (1-\rho)^{\beta - 1}}{beta(\alpha, \beta)}} $$

### Q1

You want to use the beta distribution to determine whether or not a coin you have is a fair coin - meaning that the coin gives you heads and tails equally. You flip the coin 10 times and get 4 heads and 6 tails. using the beta distribution, what is the probability that the coin will land on heads more than 60 percent of the time?


```python
from scipy.stats import beta

_alpha = 4
_beta = 6

model = beta(_alpha, _beta)
model.pdf(0.6)
```




    1.1147673600000005


