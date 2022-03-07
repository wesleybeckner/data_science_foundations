<a href="https://colab.research.google.com/github/wesleybeckner/data_science_foundations/blob/main/notebooks/solutions/SOLN_L5_Writing_Unit_Tests.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Data Science Foundations <br> Lab 5: Writing Unit Tests

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com
<br>

---

<br>

In this lab, we will try our hand at writing unit tests

<br>

---

## Import Libraries


```python
import random
import numpy as np
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
```

## Types of Tests

There are two main types of tests we want to distinguish:
* **_Unit test_**: an automatic test to test the internal workings of a class or function. It should be a stand-alone test which is not related to other resources.
* **_Integration test_**: an automatic test that is done on an environment, it tests the coordination of different classes and functions as well as with the running environment. This usually precedes sending code to a QA team.

To this I will add:

* **_Acid test_**: extremely rigorous tests that push beyond the intended use cases for your classes/functions. Written when you, like me, cannot afford QA employees to actually test your code. (word origin: [gold acid tests in the 1850s](https://en.wikipedia.org/wiki/Acid_test_(gold)), [acid tests in the 70's](https://en.wikipedia.org/wiki/Acid_Tests))
* **_EDIT_**: you could also call this a corner, or an edge case

In this lab we will focus on _unit tests_.

## Unit Tests

Each unit test should test the smallest portion of your code possible, i.e. a single method or function. Any random number generators should be seeded so that they run the exact same way every time. Unit tests should not rely on any local files or the local environment. 

Why bother with Unit Tests when we have Integration tests?

A major challenge with integration testing is when an integration test fails. It’s very hard to diagnose a system issue without being able to isolate which part of the system is failing. Here comes the unit test to the rescue. 

Let's take a simple example. If I wanted to test that the sume of two numbers is correct


```python
assert sum([2, 5]) == 7, "should be 7"
```

Nothing is sent to the print out because the condition is satisfied. If we run, however:

```
assert sum([2, 4]) == 7, "should be 7"
```

we get an error message:

```
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
<ipython-input-3-d5724b127818> in <module>()
----> 1 assert sum([2, 4]) == 7, "should be 7"

AssertionError: should be 7
```


To make this a Unit Test, you will want to wrap it in a function


```python
def test_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"

test_sum()
print("Everything passed")
```

    Everything passed


And if we include a test that does not pass:

```
def test_sum():
  assert sum([3, 3]) == 6, "Should be 6"

def test_my_broken_func():
  assert sum([1, 2]) == 5, "Should be 5"

test_sum()
test_my_broken_func()
print("Everything passed")
```



Here our test fails, because the sum of 1 and 2 is 3 and not 5. We get a traceback that tells us the source of the error:

```
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
<ipython-input-13-8a552fbf52bd> in <module>()
      6 
      7 test_sum()
----> 8 test_my_broken_func()
      9 print("Everything passed")

<ipython-input-13-8a552fbf52bd> in test_my_broken_func()
      3 
      4 def test_my_broken_func():
----> 5   assert sum([1, 2]) == 5, "Should be 5"
      6 
      7 test_sum()

AssertionError: Should be 5
```



Before sending us on our merry way to practice writing unit tests, we will want to ask, what do I want to write a test about? Here, we've been testing sum(). There are many behaviors in sum() we could check, such as:

* Does it sum a list of whole numbers (integers)?
* Can it sum a tuple or set?
* Can it sum a list of floats?
* What happens if one of the numbers is negative? etc..

In the end, what you test is up to you, and depends on your intended use cases. As a general rule of thumb, your unit test should test what is relevant.

The only caveat to that, is that many continuous integration services (like [TravisCI](https://travis-ci.com/)) will benchmark you based on the percentage of lines of code you have that are covered by your unit tests (ex: [85% coverage](https://github.com/wesleybeckner/gains)).

## ✍🏽 Q1 Write a Unit Test

Remember our Pokeball discussion in [Python Foundations](https://wesleybeckner.github.io/python_foundations/S4_Object_Oriented_Programming/)? We'll return to that here. This time writing unit tests for our classes.

Sometimes when writing unit tests, it can be more complicated than checking the return value of a function. Think back on our pokemon example:

<br>

<p align=center>
<img src="https://raw.githubusercontent.com/wesleybeckner/python_foundations/main/assets/pokeballs.png"></img>
</p>

```
class Pokeball:
  def __init__(self, contains=None, type_name="poke ball"):
    self.contains = contains
    self.type_name = type_name
    self.catch_rate = 0.50 # note this attribute is not accessible upon init

  # the method catch, will update self.contains, if a catch is successful
  # it will also use self.catch_rate to set the performance of the catch
  def catch(self, pokemon):
    if self.contains == None:
      if random.random() < self.catch_rate:
        self.contains = pokemon
        print(f"{pokemon} captured!")
      else:
        print(f"{pokemon} escaped!")
        pass
    else:
      print("pokeball is not empty!")
      
  def release(self):
    if self.contains == None:
      print("Pokeball is already empty")
    else:
      print(self.contains, "has been released")
      self.contains = None
```

If I wanted to write a unit test for the release method, I couldn't directly check for the output of a function. I'll have to check for a **_side effect_**, in this case, the change of an attribute belonging to a pokeball object; that is the change to the attribute _contains_.




```python
class Pokeball:
  def __init__(self, contains=None, type_name="poke ball"):
    self.contains = contains
    self.type_name = type_name
    self.catch_rate = 0.50 # note this attribute is not accessible upon init

  # the method catch, will update self.contains, if a catch is successful
  # it will also use self.catch_rate to set the performance of the catch
  def catch(self, pokemon):
    if self.contains == None:
      if random.random() < self.catch_rate:
        self.contains = pokemon
        print(f"{pokemon} captured!")
      else:
        print(f"{pokemon} escaped!")
        pass
    else:
      print("pokeball is not empty!")
      
  def release(self):
    if self.contains == None:
      print("Pokeball is already empty")
    else:
      print(self.contains, "has been released")
      self.contains = None
```

In the following cell, finish the code to test the functionality of the _release_ method:


```python
def test_release():
  ball = Pokeball()
  ball.contains = 'Pikachu'
  ball.release()
  # turn the pseudo code below into an assert statement
  
  ### YOUR CODE HERE ###
  assert ball.contains == None, "ball is not empty!"
```


```python
test_release()
```

    Pikachu has been released


## ⛹️ Q2 Write a Unit Test for the Catch Rate

First, we will check that the succcessful catch is operating correctly. Remember that we depend on `random.random` and condition our success on whether that random value is less than the `catch_rate` of the pokeball:

```
if self.contains == None:
      if random.random() < self.catch_rate:
        self.contains = pokemon
```

so to test whether the successful catch is working we will seed our random number generator with a value that returns less than the `catch_rate` of the pokeball and then write our assert statement:



```python
def test_successful_catch():
  # choose a random seed such that
  # we know the catch call should succeed
  
  ### YOUR CODE BELOW ###
  random.seed(1)
  ball = Pokeball()
  ball.catch('Psyduck') # Someone's fave pokemon (bless 'em)

  ### YOUR CODE BELOW ###
  assert ball.contains == 'Psyduck', "ball did not catch as expected"
```

NICE. Now we will do the same thing again, this time testing for an unsuccessful catch. SO in order to do this, we need to choose a random seed that will cause our catch to fail:


```python
def test_unsuccessful_catch():
  # choose a random seed such that
  # we know the catch call should FAIL
  
  ### YOUR CODE BELOW ###
  random.seed(0)
  ball = Pokeball()
  ball.catch('Psyduck') 

  ### YOUR CODE BELOW ###
  assert ball.contains == None, "ball did not fail as expected"
```

When you are finished test your functions below


```python
test_unsuccessful_catch()
```

    Psyduck escaped!



```python
test_successful_catch()
```

    Psyduck captured!


## ⚖️ Q3 Write a Unit Test that Checks Whether the Overall Catch Rate is 50/50

For this one, we're going to take those same ideas around seeding the random number generator. However, here we'd like to run the catch function multiple times to check whether it is truly creating a 50/50 catch rate situation.

Here's a pseudo code outline:

1. seed the random number generator
2. for 100 iterations: 
    * create a pokeball
    * try to catch something
    * log whether it was successful
3. check that for the 100 attempts the success was approximately 50/50

_note:_ you can use my `suppress stdout()` function to suppress the print statements from `ball.catch`

ex:

```
with suppress_stdout():
  print("HELLO OUT THERE!")
```

---

_quick segway_: what is the actual behavior of `random.seed()`? Does it produce the same number every time we call `random.random()` now? Check for yourself:


```python
random.seed(42)
[random.random() for i in range(5)]
```




    [0.6394267984578837,
     0.025010755222666936,
     0.27502931836911926,
     0.22321073814882275,
     0.7364712141640124]



We see that it still produces random numbers with each call to `random.random`. However, those numbers are the same with every execution of the cell. What happens when we do this:


```python
[random.random() for i in range(5)]
```




    [0.5449414806032167,
     0.2204406220406967,
     0.5892656838759087,
     0.8094304566778266,
     0.006498759678061017]



The numbers are different. BUT:


```python
random.seed(42)
[random.random() for i in range(10)]
```




    [0.6394267984578837,
     0.025010755222666936,
     0.27502931836911926,
     0.22321073814882275,
     0.7364712141640124,
     0.6766994874229113,
     0.8921795677048454,
     0.08693883262941615,
     0.4219218196852704,
     0.029797219438070344]



We see them here in the bottom half of the list again. So, random.seed() is _seeding_ the random number generator such that it will produce the same sequence of random numbers every time from the given seed. This will reset whenever random.seed() is set again. This behavior is useful because it allows us to continue using random number generation in our code, (for testing, creating examples and demos, etc.) but it will be reproducable each time.

_End Segway_

---


```python
# 1. seed the random number generator
# 2. for 100 iterations: 
#     * create a pokeball
#     * try to catch something
#     * log whether it was successful
# 3. check that for the 100 attempts the success was approximately 50/50
def test_catch_rate():
  ### YOUR CODE HERE ###
  results = 0
  random.seed(42)
  for i in range(100):
    ball = Pokeball()
    with suppress_stdout():
      ball.catch("Charzard")
    if ball.contains != None:
      results += 1
  results = results/100
  
  ### END YOUR CODE ###
  assert np.abs(np.mean(results) - 0.5) < 0.1, "catch rate not 50/50"
test_catch_rate()
```

## Test Runners

When we start to create many tests like this, it can be cumbersome to run them all at once and log which ones fail. To handle our unit tests we use what are called **_test runners_**. We won't dedicate time to any single one here but the three most common are:

* unittest
* nose2
* pytest

unittest is built into python. I don't like it because you have to follow a strict class/method structure when writing the tests. nose2 is popular with many useful features and is generally good for high volumes of tests. My favorite is pytest, it's flexible and has an ecosystem of plugins for extensibility. 
