<a href="https://colab.research.google.com/github/wesleybeckner/data_science_foundations/blob/main/notebooks/exercises/E5_Writing_Unit_Tests.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Data Science Foundations, Lab 5: Writing Unit Tests

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com
<br>

---

<br>

We will try our hand at writing unit tests

<br>

---

## Import Libraries


```python
# for numpy section
import numpy as np
np.random.seed(42)
```


```python
# for debugging section
import random
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

### 🏋️ Exercise 1: Debug a Class Method

Your friend is developing a new pokemon game. They are excited to release but are running into some trouble! 


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


class Pokemon():
  def __init__(self, name, weight, speed, type_):
    self.name = name
    self.weight = weight
    self.speed = speed
    self.type_ = type_

class FastBall(Pokeball):
  def __init__(self, contains=None, type_name="Fastball"):
    Pokeball.__init__(self, contains, type_name)
    self.catch_rate = 0.6

  def catch_fast(self, pokemon):
    if pokemon.speed > 100:
      if self.contains == None:
        self.contains = pokemon
        print(pokemon.name, "has been captured")
      else:
        print("Pokeball is not empty")
    else:
      self.catch(pokemon)

```

They're concerned that the object `FastBall` doesn't return the pokemon's name when executing `print(fast.contains)` when they know the pokeball contains a pokemon. Help them find the bug, then write the following unit tests:

1. showing that the pokeball updates properly with the name of the pokemon after it makes a capture of a pokemon with a speed > 100
2. showing that the `catch_rate` of 0.6 is resulting in a 60% catch rate for pokemon with speeds < 100


```python
# Your friend shows you this code 
fast = FastBall()

mewtwo = Pokemon(name='Mewtwo', 
                 weight=18,
                 speed=110, 
                 type_='Psychic')

print(fast.contains)

fast.catch_fast(mewtwo)

# this is the line they are concerned about
# why does this not return MewTwo?
print(fast.contains)

fast.catch_fast(mewtwo)
```

    None
    Mewtwo has been captured
    <__main__.Pokemon object at 0x7fd4bfe612d0>
    Pokeball is not empty


# Part 2 (Optional): Use a Test Runner

Create the following files:

* `pokemon.py`
* `test_pokemon.py`

paste the following into `pokemon.py`:

```
import random
import numpy as np

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


class Pokemon():
  def __init__(self, name, weight, speed, type_):
    self.name = name
    self.weight = weight
    self.speed = speed
    self.type_ = type_

class FastBall(Pokeball):
  def __init__(self, contains=None, type_name="Fastball"):
    Pokeball.__init__(self, contains, type_name)
    self.catch_rate = 0.6

  def catch_fast(self, pokemon):
    if pokemon.speed > 100:
      if self.contains == None:
        self.contains = pokemon
        print(pokemon.name, "has been captured")
      else:
        print("Pokeball is not empty")
    else:
      self.catch(pokemon)

```

in `test_pokemon.py` paste any unit tests you've written along with the imports at the top of the file (be sure to import any other libraries you used in your unit tests as well)

```
from pokemon import *
import random
import numpy as np

### YOUR UNIT TESTS HERE ###
def test_<name_of_your_test>():
  # ....
  assert <your assert statement>
```

make sure `pokemon.py` and `test_pokemon.py` are in the same directory then run the command

```
pytest
```

from the command line. You should get a readout like the following

```
================================================= test session starts ==================================================
platform linux -- Python 3.8.1, pytest-6.2.1, py-1.10.0, pluggy-0.13.1
rootdir: /mnt/c/Users/wesley/Documents/apps/temp_c3_l2
plugins: dash-1.20.0, anyio-2.2.0
collected 1 item

test_pokemon.py .                                                                                                [100%]

================================================== 1 passed in 0.06s ===================================================
```
