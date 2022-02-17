<a href="https://colab.research.google.com/github/wesleybeckner/data_science_foundations/blob/main/notebooks/project/P3_1_Step_Look_Ahead_Agents.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Data Science Foundations <br> Project Part 3: 1-Step Look Ahead

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com

---

<br>

Today we're working on a more advanced AI structure: 1-step lookahead.

<br>

---


<a name='top'></a>



<a name='x.0'></a>

## 3.0 Preparing Environment and Importing Data

[back to top](#top)

<a name='x.0.1'></a>

### 3.0.1 Import Packages

[back to top](#top)


```python
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TicTacToe:
  # can preset winner and starting player
  def __init__(self, winner='', start_player=''): 
    self.winner = winner
    self.start_player = start_player
    self.board = {1: ' ',
         2: ' ',
         3: ' ',
         4: ' ',
         5: ' ',
         6: ' ',
         7: ' ',
         8: ' ',
         9: ' ',}
    self.win_patterns = [[1,2,3], [4,5,6], [7,8,9],
                [1,4,7], [2,5,8], [3,6,9],
                [1,5,9], [7,5,3]]
         
  # the other functions are now passed self
  def visualize_board(self):
    print(
      "|{}|{}|{}|\n|{}|{}|{}|\n|{}|{}|{}|\n".format(*self.board.values())
      )

  def check_winning(self):
    for pattern in self.win_patterns:
      values = [self.board[i] for i in pattern] 
      if values == ['X', 'X', 'X']:
        self.winner = 'X' # we update the winner status
        return "'X' Won!"
      elif values == ['O', 'O', 'O']:
        self.winner = 'O'
        return "'O' Won!"
    return ''

  def check_stalemate(self):
    if (' ' not in self.board.values()) and (self.check_winning() == ''):
      self.winner = 'Stalemate'
      return "It's a stalemate!"

class GameEngine(TicTacToe):
  def __init__(self, setup='auto'):
    super().__init__()
    self.setup = setup

  def heuristic_ai(self, player_label):
    opponent = ['X', 'O']
    opponent.remove(player_label)
    opponent = opponent[0]

    avail_moves = [i for i in self.board.keys() if self.board[i] == ' ']
    temp_board = self.board.copy()
    middle = 5
    corner = [1,3,7,9]
    side = [2,4,6,8]
    the_final_move = None
    # first check for a winning move
    move_found = False
    for move in avail_moves:
      temp_board[move] = player_label
      for pattern in self.win_patterns:
          values = [temp_board[i] for i in pattern] 
          if values == [player_label, player_label, player_label]:
            move_found = True   
            the_final_move = move
            break
      if move_found:   
        break
      else:
        temp_board[move] = ' '

    # check if the opponent has a winning move
    if move_found == False:
      for move in avail_moves:
        temp_board[move] = opponent
        for pattern in self.win_patterns:
            values = [temp_board[i] for i in pattern] 
            if values == [opponent, opponent, opponent]:
              move_found = True       
              break
        if move_found:   
          break
        else:
          temp_board[move] = ' '

    # check corners
    if move_found == False:
      move_corner = [val for val in avail_moves if val in corner]
      if len(move_corner) > 0:
        move = random.choice(move_corner)
        move_found = True
        
    # check if middle avail
    if move_found == False:
      if middle in avail_moves:
        move_found = True
        move = middle

    # check side
    if move_found == False:
      move_side = [val for val in avail_moves if val in side]
      if len(move_side) > 0:
        move = random.choice(move_side)
        move_found = True

    return move

  def random_ai(self):
    while True:
      move = random.randint(1,9)
      if self.board[move] != ' ':
        continue
      else:
        break
    return move

  def setup_game(self):

    if self.setup == 'user':
      players = int(input("How many Players? (type 0, 1, or 2)"))
      self.player_meta = {'first': {'label': 'X',
                                    'type': 'ai'}, 
                    'second': {'label': 'O',
                                    'type': 'human'}}
      if players != 2:
        ########## 
        # Allow the user to set the ai level
        ########## 
        level = int(input("select AI level (1, 2)"))
        if level == 1:
          self.ai_level = 1
        elif level == 2:
          self.ai_level = 2
        else:
          print("Unknown AI level entered, this will cause problems")

      if players == 1:
        first = input("who will go first? (X, (AI), or O (Player))")
        if first == 'O':
          self.player_meta = {'second': {'label': 'X',
                                    'type': 'ai'}, 
                        'first': {'label': 'O',
                                    'type': 'human'}}
        

        

      elif players == 0:
        first = random.choice(['X', 'O'])
        if first == 'O':
          self.player_meta = {'second': {'label': 'X',
                                    'type': 'ai'}, 
                        'first': {'label': 'O',
                                    'type': 'ai'}}                                
        else:
          self.player_meta = {'first': {'label': 'X',
                                    'type': 'ai'}, 
                        'second': {'label': 'O',
                                    'type': 'ai'}}

        

    elif self.setup == 'auto':
      first = random.choice(['X', 'O'])
      if first == 'O':
        self.start_player = 'O'
        self.player_meta = {'second': {'label': 'X',
                                  'type': 'ai'}, 
                      'first': {'label': 'O',
                                  'type': 'ai'}}                                
      else:
        self.start_player = 'X'
        self.player_meta = {'first': {'label': 'X',
                                  'type': 'ai'}, 
                      'second': {'label': 'O',
                                  'type': 'ai'}}
      ########## 
      # and automatically set the ai level otherwise
      ##########                             
      self.ai_level = 2

  def play_game(self):
    while True:
      for player in ['first', 'second']:  
        self.visualize_board()
        player_label = self.player_meta[player]['label']
        player_type = self.player_meta[player]['type']

        if player_type == 'human':
          move = input("{}, what's your move?".format(player_label))
          # we're going to allow the user to quit the game from the input line
          if move in ['q', 'quit']:
            self.winner = 'F'
            print('quiting the game')
            break

          move = int(move)
          if self.board[move] != ' ':
            while True:
              move = input("{}, that position is already taken! "\
                          "What's your move?".format(player_label))  
              move = int(move)            
              if self.board[move] != ' ':
                continue
              else:
                break

        else:
          ##########
          # Our level 1 ai agent (random)
          ##########
          if self.ai_level == 1:
            move = self.random_ai()

          ##########
          # Our level 2 ai agent (heuristic)
          ##########
          elif self.ai_level == 2:
            move = self.heuristic_ai(player_label)

        self.board[move] = player_label

        # the winner varaible will now be check within the board object
        self.check_winning()
        self.check_stalemate()

        if self.winner == '':
          continue

        elif self.winner == 'Stalemate':
          print(self.check_stalemate())
          self.visualize_board()
          break

        else:
          print(self.check_winning())
          self.visualize_board()
          break
      if self.winner != '':
        return self
```

<a name='x.0.1'></a>

### 3.0.2 Load Dataset

[back to top](#top)

## 3.1 Rethinking gameplay

To implement the broader strategies used in game theory and machine learning, we need to rebroadcast our approach to creating our AI agent. In the heurstical agent model, we thought in terms of checking for specific move types, defined by what kind of advantage they give us during game play, i.e. see if a winning move is available, a blocking move, if a corner place is free, etc. Rather than thinking with this _look and check_ mindset that is centered around specific strategies and our own prior knowledge about the game (we _know_ that a center piece is statistically likely to give us a higher chance of winning) we will evaluate every available move to the AI, and rate them quantitatively. 

> **_switching from ordinal to interval_** Notice the datatype change when we move from giving simple preferences of moves to actual scores of moves. Catalog this in your mind for future reference when considering datatypes!



### 3.1.1 One-Step Look Ahead

For now, when we rate our boards, we will only look 1-step ahead in gameplay. Hence the name we give this AI strategy, 1-step lookahead

The beginning portion of our code will look about the same as the heuristic AI model. Recall:

```
  def heuristic_ai(self, player_label):
    opponent = ['X', 'O']
    opponent.remove(player_label)
    opponent = opponent[0]

    avail_moves = [i for i in self.board.keys() if self.board[i] == ' ']
    temp_board = self.board.copy()
```
but now, instead of searching progressively through our preferred move-types (winning, middle, etc.) . We are going to give every available move (1, 3, 7, etc.) a score. Our score regimen will look like the following:

* 100 pts: winning move
* 10 pts: blocks an opponents winning move
* 1 pt: every other move

### Q1 Rewrite avail_moves

define avail_moves as a dictionary of available moves with scores for each move as empty strings. We will update this dictionary with numerical scores in the next step


```python
# we're going to steal the parameter names to
# prototype our new function
self = TicTacToe()
player_label = 'X'

opponent = ['X', 'O']
opponent.remove(player_label)
opponent = opponent[0]

# instead of a list, we want avail_moves to now be a dictionary that will
# contain the move and its score

avail_moves = {i:' ' for i in self.board if self.board[i] == ' '}

temp_board = self.board.copy()
```


```python
avail_moves
```




    {1: ' ', 2: ' ', 3: ' ', 4: ' ', 5: ' ', 6: ' ', 7: ' ', 8: ' ', 9: ' '}



### Q2 Score each move in `avail_moves`

Now let's fold this into our new `one_step_ai` function. Remember:

* 100 pts: winning move
* 10 pts: blocks an opponents winning move
* 1 pt: every other move


```python
# the beginning portion of our code will look about the same
# as the heuristic AI model
def one_step_ai(self, player_label):
  opponent = ['X', 'O']
  opponent.remove(player_label)
  opponent = opponent[0]

  ##############################################################################
  ############################# DEFINE avail_moves #############################
  ##############################################################################
  avail_moves = {i:' ' for i in self.board if self.board[i] == ' '}
  for move in avail_moves.keys():
    avail_moves[move] = 1
  temp_board = self.board.copy()

  # first check for a winning move
  # we're now looping through the keys of our dictionary
  for move in avail_moves.keys():
    temp_board[move] = player_label
    for pattern in self.win_patterns:
        values = [temp_board[i] for i in pattern] 
        if values == [player_label, player_label, player_label]:
          ######################################################################
          # if we found a winning move we want to update the move with a score #
          ######################################################################
          # your code to update avail_moves with a score
          avail_moves[move] = 100
    temp_board[move] = ' '

  ##############################################################################
  ################## Check if the opponent has a winning move ##################
  ##############################################################################
  for move in avail_moves.keys():
    temp_board[move] = opponent
    for pattern in self.win_patterns:
        values = [temp_board[i] for i in pattern] 
        if values == [opponent, opponent, opponent]:
          avail_moves[move] = 10

    temp_board[move] = ' '
  ##############################################################################
  ################### All remaining moves receive a score of 1
  ##############################################################################

  return avail_moves
```

### Q3 Test `one_step_ai`

That's great, but how do we check that our code will work when a winning move is available, or a losing move is just around the corner? let's create a unit test for these!


```python
# just defining a new game
self = TicTacToe()
player_label = 'X'
```


```python
# seeding the board with some X's
self.board[1] = 'X'
self.board[2] = 'X'
self.board
```




    {1: 'X', 2: 'X', 3: ' ', 4: ' ', 5: ' ', 6: ' ', 7: ' ', 8: ' ', 9: ' '}



Now test the winning move. Your code should return `100` at move `3` and 1 everywhere else


```python
one_step_ai(self, player_label)
```




    {3: 100, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}



We can test the losing move by reversing the players


```python
player_label = 'O'
one_step_ai(self, player_label)
```




    {3: 10, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}



great! Let's keep these shorthand codes in mind when we go to write actual unit tests with the `one_step_ai` function embedded in the `GameEngine` module.

We're not done yet, recall that our other ai agents returned the actual selected move, not a dictionary of the moves with scores. We need to create a move from this dictionary and return it.

Here's what the general procedure will look like:

1. Grab the maximum score (after assigning scores to all of avail_moves)
2. Select all moves that have this maximum score
3. Return a random selection of the moves with the max score

and then in code format:


```python
avail_moves = one_step_ai(self, player_label)

# 1. grab the maximum score
max_score = max(avail_moves.values())

# 2. select all moves that have this maximum score
valid = []
for key, value in avail_moves.items():
  if value == max_score:
    valid.append(key)

# 3. return a random selection of the moves with the max score
move = random.choice(valid)
move
```




    3



## 3.2 Putting it all together



### Q4 Finish `one_step_ai` to return a move

Let's see if we can rewrite our game engine to take new AI models in as a passable parameter. This way our base module will be much cleaner, and allow us to continue to write new functions for the base engine as long as they pass along the same variables.


```python
# the beginning portion of our code will look about the same
# as the heuristic AI model
def one_step_ai(board, win_patterns, player_label):
  opponent = ['X', 'O']
  opponent.remove(player_label)
  opponent = opponent[0]
  temp_board = board.copy()

  # define avail_moves
  ##############################################################################
  ############################# DEFINE avail_moves #############################
  ##############################################################################
  avail_moves = {i:' ' for i in board if board[i] == ' '}
  for move in avail_moves.keys():
    avail_moves[move] = 1
  temp_board = board.copy()

  # first check for a winning move
  # we're now looping through the keys of our dictionary
  for move in avail_moves.keys():
    temp_board[move] = player_label
    for pattern in win_patterns:
        values = [temp_board[i] for i in pattern] 
        if values == [player_label, player_label, player_label]:
          ######################################################################
          # if we found a winning move we want to update the move with a score #
          ######################################################################
          # your code to update avail_moves with a score
          avail_moves[move] = 100
    temp_board[move] = ' '

  ##############################################################################
  ################## Check if the opponent has a winning move ##################
  ##############################################################################
  for move in avail_moves.keys():
    temp_board[move] = opponent
    for pattern in win_patterns:
        values = [temp_board[i] for i in pattern] 
        if values == [opponent, opponent, opponent]:
          avail_moves[move] = 10

    temp_board[move] = ' '
  # 1. grab the maximum score
  max_score = max(avail_moves.values())

  # 2. select all moves that have this maximum score
  valid = []
  for key, value in avail_moves.items():
    if value == max_score:
      valid.append(key)

  # 3. return a random selection of the moves with the max score
  move = random.choice(valid)
  
  return move
```

### 3.2.1 Allow `GameEngine` to take an ai agent as a passable parameter

Let's rewrite our `GameEngine` to take an ai agent as a passable parameter under `user_ai`. The default value will be `None`

Additional `user_ai` criteria will be that `user_ai` receives `board`, `win_patterns` and `player_label` and returns `move`.


```python
class GameEngine(TicTacToe):
  def __init__(self, setup='auto', user_ai=None):
    super().__init__()
    self.setup = setup
    self.user_ai = user_ai

  def heuristic_ai(self, player_label):
    opponent = ['X', 'O']
    opponent.remove(player_label)
    opponent = opponent[0]

    avail_moves = [i for i in self.board.keys() if self.board[i] == ' ']
    temp_board = self.board.copy()
    middle = 5
    corner = [1,3,7,9]
    side = [2,4,6,8]

    # first check for a winning move
    move_found = False
    for move in avail_moves:
      temp_board[move] = player_label
      for pattern in self.win_patterns:
          values = [temp_board[i] for i in pattern] 
          if values == [player_label, player_label, player_label]:
            move_found = True       
            break
      if move_found:   
        break
      else:
        temp_board[move] = ' '

    # check if the opponent has a winning move
    if move_found == False:
      for move in avail_moves:
        temp_board[move] = opponent
        for pattern in self.win_patterns:
            values = [temp_board[i] for i in pattern] 
            if values == [opponent, opponent, opponent]:
              move_found = True       
              break
        if move_found:   
          break
        else:
          temp_board[move] = ' '

    # check corners
    if move_found == False:
      move_corner = [val for val in avail_moves if val in corner]
      if len(move_corner) > 0:
        move = random.choice(move_corner)
        move_found = True
        
    # check if middle avail
    if move_found == False:
      if middle in avail_moves:
        move_found = True
        move = middle

    # check side
    if move_found == False:
      move_side = [val for val in avail_moves if val in side]
      if len(move_side) > 0:
        move = random.choice(move_side)
        move_found = True

    return move

  def random_ai(self):
    while True:
      move = random.randint(1,9)
      if self.board[move] != ' ':
        continue
      else:
        break
    return move

  def setup_game(self):

    if self.setup == 'user':
      players = int(input("How many Players? (type 0, 1, or 2)"))
      self.player_meta = {'first': {'label': 'X',
                                    'type': 'ai'}, 
                    'second': {'label': 'O',
                                    'type': 'human'}}
      if players != 2:
        ########## 
        # Allow the user to set the ai level
        ########## 

        ### if they have not provided an ai_agent
        if self.user_ai == None:
          level = int(input("select AI level (1, 2)"))
          if level == 1:
            self.ai_level = 1
          elif level == 2:
            self.ai_level = 2
          else:
            print("Unknown AI level entered, this will cause problems")
        else:
          self.ai_level = 3

      if players == 1:
        first = input("who will go first? (X, (AI), or O (Player))")
        if first == 'O':
          self.player_meta = {'second': {'label': 'X',
                                    'type': 'ai'}, 
                        'first': {'label': 'O',
                                    'type': 'human'}}
        

      elif players == 0:
        first = random.choice(['X', 'O'])
        if first == 'O':
          self.player_meta = {'second': {'label': 'X',
                                    'type': 'ai'}, 
                        'first': {'label': 'O',
                                    'type': 'ai'}}                                
        else:
          self.player_meta = {'first': {'label': 'X',
                                    'type': 'ai'}, 
                        'second': {'label': 'O',
                                    'type': 'ai'}}

        
    elif self.setup == 'auto':
      first = random.choice(['X', 'O'])
      if first == 'O':
        self.start_player = 'O'
        self.player_meta = {'second': {'label': 'X',
                                  'type': 'ai'}, 
                      'first': {'label': 'O',
                                  'type': 'ai'}}                                
      else:
        self.start_player = 'X'
        self.player_meta = {'first': {'label': 'X',
                                  'type': 'ai'}, 
                      'second': {'label': 'O',
                                  'type': 'ai'}}
      ########## 
      # and automatically set the ai level otherwise
      ##########  
      if self.user_ai == None:                           
        self.ai_level = 2
      else:
        self.ai_level = 3

  def play_game(self):
    while True:
      for player in ['first', 'second']:  
        self.visualize_board()
        player_label = self.player_meta[player]['label']
        player_type = self.player_meta[player]['type']

        if player_type == 'human':
          move = input("{}, what's your move?".format(player_label))
          # we're going to allow the user to quit the game from the input line
          if move in ['q', 'quit']:
            self.winner = 'F'
            print('quiting the game')
            break

          move = int(move)
          if self.board[move] != ' ':
            while True:
              move = input("{}, that position is already taken! "\
                          "What's your move?".format(player_label))  
              move = int(move)            
              if self.board[move] != ' ':
                continue
              else:
                break

        else:
          ##########
          # Our level 1 ai agent (random)
          ##########
          if self.ai_level == 1:
            move = self.random_ai()

          ##########
          # Our level 2 ai agent (heuristic)
          ##########
          elif self.ai_level == 2:
            move = self.heuristic_ai(player_label)

          ##########
          # Our user-defined AI agent
          ##########
          elif self.ai_level == 3:
            move = self.user_ai(self.board, self.win_patterns, player_label)

        self.board[move] = player_label

        # the winner varaible will now be check within the board object
        self.check_winning()
        self.check_stalemate()

        if self.winner == '':
          continue

        elif self.winner == 'Stalemate':
          print(self.check_stalemate())
          self.visualize_board()
          break

        else:
          print(self.check_winning())
          self.visualize_board()
          break
      if self.winner != '':
        return self
```

Test the `auto` and `user` functions


```python
game = GameEngine(setup='user', user_ai=one_step_ai)
```


```python
game.setup_game()
```

    How many Players? (type 0, 1, or 2) 1
    who will go first? (X, (AI), or O (Player)) X



```python
game.play_game()
```

    | | | |
    | | | |
    | | | |
    
    | | | |
    |X| | |
    | | | |
    


    O, what's your move? 3


    | | |O|
    |X| | |
    | | | |
    
    | |X|O|
    |X| | |
    | | | |
    


    O, what's your move? 6


    | |X|O|
    |X| |O|
    | | | |
    
    | |X|O|
    |X| |O|
    | | |X|
    


    O, what's your move? 1


    |O|X|O|
    |X| |O|
    | | |X|
    
    |O|X|O|
    |X| |O|
    | |X|X|
    


    O, what's your move? 5


    |O|X|O|
    |X|O|O|
    | |X|X|
    
    'X' Won!
    |O|X|O|
    |X|O|O|
    |X|X|X|
    





    <__main__.GameEngine at 0x7f4614708850>



## 3.3 Write Unit Tests for the New Code

There are many tests we could write here


```python
def test_user_ai():
  random.seed(42)
  game = GameEngine(setup='auto', user_ai=one_step_ai)
  game.setup_game()
  outcome = game.play_game()
  assert outcome.winner == 'X', 'X should have won!'
```


```python
test_user_ai()
```

    | | | |
    | | | |
    | | | |
    
    |X| | |
    | | | |
    | | | |
    
    |X| | |
    | | |O|
    | | | |
    
    |X| |X|
    | | |O|
    | | | |
    
    |X|O|X|
    | | |O|
    | | | |
    
    |X|O|X|
    | |X|O|
    | | | |
    
    |X|O|X|
    | |X|O|
    |O| | |
    
    'X' Won!
    |X|O|X|
    | |X|O|
    |O| |X|
    

