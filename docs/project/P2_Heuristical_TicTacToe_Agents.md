<a href="https://colab.research.google.com/github/wesleybeckner/technology_fundamentals/blob/main/C3%20Machine%20Learning%20I/LABS_PROJECT/Tech_Fun_C3_P4_Game_AI%2C_Heuristical_Agents.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Technology Fundamentals Course 3, Project Part 4: Heuristical Agents (Symbolic AI)

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com

**Teaching Assitants**: Varsha Bang, Harsha Vardhan

**Contact**: vbang@uw.edu, harshav@uw.edu
<br>

---

<br>

We makin' some wack AI today

<br>

---

<br>

<a name='top'></a>


<a name='x.0'></a>

## 4.0 Preparing Environment and Importing Data

[back to top](#top)

<a name='x.0.1'></a>

### 4.0.1 Import Packages

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

  def setup_game(self):

    if self.setup == 'user':
      players = int(input("How many Players? (type 0, 1, or 2)"))
      self.player_meta = {'first': {'label': 'X',
                                    'type': 'human'}, 
                    'second': {'label': 'O',
                                    'type': 'human'}}
      if players == 1:
        first = input("who will go first? (X, (AI), or O (Player))")
        if first == 'O':
          self.player_meta = {'second': {'label': 'X',
                                    'type': 'ai'}, 
                        'first': {'label': 'O',
                                    'type': 'human'}}
        else:
          self.player_meta = {'first': {'label': 'X',
                                    'type': 'ai'}, 
                        'second': {'label': 'O',
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
                          "What's your move?".format(player))  
              move = int(move)            
              if self.board[move] != ' ':
                continue
              else:
                break

        else:
          while True:
            move = random.randint(1,9)
            if self.board[move] != ' ':
              continue
              print('test')
            else:
              break

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

### 4.0.2 Load Dataset

[back to top](#top)

## 4.3 AI Heuristics

Develop a better AI based on your analyses of game play so far.

### Q1

In our groups, let's discuss what rules we would like to hard code in. Harsha, Varsha and I will help you with the flow control to program these rules


```python
# we will define some variables to help us define the types of positions
middle = 5
side = [2, 4, 6, 8]
corner = [1, 3, 7, 9]
```


```python
# recall that our board is a dictionary
tictactoe = TicTacToe()
tictactoe.board
```




    {1: ' ', 2: ' ', 3: ' ', 4: ' ', 5: ' ', 6: ' ', 7: ' ', 8: ' ', 9: ' '}




```python
# and we have a win_patterns object to help us with the algorithm
tictactoe.win_patterns
```




    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9],
     [1, 4, 7],
     [2, 5, 8],
     [3, 6, 9],
     [1, 5, 9],
     [7, 5, 3]]



for example, if we want to check if the middle piece is available, and play it if it is. How do we do that?


```python
# set some key variables
player = 'X'
opponent = 'O'
avail_moves = [i for i in tictactoe.board.keys() if tictactoe.board[i] == ' ']

# a variable that will keep track if we've found a move we like or not
move_found = False

# <- some other moves we might want to make would go here -> #

# and now for our middle piece play
if move_found == False: # if no other move has been found yet
  if middle in avail_moves: # if middle is available
    move_found = True # then change our move_found status
    move = middle # update our move
```

Our standard approach will be to always ***return a move by the agent***. Whether the agent is heruistical or from some other ML framework we ***always want to return a move***

Repeate after me: ***ALWAYS RETURN A MOVE***. Make sure you know what move is. Make sure you know what it is. And return it. Return a move. The purpose of the next lines of code we will write is to return a move.

Make sure your code returns a move.

### Q2

Write down your algorithm steps in markdown. i.e.

1. play a corner piece
2. play to opposite corner from the opponent, etc.
3. ....etc.

### Q3

Begin to codify your algorithm from Q3. Make sure that no matter what, you ***return a move***


```python
# some starting variables for you
player_label = 'X'
opponent = 'O'
avail_moves = [i for i in tictactoe.board.keys() if tictactoe.board[i] == ' ']

# temp board will allow us to play hypothetical moves and see where they get us
# in case you need it
temp_board = tictactoe.board.copy()

```

## 4.4 Wrapping our Agent

Now that we've created a conditional tree for our AI to make a decision, we need to integrate this within the gaming framework we've made so far. How should we do this? Let's define this thought pattern or tree as an agent.

Recall our play_game function within `GameEngine`



```python
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
                          "What's your move?".format(player))  
              move = int(move)            
              if self.board[move] != ' ':
                continue
              else:
                break

        ########################################################################
        ##################### WE WANT TO CHANGE THESE LINES ####################
        ########################################################################
        else:
          while True:
            move = random.randint(1,9)
            if self.board[move] != ' ':
              continue
              print('test')
            else:
              break

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

### 4.4.1 Redefining the Random Agent

In particular, we want to change lines 30-37 to take our gaming agent in as a parameter to make decisions. Let's try this.

In `setup_game` we want to have the option to set the AI type/level. In `play_game` we want to make a call to that AI to make the move. For instance, our random AI will go from:

```
while True:
  move = random.randint(1,9)
  if self.board[move] != ' ':
    continue
  else:
    break
```

to:

```
def random_ai(self):
  while True:
    move = random.randint(1,9)
    if self.board[move] != ' ':
      continue
    else:
      break
  return move
```



```python
class GameEngine(TicTacToe):
  def __init__(self, setup='auto'):
    super().__init__()
    self.setup = setup

  ##############################################################################
  ########## our fresh off the assembly line tictactoe playing robot ###########
  ##############################################################################
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
                                    'type': 'human'}, 
                    'second': {'label': 'O',
                                    'type': 'human'}}
      if players != 2:
        ######################################################################## 
        ################# Allow the user to set the ai level ###################
        ######################################################################## 
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
        else:
          self.player_meta = {'first': {'label': 'X',
                                    'type': 'ai'}, 
                        'second': {'label': 'O',
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
      ##########################################################################
      ############## and automatically set the ai level otherwise ##############
      ##########################################################################
      self.ai_level = 1

  def play_game(self):
    while True:
      for player in ['first', 'second']:  
        self.visualize_board()
        player_label = self.player_meta[player]['label']
        player_type = self.player_meta[player]['type']

        if player_type == 'human':
          move = input("{}, what's your move?".format(player_label))
          if move in ['q', 'quit']:
            self.winner = 'F'
            print('quiting the game')
            break

          move = int(move)
          if self.board[move] != ' ':
            while True:
              move = input("{}, that position is already taken! "\
                          "What's your move?".format(player))  
              move = int(move)            
              if self.board[move] != ' ':
                continue
              else:
                break

        else:
          if self.ai_level == 1:
            move = self.random_ai()

          ######################################################################
          ############## we will leave this setting empty for now ##############
          ######################################################################
          elif self.ai_level == 2:
            pass

        self.board[move] = player_label
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

Let's test that our random ai works now in this format


```python
random.seed(12)
game = GameEngine(setup='auto')
game.setup_game()
game.play_game()
```

    | | | |
    | | | |
    | | | |
    
    | | | |
    | |O| |
    | | | |
    
    | | | |
    | |O| |
    | | |X|
    
    | | | |
    | |O|O|
    | | |X|
    
    | | |X|
    | |O|O|
    | | |X|
    
    | | |X|
    | |O|O|
    |O| |X|
    
    |X| |X|
    | |O|O|
    |O| |X|
    
    |X| |X|
    | |O|O|
    |O|O|X|
    
    |X| |X|
    |X|O|O|
    |O|O|X|
    
    'O' Won!
    |X|O|X|
    |X|O|O|
    |O|O|X|
    





    <__main__.GameEngine at 0x7fadbea428d0>



Let's try it with a user player:


```python
random.seed(12)
game = GameEngine(setup='user')
game.setup_game()
game.play_game()
```

    How many Players? (type 0, 1, or 2)2
    | | | |
    | | | |
    | | | |
    
    X, what's your move?q
    quiting the game





    <__main__.GameEngine at 0x7fadbea25e90>



### Q4

Now let's fold in our specialized AI agent. Add your code under the `heurstic_ai` function. Note that the `player_label` is passed as an input parameter now


```python
class GameEngine(TicTacToe):
  def __init__(self, setup='auto'):
    super().__init__()
    self.setup = setup

  ##############################################################################
  ################### YOUR BADASS HEURISTIC AGENT GOES HERE ####################
  ##############################################################################
  def heuristic_ai(self, player_label):

    # SOME HELPER VARIABLES IF YOU NEED THEM 
    opponent = ['X', 'O']
    opponent.remove(player_label)
    opponent = opponent[0]

    avail_moves = [i for i in self.board.keys() if self.board[i] == ' ']
    temp_board = self.board.copy()
    
    ################## YOUR CODE GOES HERE, RETURN THAT MOVE! ##################
    while True: # DELETE LINES 20 - 25, USED FOR TESTING PURPOSES ONLY
      move = random.randint(1,9)
      if self.board[move] != ' ':
        continue
      else:
        break
    ############################################################################
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
                                    'type': 'human'}, 
                    'second': {'label': 'O',
                                    'type': 'human'}}
      if players != 2:
        ######################################################################## 
        ################# Allow the user to set the ai level ###################
        ######################################################################## 
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
        else:
          self.player_meta = {'first': {'label': 'X',
                                    'type': 'ai'}, 
                        'second': {'label': 'O',
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
      ##########################################################################
      ############## and automatically set the ai level otherwise ##############
      ##########################################################################
      self.ai_level = 1

  def play_game(self):
    while True:
      for player in ['first', 'second']:  
        self.visualize_board()
        player_label = self.player_meta[player]['label']
        player_type = self.player_meta[player]['type']

        if player_type == 'human':
          move = input("{}, what's your move?".format(player_label))
          if move in ['q', 'quit']:
            self.winner = 'F'
            print('quiting the game')
            break

          move = int(move)
          if self.board[move] != ' ':
            while True:
              move = input("{}, that position is already taken! "\
                          "What's your move?".format(player))  
              move = int(move)            
              if self.board[move] != ' ':
                continue
              else:
                break

        else:
          if self.ai_level == 1:
            move = self.random_ai()

          ######################################################################
          ############## we will leave this setting empty for now ##############
          ######################################################################
          elif self.ai_level == 2:
            move = self.heuristic_ai(player_label)

        self.board[move] = player_label
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

### Q5

And we'll test that it works!


```python
random.seed(12)
game = GameEngine(setup='user')
game.setup_game()
game.play_game()
```

    How many Players? (type 0, 1, or 2)1
    select AI level (1, 2)2
    who will go first? (X, (AI), or O (Player))O
    | | | |
    | | | |
    | | | |
    
    O, what's your move?5
    | | | |
    | |O| |
    | | | |
    
    | | | |
    | |O| |
    | |X| |
    
    O, what's your move?9
    | | | |
    | |O| |
    | |X|O|
    
    | | | |
    | |O|X|
    | |X|O|
    
    O, what's your move?1
    'O' Won!
    |O| | |
    | |O|X|
    | |X|O|
    





    <__main__.GameEngine at 0x7fadbe93f610>



### Q6 

Test the autorun feature!


```python
game = GameEngine(setup='auto')
game.setup_game()
game.play_game()
```

    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    |O| | |
    
    | |X| |
    | | | |
    |O| | |
    
    |O|X| |
    | | | |
    |O| | |
    
    |O|X| |
    | | | |
    |O| |X|
    
    'O' Won!
    |O|X| |
    |O| | |
    |O| |X|
    





    <__main__.GameEngine at 0x7fadbe8cc050>


