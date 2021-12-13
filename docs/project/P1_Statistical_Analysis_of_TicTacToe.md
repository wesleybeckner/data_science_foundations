<a href="https://colab.research.google.com/github/ronva-h/technology_fundamentals/blob/main/C3%20Machine%20Learning%20I/LABS_PROJECT/Tech%20Fun%20C3%20P3%20Game%20AI%2C%20Statistics.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Technology Fundamentals Course 3, Project Part 3: Statistical Analysis 

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com

**Teaching Assistants**: Varsha Bang, Harsha Vardhan

**Contact**: vbang@uw.edu, harshav@uw.edu
<br>

---

<br>

Today we are going to perform statistical analysis on data generated from our tictactoe program!

<br>

---

<br>

<a name='x.0'></a>

## 2.0 Preparing Environment and Importing Data

[back to top](#top)

<a name='x.0.1'></a>

### 2.0.1 Import Packages

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
                                    'type': 'ai'}, 
                    'second': {'label': 'O',
                                    'type': 'human'}}
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

### 2.0.2 Load Dataset

[back to top](#top)


```python
data = {}
for i in range(1000):
  game = GameEngine()
  game.setup_game()
  board = game.play_game()
  data['game {}'.format(i)] = {'board': board.board,
          'winner': board.winner,
          'starting player': board.start_player}
```

    [1;30;43mStreaming output truncated to the last 5000 lines.[0m
    
    |X|O| |
    | | |O|
    |X| | |
    
    |X|O| |
    | |O|O|
    |X| | |
    
    |X|O| |
    | |O|O|
    |X|X| |
    
    |X|O| |
    | |O|O|
    |X|X|O|
    
    'X' Won!
    |X|O| |
    |X|O|O|
    |X|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |X| | |
    | | | |
    
    | | |O|
    |X| | |
    | | | |
    
    | | |O|
    |X| | |
    | |X| |
    
    | | |O|
    |X| | |
    | |X|O|
    
    | | |O|
    |X| |X|
    | |X|O|
    
    | | |O|
    |X| |X|
    |O|X|O|
    
    | |X|O|
    |X| |X|
    |O|X|O|
    
    |O|X|O|
    |X| |X|
    |O|X|O|
    
    'X' Won!
    |O|X|O|
    |X|X|X|
    |O|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | |O| |
    | | | |
    | | | |
    
    | |O| |
    |X| | |
    | | | |
    
    | |O|O|
    |X| | |
    | | | |
    
    |X|O|O|
    |X| | |
    | | | |
    
    |X|O|O|
    |X| | |
    | | |O|
    
    |X|O|O|
    |X|X| |
    | | |O|
    
    'O' Won!
    |X|O|O|
    |X|X|O|
    | | |O|
    
    | | | |
    | | | |
    | | | |
    
    | | |X|
    | | | |
    | | | |
    
    | | |X|
    | | | |
    |O| | |
    
    | | |X|
    | | | |
    |O| |X|
    
    | | |X|
    | | |O|
    |O| |X|
    
    | | |X|
    | |X|O|
    |O| |X|
    
    | | |X|
    |O|X|O|
    |O| |X|
    
    | | |X|
    |O|X|O|
    |O|X|X|
    
    | |O|X|
    |O|X|O|
    |O|X|X|
    
    'X' Won!
    |X|O|X|
    |O|X|O|
    |O|X|X|
    
    | | | |
    | | | |
    | | | |
    
    | |O| |
    | | | |
    | | | |
    
    | |O| |
    |X| | |
    | | | |
    
    | |O| |
    |X| | |
    | | |O|
    
    |X|O| |
    |X| | |
    | | |O|
    
    |X|O|O|
    |X| | |
    | | |O|
    
    |X|O|O|
    |X|X| |
    | | |O|
    
    'O' Won!
    |X|O|O|
    |X|X|O|
    | | |O|
    
    | | | |
    | | | |
    | | | |
    
    | |O| |
    | | | |
    | | | |
    
    | |O|X|
    | | | |
    | | | |
    
    | |O|X|
    | |O| |
    | | | |
    
    | |O|X|
    | |O| |
    |X| | |
    
    |O|O|X|
    | |O| |
    |X| | |
    
    |O|O|X|
    | |O|X|
    |X| | |
    
    |O|O|X|
    |O|O|X|
    |X| | |
    
    'X' Won!
    |O|O|X|
    |O|O|X|
    |X| |X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | |X|
    | | | |
    
    | | | |
    | | |X|
    | |O| |
    
    | | | |
    | | |X|
    |X|O| |
    
    |O| | |
    | | |X|
    |X|O| |
    
    |O|X| |
    | | |X|
    |X|O| |
    
    |O|X| |
    | |O|X|
    |X|O| |
    
    |O|X| |
    | |O|X|
    |X|O|X|
    
    |O|X|O|
    | |O|X|
    |X|O|X|
    
    It's a stalemate!
    |O|X|O|
    |X|O|X|
    |X|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | |O| |
    | | | |
    | | | |
    
    | |O| |
    | | | |
    | |X| |
    
    | |O| |
    | | | |
    | |X|O|
    
    | |O|X|
    | | | |
    | |X|O|
    
    | |O|X|
    | | | |
    |O|X|O|
    
    |X|O|X|
    | | | |
    |O|X|O|
    
    |X|O|X|
    | | |O|
    |O|X|O|
    
    |X|O|X|
    | |X|O|
    |O|X|O|
    
    It's a stalemate!
    |X|O|X|
    |O|X|O|
    |O|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | |X| |
    | | | |
    | | | |
    
    | |X| |
    | | | |
    | |O| |
    
    | |X| |
    | |X| |
    | |O| |
    
    | |X| |
    | |X| |
    | |O|O|
    
    | |X| |
    |X|X| |
    | |O|O|
    
    | |X|O|
    |X|X| |
    | |O|O|
    
    'X' Won!
    | |X|O|
    |X|X|X|
    | |O|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |O| |
    | | | |
    
    | |X| |
    | |O| |
    | | | |
    
    |O|X| |
    | |O| |
    | | | |
    
    |O|X|X|
    | |O| |
    | | | |
    
    |O|X|X|
    |O|O| |
    | | | |
    
    |O|X|X|
    |O|O| |
    | | |X|
    
    'O' Won!
    |O|X|X|
    |O|O| |
    |O| |X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |O| |
    
    | |X| |
    | | | |
    | |O| |
    
    |O|X| |
    | | | |
    | |O| |
    
    |O|X| |
    | | | |
    | |O|X|
    
    |O|X| |
    | | |O|
    | |O|X|
    
    |O|X| |
    | |X|O|
    | |O|X|
    
    |O|X|O|
    | |X|O|
    | |O|X|
    
    |O|X|O|
    |X|X|O|
    | |O|X|
    
    It's a stalemate!
    |O|X|O|
    |X|X|O|
    |O|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |O| |
    | | | |
    
    | | | |
    |X|O| |
    | | | |
    
    | | |O|
    |X|O| |
    | | | |
    
    | | |O|
    |X|O|X|
    | | | |
    
    | | |O|
    |X|O|X|
    | |O| |
    
    | | |O|
    |X|O|X|
    |X|O| |
    
    'O' Won!
    | |O|O|
    |X|O|X|
    |X|O| |
    
    | | | |
    | | | |
    | | | |
    
    | |O| |
    | | | |
    | | | |
    
    | |O|X|
    | | | |
    | | | |
    
    | |O|X|
    | | | |
    |O| | |
    
    | |O|X|
    | |X| |
    |O| | |
    
    | |O|X|
    |O|X| |
    |O| | |
    
    | |O|X|
    |O|X|X|
    |O| | |
    
    'O' Won!
    |O|O|X|
    |O|X|X|
    |O| | |
    
    | | | |
    | | | |
    | | | |
    
    | | |X|
    | | | |
    | | | |
    
    | | |X|
    | |O| |
    | | | |
    
    |X| |X|
    | |O| |
    | | | |
    
    |X| |X|
    |O|O| |
    | | | |
    
    |X| |X|
    |O|O| |
    |X| | |
    
    |X| |X|
    |O|O| |
    |X|O| |
    
    |X| |X|
    |O|O|X|
    |X|O| |
    
    |X| |X|
    |O|O|X|
    |X|O|O|
    
    'X' Won!
    |X|X|X|
    |O|O|X|
    |X|O|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | |O|
    | | | |
    
    | |X| |
    | | |O|
    | | | |
    
    | |X| |
    | | |O|
    | |O| |
    
    | |X| |
    | | |O|
    |X|O| |
    
    | |X|O|
    | | |O|
    |X|O| |
    
    | |X|O|
    | | |O|
    |X|O|X|
    
    | |X|O|
    | |O|O|
    |X|O|X|
    
    | |X|O|
    |X|O|O|
    |X|O|X|
    
    It's a stalemate!
    |O|X|O|
    |X|O|O|
    |X|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    |O| | |
    
    | | |X|
    | | | |
    |O| | |
    
    |O| |X|
    | | | |
    |O| | |
    
    |O| |X|
    | | | |
    |O|X| |
    
    |O| |X|
    | | |O|
    |O|X| |
    
    |O| |X|
    |X| |O|
    |O|X| |
    
    |O| |X|
    |X| |O|
    |O|X|O|
    
    |O| |X|
    |X|X|O|
    |O|X|O|
    
    It's a stalemate!
    |O|O|X|
    |X|X|O|
    |O|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | | |O|
    
    | | | |
    | |X| |
    | | |O|
    
    | | | |
    |O|X| |
    | | |O|
    
    | | | |
    |O|X|X|
    | | |O|
    
    |O| | |
    |O|X|X|
    | | |O|
    
    |O| |X|
    |O|X|X|
    | | |O|
    
    |O| |X|
    |O|X|X|
    | |O|O|
    
    'X' Won!
    |O| |X|
    |O|X|X|
    |X|O|O|
    
    | | | |
    | | | |
    | | | |
    
    | |O| |
    | | | |
    | | | |
    
    | |O| |
    |X| | |
    | | | |
    
    | |O| |
    |X| | |
    | |O| |
    
    | |O| |
    |X| | |
    |X|O| |
    
    | |O| |
    |X| |O|
    |X|O| |
    
    | |O|X|
    |X| |O|
    |X|O| |
    
    | |O|X|
    |X| |O|
    |X|O|O|
    
    'X' Won!
    |X|O|X|
    |X| |O|
    |X|O|O|
    
    | | | |
    | | | |
    | | | |
    
    | | |O|
    | | | |
    | | | |
    
    | | |O|
    | |X| |
    | | | |
    
    | |O|O|
    | |X| |
    | | | |
    
    | |O|O|
    | |X| |
    |X| | |
    
    | |O|O|
    | |X|O|
    |X| | |
    
    |X|O|O|
    | |X|O|
    |X| | |
    
    |X|O|O|
    |O|X|O|
    |X| | |
    
    'X' Won!
    |X|O|O|
    |O|X|O|
    |X| |X|
    
    | | | |
    | | | |
    | | | |
    
    | |O| |
    | | | |
    | | | |
    
    | |O| |
    |X| | |
    | | | |
    
    | |O|O|
    |X| | |
    | | | |
    
    | |O|O|
    |X| | |
    | |X| |
    
    | |O|O|
    |X| | |
    |O|X| |
    
    | |O|O|
    |X|X| |
    |O|X| |
    
    | |O|O|
    |X|X| |
    |O|X|O|
    
    |X|O|O|
    |X|X| |
    |O|X|O|
    
    'O' Won!
    |X|O|O|
    |X|X|O|
    |O|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    |O| | |
    
    | | |X|
    | | | |
    |O| | |
    
    | | |X|
    |O| | |
    |O| | |
    
    | | |X|
    |O|X| |
    |O| | |
    
    | | |X|
    |O|X| |
    |O|O| |
    
    |X| |X|
    |O|X| |
    |O|O| |
    
    |X|O|X|
    |O|X| |
    |O|O| |
    
    |X|O|X|
    |O|X|X|
    |O|O| |
    
    'O' Won!
    |X|O|X|
    |O|X|X|
    |O|O|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |O| |
    | | | |
    
    | | | |
    | |O|X|
    | | | |
    
    | | |O|
    | |O|X|
    | | | |
    
    |X| |O|
    | |O|X|
    | | | |
    
    |X| |O|
    | |O|X|
    | | |O|
    
    |X| |O|
    |X|O|X|
    | | |O|
    
    |X|O|O|
    |X|O|X|
    | | |O|
    
    |X|O|O|
    |X|O|X|
    | |X|O|
    
    'O' Won!
    |X|O|O|
    |X|O|X|
    |O|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    |O| | |
    
    | |X| |
    | | | |
    |O| | |
    
    | |X| |
    | | | |
    |O|O| |
    
    |X|X| |
    | | | |
    |O|O| |
    
    |X|X| |
    | | |O|
    |O|O| |
    
    'X' Won!
    |X|X|X|
    | | |O|
    |O|O| |
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | | |X|
    
    | | | |
    | | | |
    |O| |X|
    
    |X| | |
    | | | |
    |O| |X|
    
    |X| | |
    | |O| |
    |O| |X|
    
    |X| | |
    |X|O| |
    |O| |X|
    
    |X|O| |
    |X|O| |
    |O| |X|
    
    |X|O| |
    |X|O|X|
    |O| |X|
    
    'O' Won!
    |X|O|O|
    |X|O|X|
    |O| |X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | |X|
    | | | |
    
    | | | |
    |O| |X|
    | | | |
    
    | |X| |
    |O| |X|
    | | | |
    
    | |X| |
    |O|O|X|
    | | | |
    
    |X|X| |
    |O|O|X|
    | | | |
    
    |X|X| |
    |O|O|X|
    | | |O|
    
    'X' Won!
    |X|X|X|
    |O|O|X|
    | | |O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | | |O|
    
    | | | |
    | |X| |
    | | |O|
    
    | | |O|
    | |X| |
    | | |O|
    
    | | |O|
    | |X|X|
    | | |O|
    
    |O| |O|
    | |X|X|
    | | |O|
    
    |O| |O|
    | |X|X|
    |X| |O|
    
    |O| |O|
    |O|X|X|
    |X| |O|
    
    |O| |O|
    |O|X|X|
    |X|X|O|
    
    'O' Won!
    |O|O|O|
    |O|X|X|
    |X|X|O|
    
    | | | |
    | | | |
    | | | |
    
    |O| | |
    | | | |
    | | | |
    
    |O|X| |
    | | | |
    | | | |
    
    |O|X| |
    |O| | |
    | | | |
    
    |O|X|X|
    |O| | |
    | | | |
    
    |O|X|X|
    |O| |O|
    | | | |
    
    |O|X|X|
    |O| |O|
    | |X| |
    
    'O' Won!
    |O|X|X|
    |O| |O|
    |O|X| |
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    |X| | |
    
    | | | |
    | |O| |
    |X| | |
    
    | | | |
    | |O| |
    |X| |X|
    
    | | | |
    | |O| |
    |X|O|X|
    
    | |X| |
    | |O| |
    |X|O|X|
    
    | |X|O|
    | |O| |
    |X|O|X|
    
    | |X|O|
    | |O|X|
    |X|O|X|
    
    | |X|O|
    |O|O|X|
    |X|O|X|
    
    It's a stalemate!
    |X|X|O|
    |O|O|X|
    |X|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |O| |
    
    | | | |
    | | |X|
    | |O| |
    
    | |O| |
    | | |X|
    | |O| |
    
    | |O|X|
    | | |X|
    | |O| |
    
    | |O|X|
    | | |X|
    | |O|O|
    
    | |O|X|
    | |X|X|
    | |O|O|
    
    | |O|X|
    |O|X|X|
    | |O|O|
    
    'X' Won!
    | |O|X|
    |O|X|X|
    |X|O|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    |X| | |
    
    | | | |
    |O| | |
    |X| | |
    
    | | | |
    |O|X| |
    |X| | |
    
    | | | |
    |O|X| |
    |X| |O|
    
    | | | |
    |O|X| |
    |X|X|O|
    
    |O| | |
    |O|X| |
    |X|X|O|
    
    'X' Won!
    |O| |X|
    |O|X| |
    |X|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | |O|
    | | | |
    | | | |
    
    |X| |O|
    | | | |
    | | | |
    
    |X| |O|
    | | | |
    |O| | |
    
    |X| |O|
    |X| | |
    |O| | |
    
    'O' Won!
    |X| |O|
    |X|O| |
    |O| | |
    
    | | | |
    | | | |
    | | | |
    
    |X| | |
    | | | |
    | | | |
    
    |X| | |
    |O| | |
    | | | |
    
    |X|X| |
    |O| | |
    | | | |
    
    |X|X| |
    |O| | |
    | | |O|
    
    |X|X| |
    |O|X| |
    | | |O|
    
    |X|X| |
    |O|X| |
    |O| |O|
    
    'X' Won!
    |X|X|X|
    |O|X| |
    |O| |O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |X| | |
    | | | |
    
    | | | |
    |X| | |
    | |O| |
    
    |X| | |
    |X| | |
    | |O| |
    
    |X| | |
    |X|O| |
    | |O| |
    
    |X| |X|
    |X|O| |
    | |O| |
    
    |X| |X|
    |X|O|O|
    | |O| |
    
    'X' Won!
    |X|X|X|
    |X|O|O|
    | |O| |
    
    | | | |
    | | | |
    | | | |
    
    |X| | |
    | | | |
    | | | |
    
    |X| |O|
    | | | |
    | | | |
    
    |X| |O|
    | | | |
    | |X| |
    
    |X| |O|
    | | |O|
    | |X| |
    
    |X| |O|
    | |X|O|
    | |X| |
    
    'O' Won!
    |X| |O|
    | |X|O|
    | |X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |O| | |
    | | | |
    
    | | | |
    |O| | |
    | | |X|
    
    | | | |
    |O|O| |
    | | |X|
    
    | | |X|
    |O|O| |
    | | |X|
    
    |O| |X|
    |O|O| |
    | | |X|
    
    |O| |X|
    |O|O| |
    | |X|X|
    
    |O|O|X|
    |O|O| |
    | |X|X|
    
    'X' Won!
    |O|O|X|
    |O|O|X|
    | |X|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |O| |
    
    | | | |
    | |X| |
    | |O| |
    
    | | |O|
    | |X| |
    | |O| |
    
    | | |O|
    |X|X| |
    | |O| |
    
    | | |O|
    |X|X| |
    |O|O| |
    
    'X' Won!
    | | |O|
    |X|X|X|
    |O|O| |
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    |X| | |
    
    | |O| |
    | | | |
    |X| | |
    
    | |O| |
    | | | |
    |X|X| |
    
    | |O| |
    |O| | |
    |X|X| |
    
    | |O| |
    |O|X| |
    |X|X| |
    
    |O|O| |
    |O|X| |
    |X|X| |
    
    'X' Won!
    |O|O|X|
    |O|X| |
    |X|X| |
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    |X| | |
    
    | | | |
    | | |O|
    |X| | |
    
    | |X| |
    | | |O|
    |X| | |
    
    | |X| |
    |O| |O|
    |X| | |
    
    |X|X| |
    |O| |O|
    |X| | |
    
    |X|X| |
    |O| |O|
    |X|O| |
    
    |X|X| |
    |O|X|O|
    |X|O| |
    
    |X|X| |
    |O|X|O|
    |X|O|O|
    
    'X' Won!
    |X|X|X|
    |O|X|O|
    |X|O|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |O| |
    
    | |X| |
    | | | |
    | |O| |
    
    |O|X| |
    | | | |
    | |O| |
    
    |O|X| |
    | |X| |
    | |O| |
    
    |O|X| |
    | |X| |
    | |O|O|
    
    |O|X| |
    | |X|X|
    | |O|O|
    
    |O|X| |
    |O|X|X|
    | |O|O|
    
    |O|X| |
    |O|X|X|
    |X|O|O|
    
    It's a stalemate!
    |O|X|O|
    |O|X|X|
    |X|O|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | |O|
    | | | |
    
    | | | |
    | | |O|
    | |X| |
    
    |O| | |
    | | |O|
    | |X| |
    
    |O| | |
    | |X|O|
    | |X| |
    
    |O| | |
    | |X|O|
    | |X|O|
    
    |O| | |
    | |X|O|
    |X|X|O|
    
    'O' Won!
    |O| |O|
    | |X|O|
    |X|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |X| | |
    | | | |
    
    | | | |
    |X| |O|
    | | | |
    
    | | | |
    |X| |O|
    | |X| |
    
    | | | |
    |X| |O|
    |O|X| |
    
    | | | |
    |X|X|O|
    |O|X| |
    
    | | |O|
    |X|X|O|
    |O|X| |
    
    'X' Won!
    | |X|O|
    |X|X|O|
    |O|X| |
    
    | | | |
    | | | |
    | | | |
    
    | | |X|
    | | | |
    | | | |
    
    | | |X|
    | | | |
    |O| | |
    
    | | |X|
    | | | |
    |O|X| |
    
    | | |X|
    | |O| |
    |O|X| |
    
    | | |X|
    |X|O| |
    |O|X| |
    
    | | |X|
    |X|O|O|
    |O|X| |
    
    | |X|X|
    |X|O|O|
    |O|X| |
    
    | |X|X|
    |X|O|O|
    |O|X|O|
    
    'X' Won!
    |X|X|X|
    |X|O|O|
    |O|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |O| |
    | | | |
    
    | | | |
    | |O|X|
    | | | |
    
    | | | |
    | |O|X|
    | | |O|
    
    | | | |
    | |O|X|
    |X| |O|
    
    | |O| |
    | |O|X|
    |X| |O|
    
    | |O|X|
    | |O|X|
    |X| |O|
    
    | |O|X|
    |O|O|X|
    |X| |O|
    
    | |O|X|
    |O|O|X|
    |X|X|O|
    
    'O' Won!
    |O|O|X|
    |O|O|X|
    |X|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |X| |
    
    | | | |
    | |O| |
    | |X| |
    
    | | |X|
    | |O| |
    | |X| |
    
    | |O|X|
    | |O| |
    | |X| |
    
    |X|O|X|
    | |O| |
    | |X| |
    
    |X|O|X|
    |O|O| |
    | |X| |
    
    |X|O|X|
    |O|O|X|
    | |X| |
    
    |X|O|X|
    |O|O|X|
    | |X|O|
    
    It's a stalemate!
    |X|O|X|
    |O|O|X|
    |X|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |X| |
    | | | |
    
    | | | |
    | |X|O|
    | | | |
    
    | | | |
    | |X|O|
    |X| | |
    
    | | |O|
    | |X|O|
    |X| | |
    
    | | |O|
    | |X|O|
    |X| |X|
    
    | | |O|
    |O|X|O|
    |X| |X|
    
    | |X|O|
    |O|X|O|
    |X| |X|
    
    | |X|O|
    |O|X|O|
    |X|O|X|
    
    'X' Won!
    |X|X|O|
    |O|X|O|
    |X|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | |O| |
    | | | |
    | | | |
    
    | |O| |
    |X| | |
    | | | |
    
    | |O| |
    |X|O| |
    | | | |
    
    | |O| |
    |X|O|X|
    | | | |
    
    | |O| |
    |X|O|X|
    |O| | |
    
    | |O| |
    |X|O|X|
    |O| |X|
    
    'O' Won!
    | |O| |
    |X|O|X|
    |O|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |X| |
    | | | |
    
    | | | |
    | |X| |
    | |O| |
    
    | | |X|
    | |X| |
    | |O| |
    
    | | |X|
    |O|X| |
    | |O| |
    
    | | |X|
    |O|X| |
    | |O|X|
    
    | | |X|
    |O|X| |
    |O|O|X|
    
    'X' Won!
    |X| |X|
    |O|X| |
    |O|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    |X| | |
    
    |O| | |
    | | | |
    |X| | |
    
    |O| |X|
    | | | |
    |X| | |
    
    |O|O|X|
    | | | |
    |X| | |
    
    |O|O|X|
    | | | |
    |X|X| |
    
    |O|O|X|
    |O| | |
    |X|X| |
    
    'X' Won!
    |O|O|X|
    |O|X| |
    |X|X| |
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |X| | |
    | | | |
    
    |O| | |
    |X| | |
    | | | |
    
    |O| | |
    |X| | |
    |X| | |
    
    |O| | |
    |X| |O|
    |X| | |
    
    |O| |X|
    |X| |O|
    |X| | |
    
    |O| |X|
    |X| |O|
    |X|O| |
    
    'X' Won!
    |O| |X|
    |X|X|O|
    |X|O| |
    
    | | | |
    | | | |
    | | | |
    
    | | |O|
    | | | |
    | | | |
    
    | | |O|
    | | |X|
    | | | |
    
    | |O|O|
    | | |X|
    | | | |
    
    | |O|O|
    | | |X|
    | |X| |
    
    | |O|O|
    | | |X|
    | |X|O|
    
    |X|O|O|
    | | |X|
    | |X|O|
    
    |X|O|O|
    |O| |X|
    | |X|O|
    
    |X|O|O|
    |O| |X|
    |X|X|O|
    
    It's a stalemate!
    |X|O|O|
    |O|O|X|
    |X|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |X| |
    
    | | | |
    | | | |
    |O|X| |
    
    | | | |
    | |X| |
    |O|X| |
    
    | | |O|
    | |X| |
    |O|X| |
    
    | | |O|
    | |X| |
    |O|X|X|
    
    | |O|O|
    | |X| |
    |O|X|X|
    
    | |O|O|
    |X|X| |
    |O|X|X|
    
    | |O|O|
    |X|X|O|
    |O|X|X|
    
    'X' Won!
    |X|O|O|
    |X|X|O|
    |O|X|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |X| | |
    | | | |
    
    | | | |
    |X| | |
    | |O| |
    
    | | |X|
    |X| | |
    | |O| |
    
    | | |X|
    |X| |O|
    | |O| |
    
    |X| |X|
    |X| |O|
    | |O| |
    
    |X|O|X|
    |X| |O|
    | |O| |
    
    'X' Won!
    |X|O|X|
    |X| |O|
    |X|O| |
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |O| |
    
    | | | |
    |X| | |
    | |O| |
    
    | | | |
    |X| | |
    |O|O| |
    
    | | | |
    |X| | |
    |O|O|X|
    
    |O| | |
    |X| | |
    |O|O|X|
    
    |O|X| |
    |X| | |
    |O|O|X|
    
    |O|X| |
    |X| |O|
    |O|O|X|
    
    |O|X| |
    |X|X|O|
    |O|O|X|
    
    It's a stalemate!
    |O|X|O|
    |X|X|O|
    |O|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    |O| | |
    
    | | |X|
    | | | |
    |O| | |
    
    | | |X|
    | |O| |
    |O| | |
    
    | | |X|
    | |O|X|
    |O| | |
    
    | | |X|
    | |O|X|
    |O| |O|
    
    | | |X|
    | |O|X|
    |O|X|O|
    
    | | |X|
    |O|O|X|
    |O|X|O|
    
    |X| |X|
    |O|O|X|
    |O|X|O|
    
    It's a stalemate!
    |X|O|X|
    |O|O|X|
    |O|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |O| | |
    | | | |
    
    | | | |
    |O| | |
    |X| | |
    
    | | | |
    |O| | |
    |X| |O|
    
    | | | |
    |O|X| |
    |X| |O|
    
    |O| | |
    |O|X| |
    |X| |O|
    
    |O| | |
    |O|X|X|
    |X| |O|
    
    |O| | |
    |O|X|X|
    |X|O|O|
    
    'X' Won!
    |O| |X|
    |O|X|X|
    |X|O|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | | |X|
    
    |O| | |
    | | | |
    | | |X|
    
    |O| |X|
    | | | |
    | | |X|
    
    |O| |X|
    | | | |
    |O| |X|
    
    |O| |X|
    |X| | |
    |O| |X|
    
    |O| |X|
    |X| | |
    |O|O|X|
    
    |O| |X|
    |X|X| |
    |O|O|X|
    
    |O| |X|
    |X|X|O|
    |O|O|X|
    
    It's a stalemate!
    |O|X|X|
    |X|X|O|
    |O|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |O| |
    
    | | | |
    | | |X|
    | |O| |
    
    |O| | |
    | | |X|
    | |O| |
    
    |O| |X|
    | | |X|
    | |O| |
    
    |O|O|X|
    | | |X|
    | |O| |
    
    'X' Won!
    |O|O|X|
    | | |X|
    | |O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |O| |
    | | | |
    
    | |X| |
    | |O| |
    | | | |
    
    | |X| |
    | |O| |
    |O| | |
    
    | |X| |
    |X|O| |
    |O| | |
    
    | |X| |
    |X|O|O|
    |O| | |
    
    |X|X| |
    |X|O|O|
    |O| | |
    
    |X|X| |
    |X|O|O|
    |O|O| |
    
    |X|X| |
    |X|O|O|
    |O|O|X|
    
    'O' Won!
    |X|X|O|
    |X|O|O|
    |O|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |X| | |
    | | | |
    
    | | |O|
    |X| | |
    | | | |
    
    | | |O|
    |X| | |
    |X| | |
    
    | |O|O|
    |X| | |
    |X| | |
    
    | |O|O|
    |X|X| |
    |X| | |
    
    | |O|O|
    |X|X|O|
    |X| | |
    
    | |O|O|
    |X|X|O|
    |X|X| |
    
    'O' Won!
    |O|O|O|
    |X|X|O|
    |X|X| |
    
    | | | |
    | | | |
    | | | |
    
    | |X| |
    | | | |
    | | | |
    
    | |X| |
    | | |O|
    | | | |
    
    | |X| |
    | |X|O|
    | | | |
    
    | |X|O|
    | |X|O|
    | | | |
    
    | |X|O|
    | |X|O|
    | | |X|
    
    | |X|O|
    | |X|O|
    | |O|X|
    
    | |X|O|
    |X|X|O|
    | |O|X|
    
    |O|X|O|
    |X|X|O|
    | |O|X|
    
    It's a stalemate!
    |O|X|O|
    |X|X|O|
    |X|O|X|
    
    | | | |
    | | | |
    | | | |
    
    |X| | |
    | | | |
    | | | |
    
    |X|O| |
    | | | |
    | | | |
    
    |X|O| |
    | |X| |
    | | | |
    
    |X|O| |
    |O|X| |
    | | | |
    
    |X|O| |
    |O|X| |
    |X| | |
    
    |X|O| |
    |O|X|O|
    |X| | |
    
    'X' Won!
    |X|O|X|
    |O|X|O|
    |X| | |
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |X| | |
    | | | |
    
    |O| | |
    |X| | |
    | | | |
    
    |O| | |
    |X| | |
    | | |X|
    
    |O|O| |
    |X| | |
    | | |X|
    
    |O|O| |
    |X| | |
    |X| |X|
    
    |O|O| |
    |X| |O|
    |X| |X|
    
    'X' Won!
    |O|O| |
    |X| |O|
    |X|X|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |O| |
    | | | |
    
    | |X| |
    | |O| |
    | | | |
    
    | |X| |
    | |O| |
    |O| | |
    
    | |X| |
    | |O|X|
    |O| | |
    
    | |X| |
    | |O|X|
    |O|O| |
    
    | |X|X|
    | |O|X|
    |O|O| |
    
    | |X|X|
    |O|O|X|
    |O|O| |
    
    'X' Won!
    | |X|X|
    |O|O|X|
    |O|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |X| | |
    | | | |
    
    | | |O|
    |X| | |
    | | | |
    
    |X| |O|
    |X| | |
    | | | |
    
    |X|O|O|
    |X| | |
    | | | |
    
    |X|O|O|
    |X| | |
    | |X| |
    
    |X|O|O|
    |X| | |
    |O|X| |
    
    |X|O|O|
    |X|X| |
    |O|X| |
    
    |X|O|O|
    |X|X| |
    |O|X|O|
    
    'X' Won!
    |X|O|O|
    |X|X|X|
    |O|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | | |X|
    
    | | | |
    | | | |
    |O| |X|
    
    | | | |
    |X| | |
    |O| |X|
    
    | |O| |
    |X| | |
    |O| |X|
    
    |X|O| |
    |X| | |
    |O| |X|
    
    |X|O| |
    |X|O| |
    |O| |X|
    
    |X|O|X|
    |X|O| |
    |O| |X|
    
    'O' Won!
    |X|O|X|
    |X|O| |
    |O|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | |O|
    | | | |
    | | | |
    
    | | |O|
    | | | |
    |X| | |
    
    | | |O|
    | | |O|
    |X| | |
    
    |X| |O|
    | | |O|
    |X| | |
    
    |X| |O|
    |O| |O|
    |X| | |
    
    |X| |O|
    |O| |O|
    |X|X| |
    
    'O' Won!
    |X| |O|
    |O| |O|
    |X|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | |O|
    | | | |
    
    | | | |
    | | |O|
    | |X| |
    
    | |O| |
    | | |O|
    | |X| |
    
    | |O| |
    |X| |O|
    | |X| |
    
    | |O| |
    |X| |O|
    |O|X| |
    
    | |O| |
    |X| |O|
    |O|X|X|
    
    |O|O| |
    |X| |O|
    |O|X|X|
    
    |O|O| |
    |X|X|O|
    |O|X|X|
    
    'O' Won!
    |O|O|O|
    |X|X|O|
    |O|X|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |X| |
    | | | |
    
    |O| | |
    | |X| |
    | | | |
    
    |O| | |
    | |X| |
    | |X| |
    
    |O| | |
    | |X|O|
    | |X| |
    
    |O| | |
    | |X|O|
    |X|X| |
    
    |O| | |
    |O|X|O|
    |X|X| |
    
    'X' Won!
    |O|X| |
    |O|X|O|
    |X|X| |
    
    | | | |
    | | | |
    | | | |
    
    | |O| |
    | | | |
    | | | |
    
    | |O| |
    | | |X|
    | | | |
    
    | |O| |
    | | |X|
    |O| | |
    
    | |O|X|
    | | |X|
    |O| | |
    
    | |O|X|
    | | |X|
    |O|O| |
    
    | |O|X|
    | |X|X|
    |O|O| |
    
    |O|O|X|
    | |X|X|
    |O|O| |
    
    'X' Won!
    |O|O|X|
    |X|X|X|
    |O|O| |
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    |O| | |
    
    | |X| |
    | | | |
    |O| | |
    
    | |X|O|
    | | | |
    |O| | |
    
    | |X|O|
    |X| | |
    |O| | |
    
    | |X|O|
    |X| | |
    |O| |O|
    
    | |X|O|
    |X| | |
    |O|X|O|
    
    'O' Won!
    | |X|O|
    |X|O| |
    |O|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |O| |
    
    |X| | |
    | | | |
    | |O| |
    
    |X|O| |
    | | | |
    | |O| |
    
    |X|O| |
    | | | |
    |X|O| |
    
    |X|O| |
    | | | |
    |X|O|O|
    
    |X|O|X|
    | | | |
    |X|O|O|
    
    'O' Won!
    |X|O|X|
    | |O| |
    |X|O|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | |O|
    | | | |
    
    | | | |
    | | |O|
    | | |X|
    
    | | | |
    | |O|O|
    | | |X|
    
    | | |X|
    | |O|O|
    | | |X|
    
    | |O|X|
    | |O|O|
    | | |X|
    
    | |O|X|
    | |O|O|
    | |X|X|
    
    'O' Won!
    | |O|X|
    |O|O|O|
    | |X|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |O| | |
    | | | |
    
    | | |X|
    |O| | |
    | | | |
    
    | | |X|
    |O| | |
    | | |O|
    
    | | |X|
    |O| | |
    |X| |O|
    
    |O| |X|
    |O| | |
    |X| |O|
    
    |O| |X|
    |O| | |
    |X|X|O|
    
    'O' Won!
    |O| |X|
    |O|O| |
    |X|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |O| |
    | | | |
    
    |X| | |
    | |O| |
    | | | |
    
    |X| |O|
    | |O| |
    | | | |
    
    |X|X|O|
    | |O| |
    | | | |
    
    |X|X|O|
    | |O| |
    | |O| |
    
    |X|X|O|
    |X|O| |
    | |O| |
    
    |X|X|O|
    |X|O|O|
    | |O| |
    
    'X' Won!
    |X|X|O|
    |X|O|O|
    |X|O| |
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |X| | |
    | | | |
    
    | | | |
    |X| | |
    |O| | |
    
    | | | |
    |X| | |
    |O|X| |
    
    |O| | |
    |X| | |
    |O|X| |
    
    |O|X| |
    |X| | |
    |O|X| |
    
    |O|X| |
    |X|O| |
    |O|X| |
    
    |O|X| |
    |X|O| |
    |O|X|X|
    
    |O|X| |
    |X|O|O|
    |O|X|X|
    
    It's a stalemate!
    |O|X|X|
    |X|O|O|
    |O|X|X|
    
    | | | |
    | | | |
    | | | |
    
    |O| | |
    | | | |
    | | | |
    
    |O| | |
    | | | |
    |X| | |
    
    |O| | |
    | | | |
    |X| |O|
    
    |O| | |
    |X| | |
    |X| |O|
    
    'O' Won!
    |O| | |
    |X|O| |
    |X| |O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |X| |
    
    | | | |
    | |O| |
    | |X| |
    
    | | |X|
    | |O| |
    | |X| |
    
    |O| |X|
    | |O| |
    | |X| |
    
    |O| |X|
    |X|O| |
    | |X| |
    
    |O| |X|
    |X|O|O|
    | |X| |
    
    |O|X|X|
    |X|O|O|
    | |X| |
    
    |O|X|X|
    |X|O|O|
    |O|X| |
    
    It's a stalemate!
    |O|X|X|
    |X|O|O|
    |O|X|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | |X|
    | | | |
    
    | | | |
    | | |X|
    | | |O|
    
    | | | |
    | | |X|
    |X| |O|
    
    | | | |
    | | |X|
    |X|O|O|
    
    | | | |
    | |X|X|
    |X|O|O|
    
    | | |O|
    | |X|X|
    |X|O|O|
    
    | |X|O|
    | |X|X|
    |X|O|O|
    
    |O|X|O|
    | |X|X|
    |X|O|O|
    
    'X' Won!
    |O|X|O|
    |X|X|X|
    |X|O|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |O| |
    | | | |
    
    | | | |
    | |O| |
    | | |X|
    
    | |O| |
    | |O| |
    | | |X|
    
    | |O| |
    | |O|X|
    | | |X|
    
    'O' Won!
    | |O| |
    | |O|X|
    | |O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | |X|
    | | | |
    
    | | |O|
    | | |X|
    | | | |
    
    | | |O|
    | | |X|
    | | |X|
    
    | |O|O|
    | | |X|
    | | |X|
    
    | |O|O|
    |X| |X|
    | | |X|
    
    | |O|O|
    |X| |X|
    |O| |X|
    
    | |O|O|
    |X| |X|
    |O|X|X|
    
    'O' Won!
    | |O|O|
    |X|O|X|
    |O|X|X|
    
    | | | |
    | | | |
    | | | |
    
    | | |O|
    | | | |
    | | | |
    
    | | |O|
    | | | |
    |X| | |
    
    | |O|O|
    | | | |
    |X| | |
    
    | |O|O|
    | | | |
    |X|X| |
    
    'O' Won!
    |O|O|O|
    | | | |
    |X|X| |
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |X| |
    | | | |
    
    | | | |
    | |X| |
    | | |O|
    
    |X| | |
    | |X| |
    | | |O|
    
    |X| | |
    | |X| |
    |O| |O|
    
    |X| |X|
    | |X| |
    |O| |O|
    
    |X| |X|
    | |X|O|
    |O| |O|
    
    'X' Won!
    |X|X|X|
    | |X|O|
    |O| |O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | | |X|
    
    | | | |
    | | |O|
    | | |X|
    
    | | | |
    | |X|O|
    | | |X|
    
    | | | |
    | |X|O|
    | |O|X|
    
    'X' Won!
    |X| | |
    | |X|O|
    | |O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | | |X|
    
    | | | |
    | | |O|
    | | |X|
    
    | | | |
    |X| |O|
    | | |X|
    
    | | |O|
    |X| |O|
    | | |X|
    
    | | |O|
    |X| |O|
    |X| |X|
    
    |O| |O|
    |X| |O|
    |X| |X|
    
    'X' Won!
    |O| |O|
    |X| |O|
    |X|X|X|
    
    | | | |
    | | | |
    | | | |
    
    |O| | |
    | | | |
    | | | |
    
    |O| | |
    | |X| |
    | | | |
    
    |O| |O|
    | |X| |
    | | | |
    
    |O| |O|
    | |X| |
    | |X| |
    
    |O| |O|
    | |X| |
    |O|X| |
    
    'X' Won!
    |O|X|O|
    | |X| |
    |O|X| |
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    |O| | |
    
    | |X| |
    | | | |
    |O| | |
    
    | |X| |
    | | |O|
    |O| | |
    
    |X|X| |
    | | |O|
    |O| | |
    
    |X|X| |
    |O| |O|
    |O| | |
    
    |X|X| |
    |O|X|O|
    |O| | |
    
    |X|X| |
    |O|X|O|
    |O| |O|
    
    'X' Won!
    |X|X|X|
    |O|X|O|
    |O| |O|
    
    | | | |
    | | | |
    | | | |
    
    | | |X|
    | | | |
    | | | |
    
    | | |X|
    | | | |
    | | |O|
    
    | | |X|
    | | | |
    | |X|O|
    
    | | |X|
    | | |O|
    | |X|O|
    
    |X| |X|
    | | |O|
    | |X|O|
    
    |X| |X|
    | |O|O|
    | |X|O|
    
    |X| |X|
    |X|O|O|
    | |X|O|
    
    |X|O|X|
    |X|O|O|
    | |X|O|
    
    'X' Won!
    |X|O|X|
    |X|O|O|
    |X|X|O|
    
    | | | |
    | | | |
    | | | |
    
    |O| | |
    | | | |
    | | | |
    
    |O| |X|
    | | | |
    | | | |
    
    |O| |X|
    | | | |
    | | |O|
    
    |O| |X|
    | | | |
    |X| |O|
    
    |O| |X|
    |O| | |
    |X| |O|
    
    |O| |X|
    |O| | |
    |X|X|O|
    
    'O' Won!
    |O| |X|
    |O|O| |
    |X|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | |X|
    | | | |
    
    | |O| |
    | | |X|
    | | | |
    
    | |O| |
    | | |X|
    | | |X|
    
    | |O|O|
    | | |X|
    | | |X|
    
    | |O|O|
    | |X|X|
    | | |X|
    
    | |O|O|
    | |X|X|
    |O| |X|
    
    'X' Won!
    |X|O|O|
    | |X|X|
    |O| |X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | |O|
    | | | |
    
    | | | |
    |X| |O|
    | | | |
    
    | | | |
    |X|O|O|
    | | | |
    
    | | | |
    |X|O|O|
    |X| | |
    
    | | |O|
    |X|O|O|
    |X| | |
    
    | | |O|
    |X|O|O|
    |X| |X|
    
    | | |O|
    |X|O|O|
    |X|O|X|
    
    | |X|O|
    |X|O|O|
    |X|O|X|
    
    It's a stalemate!
    |O|X|O|
    |X|O|O|
    |X|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |X| |
    
    |O| | |
    | | | |
    | |X| |
    
    |O|X| |
    | | | |
    | |X| |
    
    |O|X| |
    | | |O|
    | |X| |
    
    |O|X| |
    | | |O|
    | |X|X|
    
    |O|X| |
    |O| |O|
    | |X|X|
    
    |O|X|X|
    |O| |O|
    | |X|X|
    
    'O' Won!
    |O|X|X|
    |O|O|O|
    | |X|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |X| |
    
    |O| | |
    | | | |
    | |X| |
    
    |O| | |
    |X| | |
    | |X| |
    
    |O| |O|
    |X| | |
    | |X| |
    
    |O| |O|
    |X| | |
    | |X|X|
    
    'O' Won!
    |O|O|O|
    |X| | |
    | |X|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | |O|
    | | | |
    
    | | | |
    | | |O|
    |X| | |
    
    |O| | |
    | | |O|
    |X| | |
    
    |O| |X|
    | | |O|
    |X| | |
    
    |O| |X|
    | | |O|
    |X| |O|
    
    'X' Won!
    |O| |X|
    | |X|O|
    |X| |O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | |X|
    | | | |
    
    |O| | |
    | | |X|
    | | | |
    
    |O| | |
    |X| |X|
    | | | |
    
    |O| | |
    |X| |X|
    | | |O|
    
    |O| |X|
    |X| |X|
    | | |O|
    
    |O| |X|
    |X| |X|
    |O| |O|
    
    'X' Won!
    |O| |X|
    |X|X|X|
    |O| |O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | | |X|
    
    | | | |
    | | |O|
    | | |X|
    
    | |X| |
    | | |O|
    | | |X|
    
    | |X|O|
    | | |O|
    | | |X|
    
    |X|X|O|
    | | |O|
    | | |X|
    
    |X|X|O|
    | |O|O|
    | | |X|
    
    |X|X|O|
    |X|O|O|
    | | |X|
    
    |X|X|O|
    |X|O|O|
    | |O|X|
    
    'X' Won!
    |X|X|O|
    |X|O|O|
    |X|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | |O|
    | | | |
    
    | | | |
    |X| |O|
    | | | |
    
    | | | |
    |X| |O|
    |O| | |
    
    |X| | |
    |X| |O|
    |O| | |
    
    |X| | |
    |X|O|O|
    |O| | |
    
    |X| | |
    |X|O|O|
    |O| |X|
    
    |X|O| |
    |X|O|O|
    |O| |X|
    
    |X|O|X|
    |X|O|O|
    |O| |X|
    
    'O' Won!
    |X|O|X|
    |X|O|O|
    |O|O|X|
    
    | | | |
    | | | |
    | | | |
    
    |X| | |
    | | | |
    | | | |
    
    |X|O| |
    | | | |
    | | | |
    
    |X|O| |
    | | | |
    | | |X|
    
    |X|O| |
    | | | |
    |O| |X|
    
    |X|O| |
    | | | |
    |O|X|X|
    
    |X|O| |
    | |O| |
    |O|X|X|
    
    |X|O| |
    | |O|X|
    |O|X|X|
    
    'O' Won!
    |X|O|O|
    | |O|X|
    |O|X|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |O| | |
    | | | |
    
    | | | |
    |O| |X|
    | | | |
    
    | | | |
    |O|O|X|
    | | | |
    
    | | | |
    |O|O|X|
    | | |X|
    
    |O| | |
    |O|O|X|
    | | |X|
    
    |O| | |
    |O|O|X|
    |X| |X|
    
    |O| |O|
    |O|O|X|
    |X| |X|
    
    'X' Won!
    |O| |O|
    |O|O|X|
    |X|X|X|
    
    | | | |
    | | | |
    | | | |
    
    | | |X|
    | | | |
    | | | |
    
    | | |X|
    | | | |
    | |O| |
    
    | | |X|
    | |X| |
    | |O| |
    
    | | |X|
    | |X|O|
    | |O| |
    
    | |X|X|
    | |X|O|
    | |O| |
    
    | |X|X|
    | |X|O|
    | |O|O|
    
    | |X|X|
    |X|X|O|
    | |O|O|
    
    'O' Won!
    | |X|X|
    |X|X|O|
    |O|O|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |X| |
    | | | |
    
    | | | |
    |O|X| |
    | | | |
    
    | | |X|
    |O|X| |
    | | | |
    
    | | |X|
    |O|X| |
    | | |O|
    
    | |X|X|
    |O|X| |
    | | |O|
    
    |O|X|X|
    |O|X| |
    | | |O|
    
    |O|X|X|
    |O|X|X|
    | | |O|
    
    'O' Won!
    |O|X|X|
    |O|X|X|
    |O| |O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | | |X|
    
    | | | |
    | | | |
    | |O|X|
    
    | |X| |
    | | | |
    | |O|X|
    
    |O|X| |
    | | | |
    | |O|X|
    
    |O|X| |
    | | | |
    |X|O|X|
    
    |O|X| |
    | | |O|
    |X|O|X|
    
    |O|X| |
    |X| |O|
    |X|O|X|
    
    |O|X|O|
    |X| |O|
    |X|O|X|
    
    It's a stalemate!
    |O|X|O|
    |X|X|O|
    |X|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | |O|
    | | | |
    | | | |
    
    | | |O|
    |X| | |
    | | | |
    
    | | |O|
    |X| |O|
    | | | |
    
    | |X|O|
    |X| |O|
    | | | |
    
    | |X|O|
    |X| |O|
    |O| | |
    
    | |X|O|
    |X|X|O|
    |O| | |
    
    |O|X|O|
    |X|X|O|
    |O| | |
    
    'X' Won!
    |O|X|O|
    |X|X|O|
    |O|X| |
    
    | | | |
    | | | |
    | | | |
    
    | | |O|
    | | | |
    | | | |
    
    | | |O|
    | | | |
    |X| | |
    
    | | |O|
    | | | |
    |X| |O|
    
    | | |O|
    | |X| |
    |X| |O|
    
    | | |O|
    |O|X| |
    |X| |O|
    
    | | |O|
    |O|X|X|
    |X| |O|
    
    | | |O|
    |O|X|X|
    |X|O|O|
    
    | |X|O|
    |O|X|X|
    |X|O|O|
    
    It's a stalemate!
    |O|X|O|
    |O|X|X|
    |X|O|O|
    
    | | | |
    | | | |
    | | | |
    
    |X| | |
    | | | |
    | | | |
    
    |X|O| |
    | | | |
    | | | |
    
    |X|O| |
    | | |X|
    | | | |
    
    |X|O| |
    | | |X|
    | |O| |
    
    |X|O| |
    | | |X|
    |X|O| |
    
    |X|O| |
    |O| |X|
    |X|O| |
    
    |X|O| |
    |O| |X|
    |X|O|X|
    
    |X|O|O|
    |O| |X|
    |X|O|X|
    
    'X' Won!
    |X|O|O|
    |O|X|X|
    |X|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    |O| | |
    
    | | | |
    | | |X|
    |O| | |
    
    | | | |
    | | |X|
    |O| |O|
    
    | | |X|
    | | |X|
    |O| |O|
    
    |O| |X|
    | | |X|
    |O| |O|
    
    |O| |X|
    | | |X|
    |O|X|O|
    
    |O|O|X|
    | | |X|
    |O|X|O|
    
    |O|O|X|
    |X| |X|
    |O|X|O|
    
    'O' Won!
    |O|O|X|
    |X|O|X|
    |O|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | |X|
    | | | |
    | | | |
    
    | | |X|
    |O| | |
    | | | |
    
    |X| |X|
    |O| | |
    | | | |
    
    |X| |X|
    |O| | |
    | | |O|
    
    |X| |X|
    |O| | |
    | |X|O|
    
    |X| |X|
    |O|O| |
    | |X|O|
    
    |X| |X|
    |O|O|X|
    | |X|O|
    
    |X| |X|
    |O|O|X|
    |O|X|O|
    
    'X' Won!
    |X|X|X|
    |O|O|X|
    |O|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | |O|
    | | | |
    | | | |
    
    | | |O|
    | | |X|
    | | | |
    
    |O| |O|
    | | |X|
    | | | |
    
    |O|X|O|
    | | |X|
    | | | |
    
    |O|X|O|
    | | |X|
    |O| | |
    
    |O|X|O|
    | | |X|
    |O| |X|
    
    'O' Won!
    |O|X|O|
    |O| |X|
    |O| |X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |X| |
    
    | | | |
    | |O| |
    | |X| |
    
    | | | |
    | |O|X|
    | |X| |
    
    | |O| |
    | |O|X|
    | |X| |
    
    |X|O| |
    | |O|X|
    | |X| |
    
    |X|O|O|
    | |O|X|
    | |X| |
    
    |X|O|O|
    |X|O|X|
    | |X| |
    
    |X|O|O|
    |X|O|X|
    | |X|O|
    
    'X' Won!
    |X|O|O|
    |X|O|X|
    |X|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    |O| | |
    
    | | |X|
    | | | |
    |O| | |
    
    | | |X|
    | | |O|
    |O| | |
    
    | | |X|
    | | |O|
    |O|X| |
    
    |O| |X|
    | | |O|
    |O|X| |
    
    |O| |X|
    | | |O|
    |O|X|X|
    
    |O|O|X|
    | | |O|
    |O|X|X|
    
    |O|O|X|
    |X| |O|
    |O|X|X|
    
    It's a stalemate!
    |O|O|X|
    |X|O|O|
    |O|X|X|
    
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
    | |O| |
    |O| |X|
    
    | | |X|
    | |O| |
    |O| |X|
    
    | |O|X|
    | |O| |
    |O| |X|
    
    |X|O|X|
    | |O| |
    |O| |X|
    
    |X|O|X|
    | |O|O|
    |O| |X|
    
    |X|O|X|
    |X|O|O|
    |O| |X|
    
    'O' Won!
    |X|O|X|
    |X|O|O|
    |O|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    |X| | |
    
    |O| | |
    | | | |
    |X| | |
    
    |O| |X|
    | | | |
    |X| | |
    
    |O| |X|
    |O| | |
    |X| | |
    
    |O| |X|
    |O| |X|
    |X| | |
    
    |O|O|X|
    |O| |X|
    |X| | |
    
    'X' Won!
    |O|O|X|
    |O|X|X|
    |X| | |
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |O| | |
    | | | |
    
    |X| | |
    |O| | |
    | | | |
    
    |X|O| |
    |O| | |
    | | | |
    
    |X|O| |
    |O| | |
    | | |X|
    
    |X|O| |
    |O| |O|
    | | |X|
    
    |X|O| |
    |O| |O|
    |X| |X|
    
    'O' Won!
    |X|O| |
    |O|O|O|
    |X| |X|
    
    | | | |
    | | | |
    | | | |
    
    | |X| |
    | | | |
    | | | |
    
    |O|X| |
    | | | |
    | | | |
    
    |O|X| |
    |X| | |
    | | | |
    
    |O|X| |
    |X| | |
    |O| | |
    
    |O|X| |
    |X| | |
    |O|X| |
    
    |O|X| |
    |X| | |
    |O|X|O|
    
    |O|X| |
    |X| |X|
    |O|X|O|
    
    'O' Won!
    |O|X| |
    |X|O|X|
    |O|X|O|
    
    | | | |
    | | | |
    | | | |
    
    |O| | |
    | | | |
    | | | |
    
    |O| | |
    | |X| |
    | | | |
    
    |O| |O|
    | |X| |
    | | | |
    
    |O| |O|
    |X|X| |
    | | | |
    
    |O| |O|
    |X|X| |
    | |O| |
    
    |O| |O|
    |X|X| |
    |X|O| |
    
    |O| |O|
    |X|X| |
    |X|O|O|
    
    |O|X|O|
    |X|X| |
    |X|O|O|
    
    'O' Won!
    |O|X|O|
    |X|X|O|
    |X|O|O|
    
    | | | |
    | | | |
    | | | |
    
    | | |O|
    | | | |
    | | | |
    
    | | |O|
    | | | |
    | | |X|
    
    | |O|O|
    | | | |
    | | |X|
    
    | |O|O|
    |X| | |
    | | |X|
    
    | |O|O|
    |X| | |
    |O| |X|
    
    |X|O|O|
    |X| | |
    |O| |X|
    
    |X|O|O|
    |X| | |
    |O|O|X|
    
    'X' Won!
    |X|O|O|
    |X|X| |
    |O|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | |X| |
    | | | |
    | | | |
    
    | |X| |
    | |O| |
    | | | |
    
    | |X|X|
    | |O| |
    | | | |
    
    | |X|X|
    | |O| |
    |O| | |
    
    | |X|X|
    | |O| |
    |O|X| |
    
    | |X|X|
    | |O| |
    |O|X|O|
    
    | |X|X|
    | |O|X|
    |O|X|O|
    
    | |X|X|
    |O|O|X|
    |O|X|O|
    
    'X' Won!
    |X|X|X|
    |O|O|X|
    |O|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |X| |
    
    | | | |
    | |O| |
    | |X| |
    
    | |X| |
    | |O| |
    | |X| |
    
    | |X|O|
    | |O| |
    | |X| |
    
    | |X|O|
    |X|O| |
    | |X| |
    
    | |X|O|
    |X|O|O|
    | |X| |
    
    | |X|O|
    |X|O|O|
    |X|X| |
    
    'O' Won!
    | |X|O|
    |X|O|O|
    |X|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | |O| |
    | | | |
    | | | |
    
    | |O| |
    |X| | |
    | | | |
    
    | |O| |
    |X| | |
    | | |O|
    
    | |O| |
    |X| | |
    |X| |O|
    
    | |O| |
    |X|O| |
    |X| |O|
    
    | |O| |
    |X|O| |
    |X|X|O|
    
    | |O|O|
    |X|O| |
    |X|X|O|
    
    'X' Won!
    |X|O|O|
    |X|O| |
    |X|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |O| |
    
    | | | |
    | | | |
    | |O|X|
    
    | | | |
    | |O| |
    | |O|X|
    
    | | | |
    |X|O| |
    | |O|X|
    
    |O| | |
    |X|O| |
    | |O|X|
    
    |O| | |
    |X|O|X|
    | |O|X|
    
    'O' Won!
    |O|O| |
    |X|O|X|
    | |O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |O| | |
    | | | |
    
    | | | |
    |O| | |
    |X| | |
    
    | | | |
    |O| | |
    |X|O| |
    
    | | | |
    |O| | |
    |X|O|X|
    
    |O| | |
    |O| | |
    |X|O|X|
    
    |O|X| |
    |O| | |
    |X|O|X|
    
    |O|X| |
    |O| |O|
    |X|O|X|
    
    |O|X| |
    |O|X|O|
    |X|O|X|
    
    It's a stalemate!
    |O|X|O|
    |O|X|O|
    |X|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | |O| |
    | | | |
    | | | |
    
    | |O| |
    | | | |
    | |X| |
    
    | |O| |
    | | | |
    | |X|O|
    
    | |O|X|
    | | | |
    | |X|O|
    
    | |O|X|
    | | | |
    |O|X|O|
    
    | |O|X|
    |X| | |
    |O|X|O|
    
    | |O|X|
    |X| |O|
    |O|X|O|
    
    |X|O|X|
    |X| |O|
    |O|X|O|
    
    It's a stalemate!
    |X|O|X|
    |X|O|O|
    |O|X|O|
    
    | | | |
    | | | |
    | | | |
    
    |X| | |
    | | | |
    | | | |
    
    |X| | |
    | |O| |
    | | | |
    
    |X| | |
    |X|O| |
    | | | |
    
    |X| | |
    |X|O| |
    | | |O|
    
    |X| |X|
    |X|O| |
    | | |O|
    
    |X| |X|
    |X|O| |
    | |O|O|
    
    'X' Won!
    |X|X|X|
    |X|O| |
    | |O|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |X| | |
    | | | |
    
    | | |O|
    |X| | |
    | | | |
    
    | | |O|
    |X| | |
    |X| | |
    
    |O| |O|
    |X| | |
    |X| | |
    
    |O| |O|
    |X|X| |
    |X| | |
    
    |O| |O|
    |X|X| |
    |X|O| |
    
    |O| |O|
    |X|X| |
    |X|O|X|
    
    |O| |O|
    |X|X|O|
    |X|O|X|
    
    It's a stalemate!
    |O|X|O|
    |X|X|O|
    |X|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |O| |
    
    | | | |
    |X| | |
    | |O| |
    
    | | |O|
    |X| | |
    | |O| |
    
    | |X|O|
    |X| | |
    | |O| |
    
    | |X|O|
    |X|O| |
    | |O| |
    
    | |X|O|
    |X|O|X|
    | |O| |
    
    'O' Won!
    | |X|O|
    |X|O|X|
    |O|O| |
    
    | | | |
    | | | |
    | | | |
    
    |X| | |
    | | | |
    | | | |
    
    |X| | |
    | | | |
    |O| | |
    
    |X| | |
    | |X| |
    |O| | |
    
    |X| | |
    | |X|O|
    |O| | |
    
    |X| | |
    |X|X|O|
    |O| | |
    
    |X| |O|
    |X|X|O|
    |O| | |
    
    'X' Won!
    |X| |O|
    |X|X|O|
    |O| |X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |X| |
    
    | | |O|
    | | | |
    | |X| |
    
    | | |O|
    | | | |
    |X|X| |
    
    | | |O|
    | | |O|
    |X|X| |
    
    | | |O|
    | |X|O|
    |X|X| |
    
    | | |O|
    |O|X|O|
    |X|X| |
    
    'X' Won!
    | | |O|
    |O|X|O|
    |X|X|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |X| |
    
    | | | |
    | | | |
    |O|X| |
    
    | | |X|
    | | | |
    |O|X| |
    
    | | |X|
    | | | |
    |O|X|O|
    
    | | |X|
    | |X| |
    |O|X|O|
    
    |O| |X|
    | |X| |
    |O|X|O|
    
    |O| |X|
    |X|X| |
    |O|X|O|
    
    |O| |X|
    |X|X|O|
    |O|X|O|
    
    'X' Won!
    |O|X|X|
    |X|X|O|
    |O|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | | |O|
    
    | | | |
    | |X| |
    | | |O|
    
    | | | |
    | |X|O|
    | | |O|
    
    | |X| |
    | |X|O|
    | | |O|
    
    | |X| |
    |O|X|O|
    | | |O|
    
    | |X|X|
    |O|X|O|
    | | |O|
    
    | |X|X|
    |O|X|O|
    | |O|O|
    
    'X' Won!
    |X|X|X|
    |O|X|O|
    | |O|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |O| |
    
    | | | |
    | | |X|
    | |O| |
    
    | | |O|
    | | |X|
    | |O| |
    
    | | |O|
    | | |X|
    | |O|X|
    
    |O| |O|
    | | |X|
    | |O|X|
    
    |O| |O|
    | | |X|
    |X|O|X|
    
    'O' Won!
    |O|O|O|
    | | |X|
    |X|O|X|
    
    | | | |
    | | | |
    | | | |
    
    | |X| |
    | | | |
    | | | |
    
    | |X| |
    | | | |
    |O| | |
    
    | |X| |
    | |X| |
    |O| | |
    
    | |X| |
    | |X|O|
    |O| | |
    
    'X' Won!
    | |X| |
    | |X|O|
    |O|X| |
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |X| |
    | | | |
    
    | | | |
    |O|X| |
    | | | |
    
    | | | |
    |O|X| |
    |X| | |
    
    | | | |
    |O|X|O|
    |X| | |
    
    |X| | |
    |O|X|O|
    |X| | |
    
    |X| | |
    |O|X|O|
    |X| |O|
    
    |X|X| |
    |O|X|O|
    |X| |O|
    
    'O' Won!
    |X|X|O|
    |O|X|O|
    |X| |O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | |X| |
    
    | |O| |
    | | | |
    | |X| |
    
    |X|O| |
    | | | |
    | |X| |
    
    |X|O|O|
    | | | |
    | |X| |
    
    |X|O|O|
    |X| | |
    | |X| |
    
    |X|O|O|
    |X|O| |
    | |X| |
    
    |X|O|O|
    |X|O|X|
    | |X| |
    
    |X|O|O|
    |X|O|X|
    | |X|O|
    
    'X' Won!
    |X|O|O|
    |X|O|X|
    |X|X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    |O| | |
    | | | |
    
    | | | |
    |O| |X|
    | | | |
    
    | | | |
    |O| |X|
    | |O| |
    
    | | | |
    |O| |X|
    | |O|X|
    
    | | | |
    |O|O|X|
    | |O|X|
    
    'X' Won!
    | | |X|
    |O|O|X|
    | |O|X|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | | |
    | | |X|
    
    | | | |
    | | | |
    | |O|X|
    
    |X| | |
    | | | |
    | |O|X|
    
    |X| | |
    | | | |
    |O|O|X|
    
    |X|X| |
    | | | |
    |O|O|X|
    
    |X|X| |
    | | |O|
    |O|O|X|
    
    |X|X| |
    |X| |O|
    |O|O|X|
    
    |X|X|O|
    |X| |O|
    |O|O|X|
    
    'X' Won!
    |X|X|O|
    |X|X|O|
    |O|O|X|
    
    | | | |
    | | | |
    | | | |
    
    |X| | |
    | | | |
    | | | |
    
    |X| | |
    | |O| |
    | | | |
    
    |X| |X|
    | |O| |
    | | | |
    
    |X| |X|
    | |O| |
    | |O| |
    
    |X| |X|
    |X|O| |
    | |O| |
    
    |X| |X|
    |X|O| |
    |O|O| |
    
    'X' Won!
    |X|X|X|
    |X|O| |
    |O|O| |
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |O| |
    | | | |
    
    | | | |
    | |O|X|
    | | | |
    
    | | |O|
    | |O|X|
    | | | |
    
    | |X|O|
    | |O|X|
    | | | |
    
    |O|X|O|
    | |O|X|
    | | | |
    
    |O|X|O|
    | |O|X|
    |X| | |
    
    'O' Won!
    |O|X|O|
    | |O|X|
    |X| |O|
    
    | | | |
    | | | |
    | | | |
    
    |O| | |
    | | | |
    | | | |
    
    |O| |X|
    | | | |
    | | | |
    
    |O| |X|
    | |O| |
    | | | |
    
    |O|X|X|
    | |O| |
    | | | |
    
    |O|X|X|
    | |O|O|
    | | | |
    
    |O|X|X|
    | |O|O|
    | |X| |
    
    'O' Won!
    |O|X|X|
    | |O|O|
    | |X|O|
    
    | | | |
    | | | |
    | | | |
    
    | | | |
    | | |X|
    | | | |
    
    | | | |
    | |O|X|
    | | | |
    
    | | | |
    | |O|X|
    |X| | |
    
    | | | |
    | |O|X|
    |X|O| |
    
    | | |X|
    | |O|X|
    |X|O| |
    
    'O' Won!
    | |O|X|
    | |O|X|
    |X|O| |
    
    | | | |
    | | | |
    | | | |
    
    | | |X|
    | | | |
    | | | |
    
    | | |X|
    | | | |
    |O| | |
    
    | |X|X|
    | | | |
    |O| | |
    
    | |X|X|
    | | |O|
    |O| | |
    
    'X' Won!
    |X|X|X|
    | | |O|
    |O| | |
    


## 3.1 Clean Data

We will first need to organize the data into a parsable format.

### Q1

 What is the object `data` and what does it contain?

 * what are the keys of data?
 * what are the keys of each game?


```python
# inspect data below by grabbing the first key in data
# what are the three different keys within each game?
data['game 0']
```




    {'board': {1: 'X',
      2: ' ',
      3: 'O',
      4: 'X',
      5: 'X',
      6: 'O',
      7: 'O',
      8: 'X',
      9: 'O'},
     'starting player': 'X',
     'winner': 'O'}



### Q2

Using those keys, iterate through every `game` in `data` and append the board, the winner, and the starting player to separate lists. Call these lists: `boards, winners, and starters`


```python
boards = []
winners = []
starters = []
for game in data:
  # YOUR CODE HERE
```

### Q3

Make a dataframe out of the list `boards` and call it `df`. Make a series out of the list `winners`. Make a series out of the list `starters`.

Make a new column of `df` called "Winner" and set it equal to the pandas Series of the winners.

Make a new column of `df` called "Starter" and set it equal to the pandas Series of the starters.


```python
# YOUR CODE HERE
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>Winner</th>
      <th>Starter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>X</td>
      <td></td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td></td>
      <td></td>
      <td>X</td>
      <td>O</td>
      <td>O</td>
    </tr>
    <tr>
      <th>1</th>
      <td>O</td>
      <td>O</td>
      <td>X</td>
      <td>O</td>
      <td>X</td>
      <td>X</td>
      <td>O</td>
      <td>X</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
    </tr>
    <tr>
      <th>2</th>
      <td>O</td>
      <td></td>
      <td></td>
      <td>X</td>
      <td>X</td>
      <td>X</td>
      <td>X</td>
      <td>O</td>
      <td>O</td>
      <td>X</td>
      <td>X</td>
    </tr>
    <tr>
      <th>3</th>
      <td>X</td>
      <td>X</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>X</td>
      <td>X</td>
      <td>O</td>
      <td>O</td>
      <td>Stalemate</td>
      <td>O</td>
    </tr>
    <tr>
      <th>4</th>
      <td>O</td>
      <td>O</td>
      <td>X</td>
      <td>O</td>
      <td>O</td>
      <td></td>
      <td>X</td>
      <td>X</td>
      <td>X</td>
      <td>X</td>
      <td>O</td>
    </tr>
  </tbody>
</table>
</div>



## 3.2 Inferential Analysis

We're going to use Bayes Rule or Bayesian Inference to make a probability of winning based on positions of the board. The formula is:

$ P(A|B) = \frac{P(B|A) * P(A)}{P(B)} = \frac{P(A \cap B)}{P(B)}$

Where $\cap$ is the intersection of $A$ and $B$.

The example we will use is the following: _what is the probability of 'O' being the winner, given that they've played the center piece._

$B$ = 'O' played the center piece

$A$ = 'O' won the game



So what is probability? We will define it in terms of frequencies. So if we are for instance asking what is the probability of player 'O' being in the center piece, it would be defined as:

$ P(B) = \frac{|O_c|} {|O_c| + |X_c| + |empty|}$

Where the pipes, `| |`, or cardinality represent the count of the indicated observation or set. In this case $O_c$ (O being in the center) and $X_c$ (X being in the center).



```python
Oc_Xc_empty = df[5].value_counts().sum()
Oc_Xc_empty
```




    1000




```python
# example of assessing the probability of B, O playing the center piece
player = 'O'
Oc = (df[5] == player).value_counts()
Oc_Xc_empty = df[5].value_counts().sum()

Oc/Oc_Xc_empty
```




    False    0.577
    True     0.423
    Name: 5, dtype: float64




```python
# we can also clean this up and replace the denominator with the whole
# observation space (which is just the total number of games, df.shape[0]).
# example of assesing probabiliy of A

(df['Winner'] == 'O').value_counts()/df.shape[0]
```




    False    0.571
    True     0.429
    Name: Winner, dtype: float64




The $P(B|A) * P(A)$ is the intersection of $B$ and $A$. The intersection is defined as the two events occuring together.
Continuing with the example, the probablity of 'O' playing the center piece AND 'O' being the winner is _the number of times these observations occured together divided by the whole observation space_:


```python
# in this view, the total times A and B occured together is 247
player = 'O'
df.loc[(df['Winner'] == player) & (df[5] == player)].shape[0]
```




    247




```python
# the total observation space is 1000 (1000 games)
df.shape[0]
```




    1000



And so we get:

$P(B|A) * P(A) = \frac{247} {1000} = 0.247 $

In code:


```python
df.loc[(df['Winner'] == player) & (df[5] == player)].shape[0]/df.shape[0]
```




    0.247



### 3.2.1 Behavioral Analysis of the Winner

#### Q4

define the 3 different board piece types and label them `middle`, `side`, and `corner`. Middle should be an int and the other two should be lists.


```python
# define the 3 different board piece types

# middle = 
# side = 
# corner = 
```

#### 3.2.1.1 What is the probability of winning after playing the middle piece?

#### Q5


```python
# A intersect B: X played middle and X won / tot games
# B: X played middle / tot games
player = 'X'

# define the intersection of A AND B, A_B
# A_B = 

# define prob B
# B = 

# return A_B over B (The prob B given A)
A_B / B
```




    0.5732758620689655



#### Q6


```python
# A intersect B: X played middle and X won / tot games
# B: X played middle / tot games
player = 'O'
# define the intersection of A AND B, A_B
# A_B = 

# define prob B
# B = 

# return A_B over B (The prob B given A)
A_B / B
```




    0.5839243498817968



#### 3.2.1.2 What is the probability of winning after playing a side piece?

#### Q7


```python
# A intersect B: O played side and O won / tot games
# B: O played side / tot games
player = 'O'

A_B = df.loc[(df[side].T.apply(lambda x: player in x.values)) &
       (df['Winner'] == player)].shape[0] / df.shape[0]
B = df.loc[(df[side].T.apply(lambda x: player in x.values))].shape[0] /\
        df.shape[0]

A_B / B
```




    0.4158609451385117




```python
# A intersect B: X played side and X won / tot games
# B: X played side / tot games

# player = # SET PLAYER

# A_B = df.loc[(df[<SET PIECE>].T.apply(lambda x: player in x.values)) &
#        (df['Winner'] == player)].shape[0] / df.shape[0]
# B = df.loc[(df[<SET PIECE>].T.apply(lambda x: player in x.values))].shape[0] /\
#         df.shape[0]

A_B / B
```




    0.38845460012026456



#### 3.2.1.3 What is the probability of winning after playing a corner piece?

### Q8


```python
# A intersect B: O played corner and O won / tot games
# B: O played corner / tot games

# player = # SET PLAYER

# A_B = df.loc[(df[<SET PIECE>].T.apply(lambda x: player in x.values)) &
#        (df['Winner'] == player)].shape[0] / df.shape[0]
# B = df.loc[(df[<SET PIECE>].T.apply(lambda x: player in x.values))].shape[0] /\
#         df.shape[0]

A_B / B
```




    0.4779116465863454



### Q9


```python
# A intersect B: X played corner and X won / tot games
# B: X played corner / tot games

# player = # SET PLAYER

# A_B = df.loc[(df[<SET PIECE>].T.apply(lambda x: player in x.values)) &
#        (df['Winner'] == player)].shape[0] / df.shape[0]
# B = df.loc[(df[<SET PIECE>].T.apply(lambda x: player in x.values))].shape[0] /\
#         df.shape[0]

A_B / B
```




    0.47386964180857316



Are these results surprising to you? Why? This [resource](https://www.cs.jhu.edu/~jorgev/cs106/ttt.pdf) may be illustrative.

## 3.3 Improving the Analysis

In this analysis, we only tracked what moves were made, not the order they were made in. It really limited our assessment! How might we change our recording of the games to track order of moves as well? Do we need to track all the moves or just the first and the winner? 
