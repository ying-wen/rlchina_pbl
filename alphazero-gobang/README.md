# Competition_Gobang

## Environment

<img src=https://jidi-images.oss-cn-beijing.aliyuncs.com/jidi/env3.png width=600>


Check details in Jidi Homepage [](http://www.jidiai.cn/compete_detail?compete=21)


### Gobang

<b>Environment Rules:</b> 
1. This game runs on a 15*15 board. The board is initially empty.
2. Both sides of the game use black and white chess pieces and place them on the intersection of the straight and horizontal lines of the board. The first to form a five-piece line wins.

<b>Action Space: </b>Discrete, a matrix with shape 2*15, representing that the action has two dimensions and each dimension has 15 values.

<b>Observation: </b>A dictionary with keys 'state_map', 'chess_player_idx', 'board_width' and 'board_height'. The value of 'state_map' is a 15*15 *1 matrix, representing the state of the chess board. The value of the matrix is 0, 1, 2, with 0 representing an empty grid, 1 representing black chess pieces and 2 representing white chess pieces. The value of 'chess_player_idx' is the player id of the game. The values of 'board_width' and 'board_height' are width of the board and height of the board respectively.

<b>Reward: </b>The winning side rewards 100, the loser 0. If it is a draw, each side gets 50.

<b>Environment ends condition: </b>When one side wins or reaches the specified number of steps(225), the game ends.

<b>Registration: </b>Go to (http://www.jidiai.cn/compete_detail?compete=21).


---
## Dependency

>conda create -n gobang python=3.7.5

>conda activate gobang

>pip install -r requirements.txt

---

## Run a game and test submission

You can locally test your submission. At Jidi platform, we evaluate your submission as same as *run_log.py*

For example,

>python run_log.py --my_ai "random" --opponent "random"


---

## Ready to submit

Random policy --> *agents/random/submission.py*
