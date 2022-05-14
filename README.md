# FreeThrowBot
Reinforcement Learning Agent Learning to shoot a basketball

To run this program you will want to use main.py as your executable file.
Around Line 60 this file generates 3 environments that will be used for training, you will need to pick and choose what combination of these environments you want to use.
The three environments are as follows:
  - FTH: A low hoop with increased reward for shooting the ball far left
  - FTV: A vertical hoop with increased reward for shooting the ball higher up
  - FTF: The real free throw environment with a hoop and rims
The goal of these three environments is to incrementally encourage specific behavior in order to guide the agent towards the final hoop.
You can also change the number of RBFs in each principal state dimension in the main.py file, the state dimensions are (pos_x, pos_y, vel_x, vel_y) of the ball.
Finally you can adjust the number of episode for training in this main.py file, default is set at 500.
