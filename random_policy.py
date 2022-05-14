import numpy as np
import gym
from freethrow2 import FreeThrowEnv


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    env = FreeThrowEnv()
    env.reset()
    #env.step([np.random.choice([0,2]),np.random.choice([0,2]),0])
    while False:
        env.render()
    x=0
    while True:
        env.reset()
        env.render()
        x=0
        while(x<100):
            env.step([np.random.choice([0,1,2]),np.random.choice([0,1,2]),0])
            env.render()
            x+=1
        print("release")
        env.step([np.random.choice([0,1,2]),np.random.choice([0,1,2]),1])
        env.render()
        while (x>86):
            env.step([np.random.choice([0,1,2]),np.random.choice([0,1,2]),0])
            env.render()
            x-=1
        