import numpy as np
import gym
from sarsa import SarsaLambda, StateActionFeatureVectorWithTile
from freethrow2 import FreeThrowEnv
import matplotlib.pyplot as plt

def test_sarsa_lamda():
    
    env = FreeThrowEnv()
    gamma = 1.
    print("Declared Env")
    
    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.nvec,
        num_tilings=10,
        tile_width=np.array([.5,.5,.5,.5,.5,1.0])
    )
    print("Declared Value Func")
    w = SarsaLambda(env, gamma, 0.8, 0.01, X, 100)
    print("Declared Feat Vec")
    def greedy_policy(s,done):
        nA = [np.arange(a) for a in env.action_space.nvec]
        a_space = np.array(np.meshgrid(nA[0],nA[1],nA[2])).T.reshape(-1,3)
        Q = [np.dot(w, X(s,done,a)) for a in a_space]
        return a_space[np.argmax(Q)]

    def _eval(render=False):
        print("\tNew Eval")
        s,poss = env.reset()
        done=False
        if render: env.render()

        G = 0.
        while not done:
            if poss:
                a = greedy_policy(s,done)
            else:
                a=[0,0,0]
            s,poss,r,done,_ = env.step(a)
            if render: env.render()

            G += r
        return G
    print("Starting Eval")
    Gs = [_eval(render=True) for _ in  range(100)]
    #_eval(True)
    plt.scatter(x=np.arange(100),y=Gs)
    plt.show()
    #assert np.max(Gs) >= -110.0, 'fail to solve mountaincar'

def new_test():
    env = FreeThrowEnv()
    gamma = 1.
    print("Declared Env")
    
    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.nvec,
        num_tilings=10,
        tile_width=np.array([.5,.5,.5,.5,.5,1.0])
    )
    print("Declared Value Func")
    def greedy_policy(s,done):
        nA = [np.arange(a) for a in env.action_space.nvec]
        a_space = np.array(np.meshgrid(nA[0],nA[1],nA[2])).T.reshape(-1,3)
        Q = [X(s,done,a) for a in a_space]
        return a_space[np.argmax(Q)]

    def _eval(render=False):
        print("\tNew Eval")
        s,poss = env.reset()
        done=False
        if render: env.render()

        G = 0.
        while not done:
            if poss:
                a = greedy_policy(s,done)
            else:
                a=[0,0,0]
            s,poss,r,done,_ = env.step(a)
            if render: env.render()

            G += r
        return G
    print("Starting Eval")
    Gs = [_eval(True) for _ in  range(100)]
    #_eval(True)
    plt.scatter(x=np.arange(100),y=Gs)
    #assert np.max(Gs) >= -110.0, 'fail to solve mountaincar'

if __name__ == "__main__":
    #new_test()
    test_sarsa_lamda()