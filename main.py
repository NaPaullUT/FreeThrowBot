import numpy as np
import gym
from sarsa import SarsaLambda, StateActionFeatureVectorWithRBF
from freethrow_horizontal_train import FreeThrowEnv as FTH
from freethrow_vertical_train import FreeThrowEnv as FTV
from freethrow_final import FreeThrowEnv as FTF
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_sarsa_lamda(envs):
    
    env = envs[-1]
    gamma = 1.
    print("Declared Env")
    #print(env.observation_space.high)
    X = StateActionFeatureVectorWithRBF(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.nvec,
        num_rbfs=np.array([4,4,8,8])
    )
    print("Declared Value Func")
    n_actions = np.prod(env.action_space.nvec)
    w = np.zeros((X.feature_vector_len())*n_actions).reshape(-1,n_actions)
    for env in envs:
        w = SarsaLambda(env, gamma, 0.8, 0.01, X, 500, w)
    print("Declared Feat Vec")
    def greedy_policy(s,done):
        nA = [np.arange(a) for a in env.action_space.nvec]
        a_space = np.array(np.meshgrid(nA[0],nA[1],nA[2])).T.reshape(-1,3)
        Q = [np.dot(w[:,int(np.where((a_space == a).all(axis=1))[0])], X(s,done,a)) for a in a_space]
        return a_space[np.argmax(Q)]

    def _eval(render=False):
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
    Gs = [_eval(render=False) for _ in  tqdm(range(100))]
    plt.scatter(x=np.arange(100),y=Gs)
    plt.show()
    #assert np.max(Gs) >= -110.0, 'fail to solve mountaincar'


if __name__ == "__main__":
    envs = [FTH(),FTV(),FTF()]
    test_sarsa_lamda(envs)
