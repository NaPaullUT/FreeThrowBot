from turtle import shape
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
import sklearn.pipeline
import sklearn.preprocessing

class StateActionFeatureVectorWithRBF():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:np.array,
                 num_rbfs:np.array,
                 ):#tile_width:np.array

        n_actions = np.prod(num_actions)
        s_dim = state_high.size
        assert s_dim == 4

        self.num_rbfs = num_rbfs
        s_width = state_high-state_low
        width = s_width/(num_rbfs-1)
        self.num_ind = np.prod(num_rbfs)

        rbf_sigma = width[0] / 2.
        self.rbf_den = 2 * rbf_sigma ** 2


        c = []
        for i in range(s_dim):
            n = num_rbfs[i]
            idx = np.arange(0,n)
            w = width[i]
            l = state_low[i]
            c.append(idx*w+l)
        self.centers = np.array(np.meshgrid(c[0],c[1],c[2],c[3])).T.reshape(-1,s_dim)

        self.d=self.num_ind*s_dim

    def phi(self,_state):
        _phi = np.zeros(self.num_ind)
        for _k in range(self.num_ind):
            _phi[_k] = np.exp(-np.linalg.norm(_state - self.centers[_k, :]) ** 2 / self.rbf_den)
        return _phi

    def __call__(self, s, done, a):
        if done:
            return np.zeros(self.num_ind)
        return self.phi(s)

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.num_ind

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithRBF,
    num_episode:int,
    w=None
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """
    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        if np.random.rand() < epsilon:
            idx = np.random.choice(a_space.shape[0])
            return a_space[idx]
        else:
            Q = [np.dot(w[:,int(np.where((a_space == a).all(axis=1))[0])], X(s,done,a)) for a in a_space]
            return a_space[np.argmax(Q)]

    nA = [np.arange(a) for a in env.action_space.nvec]
    a_space = np.array(np.meshgrid(nA[0],nA[1],nA[2])).T.reshape(-1,3)
    n_actions = np.prod(env.action_space.nvec)
    if w is None:
        w = np.zeros((X.feature_vector_len())*n_actions).reshape(-1,n_actions)

    for y in tqdm(range(num_episode)):
        s, poss = env.reset()
        done=False
        if(poss):
            a = epsilon_greedy_policy(s,done,w,epsilon=0.1)
        else:
            a=[0,0,0]
        x=X(s,done,a)
        z=np.zeros(x.size)
        q_old = 0
        while not done:
            s, poss, r, done, info =env.step(a)
            if poss:
                next_a = epsilon_greedy_policy(s,done,w)
            else:
                next_a=[0,0,0]
            next_x=X(s,done,next_a)

            w_col = int(np.where((a_space == a).all(axis=1))[0])
            w_col_next = int(np.where((a_space == next_a).all(axis=1))[0])
            q = np.dot(w[:,w_col],x)
            next_q=np.dot(w[:,w_col_next],next_x)

            tde = r + gamma*next_q - q

            z = gamma*lam*z + (1-alpha*gamma*lam*np.dot(z,x))*x
            w[:,w_col] = w[:,w_col] + alpha*(tde+q-q_old)*z - alpha*(q-q_old)*x

            q_old = next_q
            x = next_x
            a = next_a

    return w