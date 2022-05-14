from this import d
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
                 num_actions:int,
                 num_rbfs:np.array,
                 ):#tile_width:np.array

        n_actions = np.prod(num_actions)
        assert n_actions==len(num_rbfs)

        s_dim = state_high.size
        assert self.s_dim == 4

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

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        idx = np.arange(0,num_tilings)
        self.start_pos = -1*(np.matmul(np.transpose(idx[None]),tile_width[None])/num_tilings-state_low)

        self.tile_width=tile_width
        
        self.num_tiles = np.ceil((state_high-state_low)/tile_width).astype(int)+1
        self.num_tilings=num_tilings
        self.d=np.prod(num_actions)*num_tilings*np.prod(self.num_tiles)

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.d

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        if done:
            return np.zeros(self.d)

        coor = ((s-self.start_pos)/self.tile_width).astype(int)
        action_offset_1=np.prod(self.num_tiles)*self.num_tilings*(a[0])
        action_offset_2=np.prod(self.num_tiles)*self.num_tilings*(a[0])*(a[1])
        action_offset_3=np.prod(self.num_tiles)*self.num_tilings*(a[0])*(a[2])*(a[1])
        action_offset=+action_offset_1+action_offset_2+action_offset_3
        idx = np.sum(coor[:]*np.arange(self.num_tiles.size),axis=1)+(np.prod(self.num_tiles)*np.arange(self.num_tilings))+action_offset
        x = np.zeros(self.d)
        x[idx] = 1
        return x

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """
    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = [np.arange(a) for a in env.action_space.nvec]
        a_space = np.array(np.meshgrid(nA[0],nA[1],nA[2])).T.reshape(-1,3)
        if np.random.rand() < epsilon:
            idx = np.random.choice(a_space.shape[0])
            return a_space[idx]
        else:
            Q = [np.dot(w, X(s,done,a)) for a in a_space]
            return a_space[np.argmax(Q)]

    w = np.zeros((X.feature_vector_len()))

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

            q = np.dot(w,x)
            next_q=np.dot(w,next_x)

            tde = r + gamma*next_q - q

            z = gamma*lam*z + (1-alpha*gamma*lam*np.dot(z,x))*x
            w = w + alpha*(tde+q-q_old)*z - alpha*(q-q_old)*x

            q_old = next_q
            x = next_x
            a = next_a

    return w