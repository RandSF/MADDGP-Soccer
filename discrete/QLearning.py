from matplotlib.pyplot import thetagrids
import numpy as np
from agents import *

# for 4X2 game versus RandomAgent
"""
|S| = 128 = 8 x 8 x 2 = 0 | 000 | 000
                 AHasBall | APos| BPos
A win = 1 x 0/4 x  *  = 001|000|*** and 001|100|***={64,96}+n(n=1,...,7)=[64, 71] U [96, 103]
B win = 0 x  *  x 3/7 = 000|***|011 and 000|***|111={3 ,7}+8n(n=1,...,7)
S % 8 = 3 OR 7
"""
class QLearningAgent(ISoccerGameAgent):
    def __init__(self,gamma,lr):
        self.reward=np.zeros((128,1))
        self.reward[3:129:8]=-100
        self.reward[7:129:8]=-100
        self.reward[64:72]=100
        self.reward[96:104]=100
        self.gamma=gamma
        self.lr=lr
        self.q=np.zeros((128,5))
    
    def act(self,state):
        state=8*state[0]+state[1]+64*state[2]
        state_action=self.q[state]
        action=state_action.argmax()
        return action
    
    def learn(self,s,a,o,s_,r,ro,d):
        s=8*s[0]+s[1]+64*s[2]
        q_cur=self.q[s][a]
        s_=8*s_[0]+s_[1]+64*s_[2]
        q_new=r+self.gamma*self.q[s_].max()
        self.q[s][a]+=self.lr*(q_new-q_cur)