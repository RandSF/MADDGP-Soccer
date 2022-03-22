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
    def __init__(self,gamma,lr,theta=0.05):
        self.reward=np.zeros((128,1))
        self.reward[3:129:8]=-100
        self.reward[7:129:8]=-100
        self.reward[64:72]=100
        self.reward[96:104]=100
        
        self.convergence= False
        self.gamma=gamma
        self.lr=lr
        self.theta=theta
        self.q=np.zeros((128,5))
    
    def act(self,s):
        s=8*s[0]+1*s[1]+64*s[2]
        s_a=self.q[s,:]
        a=s_a.argmax()
        return a
        
    
    def learn(self,s, action, opponentAction, s_, reward, opponent_reward,done):
        # if self.convergence:
        #     return
        s=8*s[0]+1*s[1]+64*s[2]
        q_cur=self.q[s][action]
        s_=8*s_[0]+1*s_[1]+64*s_[2]
        q_new=self.reward[s_]+self.gamma * max(self.q[s_])
        delta=(q_new-q_cur)
        self.q[s][action]+=self.lr * delta
        # if delta < abs(delta):
        #     print("QLearnign Converged!")
        #     self.convergence= True
            
            

    