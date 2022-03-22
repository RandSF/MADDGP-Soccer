from soccer import SoccerEnviroment
from agents import *
from randomAgent import randomPlayAgent
from matplotlib import pyplot as plt
from game_interface import SoccerGame
import ddpgAgent
import QLearning
import torch
from numpy import random
# from MarkovSoccerGame.QlearningAgent import QLearning
# from MarkovSoccerGame.foeQ import FoeQ
# from MarkovSoccerGame.friendQ import FriendQ
# from MarkovSoccerGame.ceQ import CEQ

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""seed"""

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.benchmark = True #for accelerating the running
setup_seed(114514)

"""
    This is for discrete state and discrete action, a isolate agent
    
"""

"""hyperparemetmer"""

MAX_EPISODES=10000
MAX_STEP=200
LR_A=0.001
LR_C=0.003
GAMMA=0.01
MEMORY_CAPACITY=10000
BATCH_SIZE=32
TAU=0.01 
RENDER=False
ENV = SoccerEnviroment()

# numEpisode = 100000
epsilon_start = 1
epsilon_decay = 0.99993
epsilon_min = 0.01
# gamma = 0.99
alpha_start = 1
alpha_decay = 0.99993
alpha_min = 0.001


""""""
DDPGAgent = ddpgAgent.DDPGAgent(ENV, GAMMA,LR_A,LR_C,BATCH_SIZE,MEMORY_CAPACITY,TAU)
QLearningAgent=QLearning.QLearningAgent(GAMMA,LR_A)
Opponent = randomPlayAgent(ENV, GAMMA)
Opponent2 = randomPlayAgent(ENV, GAMMA)

game1 = SoccerGame(MAX_EPISODES, epsilon_start, epsilon_decay, epsilon_min, 
                   ENV, DDPGAgent, Opponent)
# game1 = SoccerGame(MAX_EPISODES, epsilon_start, epsilon_decay, epsilon_min, 
#                    ENV, QLearningAgent, Opponent)
win_rate,q_value=game1.train()

import pandas as pd
pd.DataFrame(win_rate).to_csv("win_rate.csv")
pd.DataFrame(np.array(q_value)).to_csv("q_value.csv")
plt.subplot(1,2,1)
plt.plot(win_rate, linewidth=0.5)
plt.subplot(1,2,2)
plt.plot(q_value, linewidth=0.5)
plt.show()

# game2 = SoccerGame(MAX_EPISODES, epsilon_start, epsilon_decay, epsilon_min, 
#                    ENV, Opponent2, Opponent)

# win_rate2=game2.train()
# plt.plot(win_rate2, linewidth=0.5)
# plt.show()