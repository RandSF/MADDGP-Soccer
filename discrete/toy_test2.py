from soccer import SoccerEnviroment
from agents import *
from randomAgent import randomPlayAgent
from matplotlib import pyplot as plt
from game_interface import SoccerGame
import ddpgAgent2
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
LR_A=0.01
LR_C=0.03
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
DDPGAgent = ddpgAgent2.DDPGAgent(ENV, GAMMA,LR_A,LR_C,BATCH_SIZE,MEMORY_CAPACITY,TAU)
Opponent = randomPlayAgent(ENV, GAMMA)


game = SoccerGame(MAX_EPISODES, epsilon_start, epsilon_decay, epsilon_min, 
                ENV, DDPGAgent, Opponent)
win_rate,q_value,pi_certain=game.train()

for i in range(len(pi_certain)):
    pi_certain[i]=pi_certain[i].numpy().tolist()
import pandas as pd
# pd.DataFrame(win_rate).to_csv("win_rate.csv")
# pd.DataFrame(np.array(q_value)).to_csv("q_value.csv")
# pd.DataFrame(pi_certain).to_csv("pi_certain.csv")
plt.subplot(2,1,1)
plt.plot(win_rate, linewidth=0.5)
plt.subplot(2,1,2)
plt.plot(q_value, linewidth=0.5)

plt.show()
