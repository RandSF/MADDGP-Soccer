from soccer import SoccerEnviroment
from agents import *
from randomAgent import randomPlayAgent
# from MarkovSoccerGame.QlearningAgent import QLearning
# from MarkovSoccerGame.foeQ import FoeQ
# from MarkovSoccerGame.friendQ import FriendQ
# from MarkovSoccerGame.ceQ import CEQ
from matplotlib import pyplot as plt
from game_interface import SoccerGame
import ddpgagent
import torch
from numpy import random
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

MAX_EPISODES=100000
MAX_STEP=200
LR_A=0.0003
LR_C=0.001
GAMMA=0.1
MEMORY_CAPACITY=10000
BATCH_SIZE=16
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
DDPGAgent = ddpgagent.DDPGAgent(ENV, GAMMA,LR_A,LR_C,BATCH_SIZE,MEMORY_CAPACITY,TAU)
Opponent = randomPlayAgent(ENV, GAMMA)
Opponent2 = randomPlayAgent(ENV, GAMMA)
game1 = SoccerGame(MAX_EPISODES, alpha_start, alpha_decay, alpha_min, 
                   epsilon_start, epsilon_decay, epsilon_min, GAMMA, ENV, DDPGAgent, Opponent)
# QLearnErr = game1.train()
win_rate=game1.train()
plt.plot(win_rate, linewidth=0.5)
plt.show()