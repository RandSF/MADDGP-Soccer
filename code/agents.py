import copy
import numpy as np
# from cvxopt import matrix, solvers
from soccer import SoccerEnviroment
from abc import ABC, abstractmethod

# this is the interface for all agents
class ISoccerGameAgent(ABC):
    def __init__(self, env: SoccerEnviroment, gamma):
        self.env = env
        self.gamma = gamma
    
    @abstractmethod
    def act(self, s):
        pass

    @abstractmethod
    def learn(self, s, action, opponentAction, s_, reward, opponent_reward, done):
        pass