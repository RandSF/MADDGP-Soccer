from collections import deque
import torch
import numpy as np


# this class is where the game is actually played
# it takes the soccer enviroment, the agent and the opponent as parameters
# and simulate a soccer game where the agent and the opponent play against each other
# the agent and opponent can use any of the alogorithms implemented here
# and they learn and behave independent of each other
# the learning parameters are provided and are the same for both players
"""
todo : overwrite the game and the random player such that they acting in the same way as DDPG
"""
class SoccerGame:
    def __init__(self, numEpisode, epsilon_start, epsilon_decay, epsilon_min,
                 env, agent, opponent, maxStep=500):
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.numEpisode = numEpisode
        self.env = env
        self.agent = agent
        self.opponent = opponent
        self.maxStep = maxStep

    """
    This function is unnecessary because we temporarily have no need to check a sepcific q-value
    """
    # sample a fixed point in agent's Q function space
    # by default the start position
    # def __sampleAgentQValue(self, s0=2, s1=1, s2=0, a=1, o=4):
    #     # special case for Q learning
    #     if len(self.agent.Q.shape)<5:
    #         return self.agent.Q[s0,s1,s2,a]
    #     return self.agent.Q[s0,s1,s2,a,o]

    # epsilon defines a unified exploration rate during training for both players
    def train(self):
        count = 0
        win_rate=[]
        q_value=[]
        # error = []
        # current_val = self.__sampleAgentQValue()
        # alpha = self.alpha_start
        epsilon = self.epsilon_start
        memory = deque(maxlen=100)
        for episode in range(self.numEpisode):
            win_rate.append(np.average(memory))
            n = 1000
            if episode % n == n-1:
                print("episode: {} / {}, win rate={:.2f}, epsilon={:4f}".format(episode,
                    self.numEpisode, np.average(memory), epsilon))
            s = self.env.reset()
            step = 0
            
            while True:
                if np.random.random()<epsilon:
                    agentAct = np.random.randint(self.env.action_space[0])
                else:
                    agentAct = self.agent.act(s)
                if np.random.random()<epsilon:
                    opponentAct = np.random.randint(self.env.action_space[0])
                else:
                    opponentAct = self.opponent.act(s)
                # if (s[0],s[1],s[2],agentAct,opponentAct)==(2,1,False,1,4): count += 1
                s_ ,reward , done= self.env.step(agentAct, opponentAct)
                self.agent.learn(s, agentAct, opponentAct,
                                        s_, reward, -reward, done)
                self.opponent.learn(s, opponentAct, agentAct,
                                        s_, -reward, reward, done)
                if done or step > self.maxStep:
                    memory.append(reward==100)
                    break
                s = s_
                step += 1
                with torch.no_grad():
                    q_value.append(self.agent.critic(torch.tensor(([[1,2,1]])),torch.tensor(([[3]]))))
            # if alpha > self.alpha_min:
            #     alpha *= self.alpha_decay
            if epsilon > self.epsilon_min:
                    epsilon *= self.epsilon_decay
            # new_val = self.__sampleAgentQValue()
            # error.append(abs(new_val - current_val))
            # current_val = new_val
        # print(count)
        return win_rate,q_value

    def play(self, render=True):
        s = self.env.reset()
        step = 0
        if render:
            self.env.render()
        while True:
            agentAct = self.agent.act(s[0], s[1], s[2])
            opponentAct = self.opponent.act(s[0], s[1], s[2])
            s_prime, reward, done = self.env.step(agentAct, opponentAct)
            if render:
                print("\n", agentAct, opponentAct)
                self.env.render()
            if done or step > self.maxStep:
                break
            s = s_prime
            step += 1
        return reward

    def evaluate(self, num=10000):
        rewards = []
        for i in range(num):
            rewards.append(self.play(False)==100)
        return np.average(rewards)