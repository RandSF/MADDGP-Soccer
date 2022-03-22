from agents import *

class randomPlayAgent(ISoccerGameAgent):
    def __init__(self, env, gamma):
        super().__init__(env, gamma)
        # stateSpace = env.state_space
        # actSpace = env.action_space
        # dimOfQ = np.concatenate((stateSpace, [actSpace, actSpace]))
        # self.Q = np.ones(dimOfQ)

    def act(self, s):
        return np.random.randint(self.env.action_space[0])

    def learn(self, s, action, opponentAction, s_, reward, opponent_reward,done):
        pass