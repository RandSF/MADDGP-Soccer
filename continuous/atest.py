import matplotlib.pyplot as plt
import numpy as np
import gym
from sympy import true
import torch
from ddpgAgent import DDPGAgent

MAX_EPISODES=200
MAX_STEP=200
LR_A=0.001
LR_C=0.002
GAMMA=0.9
MEMORY_CAPACITY=10000
BATCH_SIZE=32
TAU=0.01 
VAR=3
DECAY=.9995
GLOBAL_R=[]
RENDER=False

env=gym.make("Pendulum-v1")
env=env.unwrapped
# env.seed(114514)


s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
ddpg=DDPGAgent(GAMMA,s_dim,a_dim,LR_A,LR_C,BATCH_SIZE,MEMORY_CAPACITY,TAU)
var=VAR
print("START!")
s=env.reset(seed=114514)
for ep in range(MAX_EPISODES):
    s=env.reset()
    ep_reward=0
    for i in range(MAX_STEP):
        if RENDER:env.render()
         
        a=ddpg.choose_action(s)
        a=np.clip(np.random.normal(a,var),-2.,2.)
        s_,r,done,info=env.step(a)
        ddpg.store_transition(s,a,r/10,s_)
        
        if ddpg.pointer>MEMORY_CAPACITY:
            var*=DECAY
            ddpg.learn()
            
        s=s_
        ep_reward+=r
        
        if i ==MAX_STEP-1:
            print("episode:{}/{}, ep_reward={:.2f}".format(ep,MAX_EPISODES,ep_reward))
            if len(GLOBAL_R)==0:
                GLOBAL_R.append(ep_reward)
            else:
                GLOBAL_R.append(.7*GLOBAL_R[-1]+.3*ep_reward)
            # GLOBAL_R.append(ep_reward)
            if ep_reward>-100 :
                RENDER=True
            # break
env.close()

plt.plot(np.arange(len(GLOBAL_R)), GLOBAL_R)
plt.xlabel('episode')
plt.ylabel('Total moving reward')
plt.show()