import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import time

"""
    This is for discrete state and discrete action, a isolate agent
    
"""

"""hyperparemetmer"""

MAX_EPISODES=200
MAX_STEP=200
LR_A=0.001
LR_C=0.002
GAMMA=0.01
MEMORY_CAPACITY=10000
BATCH_SIZE=32
TAU=0.01
RENDER=False



"""DDPG"""

class Actor(nn.Module):
    def __init__(self,s_size,a_size):
        super(Actor,self).__init__()
        self.fc1=nn.Linear(s_size,(s_size-a_size)//2)
        self.fc2=nn.Linear((s_size-a_size)//2,a_size)
        """initial the weight"""
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2.weight.data.normal_(0,0.1)
        
    def forward(self,x):
        x=x.to(torch.float32)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.softmax(x)  # in soccer game a is discrete
        return x
    
class Critic(nn.Module):
    def __init__(self,s_size,a_size):
        super(Critic,self).__init__()
        self.fc1=nn.Linear(s_size+a_size,(s_size+a_size)//2)
        self.fc2=nn.Linear((s_size+a_size)//2,1)
        """initial the weight"""
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2.weight.data.normal_(0,0.1)
        """网上的实现是分别生成S->R,A->R的两个网络, 然后第一层激活后
        两个值相加再进入第二层"""
        
    def forward(self,s,a):
        # print(s.shape)
        # print(a.shape)
        s=torch.cat((s,a),1)
        s=s.to(torch.float32)
        s=self.fc1(s)
        s=F.relu(s)
        s=self.fc2(s)
        return s
    
class DDPG(object):
    def __init__(self,s_size, a_size, a_bound):  
        self.s_size, self.a_size, self.a_bound = s_size, a_size, a_bound
        self.memory=np.zeros([MEMORY_CAPACITY, 2*s_size+1*a_size],dtype=np.float32)
        self.pointer=0 # serves as updating the memory data(see store_transition)
        """create networks"""
        self.actor=Actor(s_size,a_size)
        self.critic=Critic(s_size,a_size)
        self.actor_target=Actor(s_size,a_size)
        self.critic_target=Critic(s_size,a_size)
        """create optimizers"""
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=LR_A)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=LR_C)
        """define the loss function  """
        self.loss=nn.MSELoss()
        # self.loss=nn.CrossEntropyLoss()
    
    def store_transition(self, s,a ,r,s_):
        transition=np.hstack((s,a,[r],s_)) #! why [r]?
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index,:]=transition
        self.pointer+=1
        
    def choose_action(self,s):
        s= torch.unsqueeze(torch.FloatTensor(s),0)
        a=self.actor(s)[0].argmax().detach()
        return a
        
    def learn(self):
        """target parameters are softly updated"""
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_target.' + x + '.data)')
        """sample a mini-batch"""
        indices=np.random.choice(MEMORY_CAPACITY,size=BATCH_SIZE)
        batch_trans=self.memory[indices,:]  # slide whose element is (s,a,r,s_)
        batch_s= torch.FloatTensor(batch_trans[:,self.s_size])
        batch_a=torch.FloatTensor(batch_trans[:,self.s_size:self.s_size+self.a_size])
        batch_r=torch.FloatTensor(batch_trans[:,-self.s_size-1:-self.s_size])
        batch_s_=torch.FloatTensor(batch_trans[:,-self.s_size:])
        """forward"""
        a=self.actor(batch_s)
        q=self.critic(batch_s,a)
        actor_loss=-torch.mean(q)
        a_target=self.actor_target(batch_s)
        q_temp=self.critic_target(batch_s,a_target)
        q_target=batch_r+GAMMA * q_temp #! remeber the terminal
        q_evalue=self.critic(batch_s,batch_a)
        td_error=self.loss(q_evalue,q_target)
        """optimize"""
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()
    
# if __name__=="__main__":
#     print("test!")
        