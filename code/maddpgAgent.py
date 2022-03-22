from agents import ISoccerGameAgent
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self,s_size,a_size):
        super(Actor,self).__init__()
        self.fc1=nn.Linear(s_size,10)
        # print(self.fc1)
        self.fc2=nn.Linear(10,a_size)
        """initial the weight"""
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2.weight.data.normal_(0,0.1)
        
    def forward(self,x):
        x=x.to(torch.float32)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.softmax(x,dim=0)  # in soccer game a is discrete
        return x
    
class Critic(nn.Module):    # here builds a central critic different from PPDGAgent
    def __init__(self,s_size,a_size):
        super(Critic,self).__init__()
        self.fc1=nn.Linear(s_size+2*a_size,abs(s_size+2*a_size)//2)
        self.fc2=nn.Linear(abs(s_size+2*a_size)//2,1)
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
    
    
class DDPGAgent(ISoccerGameAgent):
    def __init__(self,env,gamma,lr_a,lr_c,batch_size,memory_capacity,tau):
        self.env=env
        self.s_size,self.a_size=len(env.state_space),len(env.action_space)
        self.gamma,self.tau=gamma,tau
        self.batch_size=batch_size
        self.memory=np.zeros([memory_capacity, 2*self.s_size+2*self.a_size + 2 + 1],dtype=np.float32)
        # This memory is expierience buffer, not the history of game
        self.memory_capacity=memory_capacity
        self.pointer=0 # serves as updating the memory data(see store_transition)
        """create networks"""
        self.actor=Actor(self.s_size,self.a_size)
        self.critic=Critic(self.s_size,self.a_size)
        self.actor_target=Actor(self.s_size,self.a_size)
        self.critic_target=Critic(self.s_size,self.a_size)
        """create optimizers"""
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=lr_a)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=lr_c)
        """define the loss"""
        self.loss=nn.MSELoss()
        
    def store_transition(self,s,a,o,r,ro,d,s_):
        transition=np.hstack((s,a,o,r,ro,d,s_))
        index=self.pointer % self.memory_capacity
        self.memory[index,:]=transition
        self.pointer+=1
        
    
    def act(self,s):
        a=self.actor(s).argmax(axis=0).detach()
        return a    # return a tensor with shape=(1,batch_a)
    
    
    def learn(self,s,a_agent,a_opponent,s_,r_agent,r_opponent,done):
        self.store_transition(s,a_agent,a_opponent,r_agent,r_opponent,done,s_)
        if self.pointer>self.memory_capacity:
            self.update()
    
    def update(self):
        """target parameters are softly updated"""
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-self.tau))')
            eval('self.actor_target.' + x + '.data.add_(self.tau*self.actor.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-self.tau))')
            eval('self.critic_target.' + x + '.data.add_(self.tau*self.critic_target.' + x + '.data)')
        """sample a mini-batch"""
        indices=np.random.choice(self.memory_capacity,size=self.batch_size)
        batch_trans=self.memory[indices,:]  # slide whose element is (s,a,o,r,ro,d,s_)
        batch_s =torch.FloatTensor(batch_trans[:,:self.s_size])
        batch_a =torch.FloatTensor(batch_trans[:,self.s_size:self.s_size+self.a_size])
        batch_o =torch.FloatTensor(batch_trans[:,self.s_size+self.a_size:self.s_size+2*self.a_size])
        batch_r =torch.FloatTensor(batch_trans[:,-self.s_size-3:-self.s_size-2])
        batch_ro=torch.FloatTensor(batch_trans[:,-self.s_size-2:-self.s_size-1])
        batch_d =torch.FloatTensor(batch_trans[:,-self.s_size-1:-self.s_size])
        batch_s_=torch.FloatTensor(batch_trans[:,-self.s_size:])
        
        """forward"""
        a=self.actor(batch_s)
        q=self.critic(batch_s,a,o)
        actor_loss=-torch.mean(q)
        
        a_target=self.actor_target(batch_s)
        q_temp=self.critic_target(batch_s,a_target,o)
        q_target=batch_r+self.gamma * (1-batch_d)* q_temp #! remeber the terminal
        q_evalue=self.critic(batch_s,batch_a,batch_o)
        td_error=self.loss(q_evalue,q_target)
        """optimize"""
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()

        