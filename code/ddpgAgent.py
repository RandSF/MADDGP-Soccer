from agents import ISoccerGameAgent
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self,s_size,a_span):   #! take care of it, a_span = 5!!
        super(Actor,self).__init__()
        self.fc1=nn.Linear(s_size,10)
        # print(self.fc1)
        self.fc2=nn.Linear(10,a_span)
        """initial the weight"""
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2.weight.data.normal_(0,0.1)
        
    def forward(self,s):
        s=s.to(torch.float32)
        s=self.fc1(s)
        s=F.relu(s)
        s=self.fc2(s)
        s=F.softmax(s,dim=-1)  # in soccer game a is discrete
        return s
    
class Critic(nn.Module):
    def __init__(self,s_size,a_size):
        super(Critic,self).__init__()
        self.fc1=nn.Linear(s_size+a_size,2)
        self.fc2=nn.Linear(2,1)
        """initial the weight"""
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2.weight.data.normal_(0,0.1)
        """网上的实现是分别生成S->R,A->R的两个网络, 然后第一层激活后
        两个值相加再进入第二层"""
        
    def forward(self,s,a):
        s=torch.cat((s,a),1)
        s=s.to(torch.float32)
        s=self.fc1(s)
        s=F.relu(s)
        s=self.fc2(s)
        return s
    
    
class DDPGAgent(ISoccerGameAgent):
    def __init__(self,env,gamma,lr_a,lr_c,batch_size,memory_capacity,tau):
        self.env=env
        self.s_size,self.a_size,self.a_span=len(env.state_space),len(env.action_space),env.action_space[0]
        # s_size,a_size mean the dimension of them (s_dim is more than better),a_span means the number of value a can take
        self.gamma,self.tau=gamma,tau
        self.batch_size=batch_size
        self.memory=np.zeros([memory_capacity, 2*self.s_size+2*self.a_size + 2 + 1],dtype=np.float32)
        # This memory is replay buffer, not the history of game
        self.memory_capacity=memory_capacity
        self.pointer=0 # serves as updating the memory data(see store_transition)
        """create networks"""
        self.actor=Actor(self.s_size,self.a_span)
        self.critic=Critic(self.s_size,self.a_size)
        self.actor_target=Actor(self.s_size,self.a_span)
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
        with torch.no_grad():
            a=self.actor(s).argmax(axis=0)#.detach()
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
        pa=self.actor(batch_s)
        a=pa.argmax(axis=1)
        a=a.reshape((a.shape[0],1))
        q=self.critic(batch_s,a)
        actor_loss=-torch.mean(q)
        """optimize actor"""
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        pa_target=self.actor_target(batch_s)
        a_target=pa_target.argmax(axis=1)
        a_target=a_target.reshape((a_target.shape[0],1))
        q_temp=self.critic_target(batch_s,a_target)
        q_target=batch_r+self.gamma * (1-batch_d) * q_temp #! remeber the terminal
        # q_target=batch_r+self.gamma * q_temp
        q_evalue=self.critic(batch_s,batch_a)
        td_error=self.loss(q_evalue,q_target)
        """optimize critic"""
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()
        

        