from this import d
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self,s_size,a_size):   #! take care of it, a_span = 5!!
        super(Actor,self).__init__()
        self.fc1=nn.Linear(s_size,30)
        # print(self.fc1)
        self.fc2=nn.Linear(30,a_size)
        """initial the weight"""
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2.weight.data.normal_(0,0.1)
        
    def forward(self,s):
        s=self.fc1(s)
        s=F.relu(s)
        s=self.fc2(s)
        a=torch.tanh(s)
        return 2*a
    
class Critic(nn.Module):
    def __init__(self,s_size,a_size):
        super(Critic,self).__init__()
        self.fc1=nn.Linear(s_size,30)
        self.fc2=nn.Linear(a_size,30)
        self.out=nn.Linear(30,1)
        """initial the weight"""
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2.weight.data.normal_(0,0.1)
        self.out.weight.data.normal_(0,0.1)
        
    def forward(self,s,a):
        # s=torch.cat((s,a),1)
        # s,a=s.float(),a.float()
        s=self.fc1(s)
        a=self.fc2(a)
        q=self.out(F.relu(s+a))
        return q
    
    
class DDPGAgent():
    def __init__(self,gamma,s_size,a_size,lr_a,lr_c,batch_size,memory_capacity,tau):
        self.s_size,self.a_size=s_size,a_size
        # s_size,a_size mean the dimension of them (s_dim is more than better),a_span means the number of value a can take
        self.gamma,self.tau=gamma,tau
        self.batch_size=batch_size
        self.memory=np.zeros([memory_capacity, 2*self.s_size+self.a_size+1],dtype=np.float32)
        # This memory is replay buffer, not the history of game
        self.memory_capacity=memory_capacity
        self.pointer=0 # serves as updating the memory data(see store_transition)
        """create networks"""
        self.actor=Actor(s_size,a_size)
        self.critic=Critic(s_size,a_size)
        self.actor_target=Actor(s_size,a_size)
        self.critic_target=Critic(s_size,a_size)
        
        """create optimizers"""
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=lr_a)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=lr_c)
        """define the loss"""
        self.loss=nn.MSELoss()
        
    def store_transition(self,s,a,r,s_):
        transition=np.hstack((s,a,[r],s_))
        index=self.pointer % self.memory_capacity
        self.memory[index,:]=transition  
        self.pointer+=1
        
    
    def choose_action(self,s): 
        #! take care of the shape!
        a=self.actor(torch.unsqueeze(torch.FloatTensor(s),0))
        return a[0].detach()
    
    

    def learn(self):
        """target parameters are softly updated"""
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-self.tau))')
            eval('self.actor_target.' + x + '.data.add_(self.tau*self.actor.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-self.tau))')
            eval('self.critic_target.' + x + '.data.add_(self.tau*self.critic.' + x + '.data)')
        """sample a mini-batch"""
        indices=np.random.choice(self.memory_capacity,size=self.batch_size)
        batch_trans=self.memory[indices,:]  # slide whose element is (s,a,o,r,ro,d,s_)
        batch_s =torch.FloatTensor(batch_trans[:,:self.s_size])
        batch_a =torch.FloatTensor(batch_trans[:,self.s_size:self.s_size+self.a_size])
        batch_r =torch.FloatTensor(batch_trans[:,-self.s_size-1:-self.s_size])
        batch_s_=torch.FloatTensor(batch_trans[:,-self.s_size:])
        """forward"""
        a=self.actor(batch_s)
        q=self.critic(batch_s,a)
        actor_loss=-torch.mean(q)
        """optimize actor"""
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        a_target=self.actor_target(batch_s)
        q_temp=self.critic_target(batch_s_,a_target)
        q_target=batch_r+self.gamma * q_temp
        q_evalue=self.critic(batch_s,batch_a)
        td_error=self.loss(q_evalue,q_target)
        """optimize critic"""
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()
        

        