# import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT , COMPLEX_MOVEMENT #A
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import torch
from collections import deque
from random import shuffle

def downscale_obs(obs, new_size = (42,42), to_gray = True):
    if to_gray:
        return resize(obs,new_size,anti_aliasing = True).max(axis = 2)
    else :
        return resize(obs,new_size,anti_aliasing = True)

def prepare_state(state):
    return torch.from_numpy(downscale_obs(state)).float().unsqueeze()

def prepare_multi_state(state1,state2):
# state1 : old tensor with recently 3 frame images with batch dim (downscaled)(batch * 3 * height * width)
# state2 : new np array with the newest 1 frame image(undownscaled)
    state1 = state1.clone()
    tmp = torch.from_numpy(downscale_obs(state2)).float()
    
    state1[0][0] = state1[0][1]
    state1[0][1] = state1[0][2]
    state1[0][2] = tmp
    return state1
    
def prepare_initial_state(state,N=3):
    state_ = torch.from_numpy(downscale_obs(state)).float()
    tmp = state_.repeat((N,1,1))
    return tmp.unsqueeze(dim=0)    

def policy(qvalues,eps = None):
    if eps is not None:
        if torch.rand(1) < eps:
            return torch.randint(low=1,high=11,size=(1,))
        else:
            return torch.argmax(qvalues)
    else:
        return torch.multinomial(torch.nn.functional.softmax(torch.nn.functional.normalize(qvalues)),num_samples=1)

class ExperienceReplay:
    def __init__(self,N=500,batch_size=100):
        self.N = N
        self.batch_size = batch_size
        self.memory = []
        self.counter = 0
        
    def suffle_memory(self):
        shuffle(self.memory)
        
    def add_memory(self,state1,action,reward,state2):
        self.counter += 1
        if self.counter % 500 == 0:
            self.suffle_memory()
            #why need shuffle? already random picking and random replacing
        
        if len(self.memory) < self < self.N :
            self.memory.append((state1,action,reward,state2))
        else:
            rand_index = np.random.randint(0,self.N-1)
            self.memory[rand_index] = (state1,action,reward,state2)

    def get_batch(self):
        if len(self.memory) < self.batch_size:
            batch_size = len(self.memory) 
        else :
            batch_size = self.batch_size 
        
        if len(self.memory) < 1 :
            print("No data in Memory")
            return None          

        ind = np.random.choice(np.arange(len(self.memory),batch_size,replace=False))
        # any data can be chosen at most once
        batch = [self.memory[i] for i in ind]
        state1_batch = torch.stack([x[0].squeeze(dim=0) for x in batch],dim=0)
        action_batch = torch.Tensor([x[1] for x in batch]).long() 
        reward_batch = torch.Tensor([x[2] for x in batch]) 
        state2_batch = torch.stack([x[3].squeeze(dim=0) for x in batch],dim=0)
        return state1_batch,action_batch,reward_batch,state2_batch
    
class Phi (torch.nn.Module):
    def __init__(self):
        super(Phi,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,32,kernel_size=(3,3),stride=2,padding=1)
        self.conv2 = torch.nn.Conv2d(32,32,kernel_size=(3,3),stride=2,padding=1)
        self.conv3 = torch.nn.Conv2d(32,32,kernel_size=(3,3),stride=2,padding=1)
        self.conv4 = torch.nn.Conv2d(32,32,kernel_size=(3,3),stride=2,padding=1)
        
    def forward(self,x):
        x = torch.nn.functional.normalize(x)
        y = torch.nn.functional.elu(self.conv1(x))
        y = torch.nn.functional.elu(self.conv2(y))
        y = torch.nn.functional.elu(self.conv3(y))
        y = torch.nn.functional.elu(self.conv4(y))
        y = y.flatten(start_dim=1)
        
class Gnet(torch.nn.Module):
    def __init__(self):
        super(Gnet,self).__init__()
        self.l1 = torch.nn.Linear(576,256)
        self.l2 = torch.nn.Linear(256,12)
                
    def forard(self,state1,state2):
        x = torch.cat((state1,state2),dim=1)
        y = self.l1(x)
        y = torch.nn.ReLU(y)
        y = self.l2(x)
        y = torch.nn.Softmax(y,dim=1)
        return y        

class Fnet(torch.nn.Module):
    def __init__(self):
        super(Fnet,self).__init__()
        self.l1 = torch.nn.Linear(300,256)
        self.l2 = torch.nn.Linear(256,288)
                
    def forard(self,state,action):
        action_ = torch.zeros(action.shape(),12)
        indicies = torch.stack((torch.arange(state.shape[0]),action.squeeze()),dim=0)
        indicies = indicies.tolist()
        action_[indicies] = 1
        x = torch.cat((state,action_),dim=1)
        y = self.l1(x)
        y = torch.nn.ReLU(y)
        y = self.l2(x)
        return y        

class QNetwork (torch.nn.Module):
    def __init__(self):
        super(QNetwork,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,32,kernel_size=(3,3),stride=2,padding=1)
        self.conv2 = torch.nn.Conv2d(32,32,kernel_size=(3,3),stride=2,padding=1)
        self.conv3 = torch.nn.Conv2d(32,32,kernel_size=(3,3),stride=2,padding=1)
        self.conv4 = torch.nn.Conv2d(32,32,kernel_size=(3,3),stride=2,padding=1)
        self.l1 = torch.nn.Linear(288, 100)
        self.l2 = torch.nn.linear(100,12)
            
    def forward(self,x):
        x = torch.nn.functional.normalize(x)
        y = torch.nn.functional.elu(self.conv1(x))
        y = torch.nn.functional.elu(self.conv2(y))
        y = torch.nn.functional.elu(self.conv3(y))
        y = torch.nn.functional.elu(self.conv4(y))
        y = y.flatten(start_dim=2)
        y = y.view(y.shape[0],-1,32)
        y = y.flatten(start_dim=1)
        y = torch.nn.functional.elu(self.l1(y))
        y = self.l2(y)
        return y 
    
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env,COMPLEX_MOVEMENT)
done = True 

params = {
    'batch_size' :150,
    'beta' : 0.2,
    'lambda' : 0.1,
    'gamma' : 0.2,
    'max_episode_len' : 100,
    'min_progress' : 15,
    'action_repeats' : 6, #被選到的action 在訓練時會連做6次
    'frames_per_state' : 3 
}

replay = ExperienceReplay(N = 1000,batch_size=params['batch_size'])
Qmodel = QNetwork()
encoder = Phi()
forward_model = Fnet()
inverse_model = Gnet()
forward_loss = torch.nn.MSELoss(reduction='none')
inverse_loss = torch.nn.CrossEntropyLoss(reduction='none')


for step in range(30):
    if done :
        state = env.reset()
    action = env.action_space.sample()
    state,reward,done,info = env.step(action)
    obs = env.render('rgb_array')
    obs_ = downscale_obs(obs)
    plt.imshow(obs_)
    plt.pause(0.1)






#A
{
# # actions for more complex movement
# COMPLEX_MOVEMENT = [
# 0    ['NOOP'],
# 1    ['right'],
# 2    ['right', 'A'],
# 3    ['right', 'B'],
#  4   ['right', 'A', 'B'],
#  5   ['A'],
#  6   ['left'],
#  7   ['left', 'A'],
#  8   ['left', 'B'],
#  9   ['left', 'A', 'B'],
#  10  ['down'],
#  11  ['up'],
# ]
}