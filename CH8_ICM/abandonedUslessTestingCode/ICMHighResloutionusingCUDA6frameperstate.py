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

device = torch.device("cuda")
print(torch.cuda.get_device_name(0)) 

def downscale_obs(obs, new_size = (84,84), to_gray = True):
    if to_gray:
        return resize(obs,new_size,anti_aliasing = True).max(axis = 2)
    else :
        return resize(obs,new_size,anti_aliasing = True)

def prepare_state(state):
    return torch.from_numpy(downscale_obs(state,to_gray=True)).float().unsqueeze(dim=0).to(device)

def prepare_multi_state(state1,state2):
# state1 : old tensor with recently 3 frame images with batch dim (downscaled)(batch * 3 * height * width)
# state2 : new np array with the newest 1 frame image(undownscaled)
    state1 = state1.clone().to(device)
    tmp = torch.from_numpy(downscale_obs(state2)).float().to(device)
    # for i 
    
    state1[0][0] = state1[0][1]
    state1[0][1] = state1[0][2]
    state1[0][2] = tmp
    return state1
    
def prepare_initial_state(state,N=6):
    state_ = torch.from_numpy(downscale_obs(state,to_gray=True)).float().to(device)
    tmp = state_.repeat((N,1,1)).to(device)
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
        
        if len(self.memory) < self.N :
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

        ind = np.random.choice(np.arange(len(self.memory)),batch_size,replace=False)
        # any data can be chosen at most once
        batch = [self.memory[i] for i in ind]
        state1_batch = torch.stack([x[0].squeeze(dim=0) for x in batch],dim=0).to(device)
        action_batch = torch.Tensor([x[1] for x in batch]).long().to(device)
        reward_batch = torch.Tensor([x[2] for x in batch]).to(device)
        state2_batch = torch.stack([x[3].squeeze(dim=0) for x in batch],dim=0).to(device)
        return state1_batch,action_batch,reward_batch,state2_batch
    
class Phi (torch.nn.Module):
    def __init__(self):
        super(Phi,self).__init__()
        self.conv1 = torch.nn.Conv2d(6,32,kernel_size=(3,3),stride=2,padding=1)
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
        return y
        
class Gnet(torch.nn.Module):
    def __init__(self):
        super(Gnet,self).__init__()
        self.l1 = torch.nn.Linear(2304,512)
        self.l2 = torch.nn.Linear(512,12)
                
    def forward(self,state1,state2):
        x = torch.cat((state1,state2),dim=1)
        y = self.l1(x)
        y = torch.nn.functional.relu(y)
        y = self.l2(y)
        y = torch.nn.functional.softmax(y,dim=1)
        return y        

class Fnet(torch.nn.Module):
    def __init__(self):
        super(Fnet,self).__init__()
        self.l1 = torch.nn.Linear(1152+12,512)
        self.l2 = torch.nn.Linear(512,1152)
                
    def forward(self,state,action):
        action_ = torch.zeros(action.shape[0],12).to(device)
        indicies = torch.stack((torch.arange(state.shape[0]).to(device),action.squeeze()),dim=0)
        indicies = indicies.tolist()
        action_[indicies] = 1
        x = torch.cat((state,action_),dim=1)
        y = self.l1(x)
        y = torch.nn.functional.relu(y)
        y = self.l2(y)
        return y        

class QNetwork (torch.nn.Module):
    def __init__(self):
        super(QNetwork,self).__init__()
        self.conv1 = torch.nn.Conv2d(6,32,kernel_size=(3,3),stride=2,padding=1)
        self.conv2 = torch.nn.Conv2d(32,32,kernel_size=(3,3),stride=2,padding=1)
        self.conv3 = torch.nn.Conv2d(32,32,kernel_size=(3,3),stride=2,padding=1)
        self.conv4 = torch.nn.Conv2d(32,32,kernel_size=(3,3),stride=2,padding=1)
        self.l1 = torch.nn.Linear(1152, 500)     
        self.l2 = torch.nn.Linear(500,100)
        self.l3 = torch.nn.Linear(100,12)
            
    def forward(self,x):
        x = torch.nn.functional.normalize(x)
        x = self.conv1(x)
        y = torch.nn.functional.elu(x)
        y = torch.nn.functional.elu(self.conv2(y))
        y = torch.nn.functional.elu(self.conv3(y))
        y = torch.nn.functional.elu(self.conv4(y))
        y = y.flatten(start_dim=2)
        y = y.view(y.shape[0],-1,32)
        y = y.flatten(start_dim=1)
        y = torch.nn.functional.elu(self.l1(y))
        y = torch.nn.functional.elu(self.l2(y))
        y = self.l3(y)
        return y 

def loss_fn(qloss,forward_loss,inverse_loss):
    loss_ = (1-params['beta']) * inverse_loss
    loss_ += params['beta'] * forward_loss
    loss_ = loss_.sum() / loss_.flatten().shape[0]
    loss = loss_ + params['lambda'] * qloss
    return loss

def reset_env():
    env.reset()
    state1 = prepare_initial_state(env.render('rgb_array'),params['frames_per_state'])
    return state1

def ICM(state1, action, state2, forward_scale = 1 , inverse_scale = 1e4):
    state1_phi = encoder(state1).to(device)
    state2_phi = encoder(state2).to(device)
    state2_phi_predict = forward_model(state1_phi.detach(),action.detach())
    forward_pred_error = forward_loss(state2_phi_predict, state2_phi.detach()).sum(dim=1).unsqueeze(dim=1)
    forward_pred_error *= forward_scale

    action_pred = inverse_model(state1_phi,state2_phi)
    inverse_pred_error = inverse_loss(action_pred,action.detach().flatten()).unsqueeze(dim=1)
    inverse_pred_error *= inverse_scale
    
    return forward_pred_error , inverse_pred_error

def minibacth_train(use_extrinsic = True):
    state1_batch, action_batch, reward_batch,state2_batch = replay.get_batch()
    action_batch = action_batch.view(action_batch.shape[0],1).to(device)
    reward_batch = reward_batch.view(reward_batch.shape[0],1).to(device)
    forward_pred_error, inverse_pred_error = ICM(state1_batch,action_batch,state2_batch)    
    
    intrinsic_reward = forward_pred_error.detach().to(device)
    intrinsic_reward *= params['eta']
    
    reward = intrinsic_reward
    
    if use_extrinsic :
        reward += reward_batch

    qvals = Qmodel(state2_batch)
    ret = reward_batch + params['gamma'] * torch.max(qvals)
    
    ret_pred = Qmodel(state1_batch)
    ret_target = ret_pred.clone()


    #indices : convert into one hot
    indices = torch.stack((torch.arange(action_batch.shape[0]).to(device),action_batch.squeeze()),dim=0).to(device)
    indices = indices.tolist()
    
    ret_target[indices] = ret.squeeze()
    q_loss = 1e5 * qloss(torch.nn.functional.normalize(ret_pred),torch.nn.functional.normalize(ret_target.detach()))

    return forward_pred_error , inverse_pred_error , q_loss

  
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env,COMPLEX_MOVEMENT)
done = True 

params = {
    'batch_size' :150,
    'beta' : 0.2,
    'lambda' : 0.2,
    'eta' : 1, #rate between intricsic and extrinsic reward
    'gamma' : 0.2,
    'max_episode_len' : 1500,
    'min_progress' : 15,
    'action_repeats' : 3, #被選到的action 在訓練時會連做6次
    'frames_per_state' : 6
}

replay = ExperienceReplay(N = 1000,batch_size=params['batch_size'])
Qmodel = QNetwork().to(device)
encoder = Phi().to(device)
forward_model = Fnet().to(device)
inverse_model = Gnet().to(device)
qloss = torch.nn.MSELoss()
forward_loss = torch.nn.MSELoss(reduction='none')
inverse_loss = torch.nn.CrossEntropyLoss(reduction='none')
all_model_params = list(Qmodel.parameters()) + list(encoder.parameters()) + list(forward_model.parameters()) + list(inverse_model.parameters()) 
optim = torch.optim.Adam(lr=0.001,params=all_model_params)


epochs = 50000
env.reset()

state1 = prepare_initial_state(env.render('rgb_array'),params['frames_per_state'])
eps = 0.15
losses = []
episode_length = 0
switch_to_eqs_greedy = 30000
state_deque = deque(maxlen = params['frames_per_state'])
inidequefiller = prepare_state(env.render('rgb_array'))
for i in range(params['frames_per_state']):
    state_deque.append(inidequefiller)
    
e_reward = 0

#initail x_pos for this mario bros game,will be updated every epoch
last_x_pos = 40
ep_lengths = []
use_explicit = False
for i in range(epochs):
    if (i % 100 == 0):
        print('Epochs', i, ':' , 'x_pos',last_x_pos)
    optim.zero_grad()
    episode_length += 1
    q_val_pred = Qmodel(state1)
    
    if i > switch_to_eqs_greedy:
        action = int(policy(q_val_pred,eps=eps))
    else:
        action = int(policy(q_val_pred))

    #######################################################################
    # 把6次action 互動當成一次的update資料
    #雖然裡面每次都會塞新的frame進去deque裡面
    # 但一次訓練的reward就是6個action 的 reward加總
    #所以那去訓練的資料其實是連續三個frame ，空三個frame ，再連續看三個frame
    ###### 不是每6個frame留一個 變成 看(0、6、12) 、(6、12 、 18)的資料
    ###### 而是一次看連續三個frame的 即(0、1、2) 、(6、7、8)
    #所以與其說是給他看三個frame更像是給他看目前的動作在幹嘛
    ########################################################################
    for j in range(params['action_repeats']):
        state2, e_reward_, done, info = env.step(action)
        last_x_pos = info['x_pos']
        if done :
            state1 = reset_env()
            break
        
        e_reward += e_reward_
        state_deque.append(prepare_state(state2))
    state2 = torch.stack(list(state_deque),dim = 1)
    replay.add_memory(state1,action,e_reward,state2)
    e_reward = 0
    if episode_length > params['max_episode_len']:
        if (info['x_pos'] - last_x_pos) < params['min_progress']:
            print("min_progress")
            done = True
        else :
            last_x_pos = info['x_pos']

    if done :
        print('done')
        ep_lengths.append(info['x_pos'])
        state1 = reset_env()
        last_x_pos = 40
        episode_length = 0
    else:
        state1 = state2
    
    if len(replay.memory) < params['batch_size']:
        continue

    forward_pred_error,inverse_pred_error,q_loss = minibacth_train(use_extrinsic=True)
    loss = loss_fn(q_loss,forward_pred_error,inverse_pred_error)
    loss_list = (q_loss.mean(),forward_pred_error.flatten().mean(),inverse_pred_error.flatten().mean())
    losses.append(loss_list)
    loss.backward()
    optim.step()
    
torch.save(Qmodel.state_dict(), "Qmodel.pth")
torch.save(encoder.state_dict(), "encoder.pth")
torch.save(forward_model.state_dict(), "forward_model.pth")
torch.save(inverse_model.state_dict(), "inverse_model.pth")
    
eps=0.1
done = True
x_pos = 40
stuckCounter = 0


for step in range(5000):  
  if (step % 10 == 0): 
    print(step,x_pos)
    plt.imshow(env.render('rgb_array'))
    plt.pause(0.00005)
  if done or (stuckCounter == 100) :
    stuckCounter = 0
    env.reset()
    state1 = prepare_initial_state(env.render('rgb_array'),params['frames_per_state'])
  q_val_pred = Qmodel(state1)
  action = int(policy(q_val_pred,eps))
  state2, reward, done, info = env.step(action)
  if(x_pos == info['x_pos']):
    stuckCounter += 1
  x_pos = info['x_pos']
  
  state2 = prepare_multi_state(state1,state2)
  state1=state2