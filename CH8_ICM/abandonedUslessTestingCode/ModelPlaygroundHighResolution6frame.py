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

eps=0.15
# done = True
# state_deque = deque(maxlen=params['frames_per_state'])
x_pos = 40
stuckCounter = 0

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
    for i in range(state1.shape[1]-1):
        state1[0][i] = state1[0][i+1]
    state1[0][state1.shape[1]-1] = tmp
    return state1
    
def prepare_initial_state(state,N=3):
    state_ = torch.from_numpy(downscale_obs(state,to_gray=True)).float().to(device)
    tmp = state_.repeat((N,1,1)).to(device)
    return tmp.unsqueeze(dim=0)    

def policy(qvalues,eps = None,tau = None):
    if eps is not None:
        if torch.rand(1) < eps:
            return torch.randint(low=1,high=11,size=(1,))
        else:
            return torch.argmax(qvalues)
    else:
        if tau is not None:
            qvalues /= tau
        return torch.multinomial(torch.nn.functional.softmax(torch.nn.functional.normalize(qvalues)),num_samples=1)

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
        y = torch.nn.functional.elu(self.conv1(x))
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


params = {
    'batch_size' :150,
    'beta' : 0.2,
    'lambda' : 0.1,
    'eta' : 1, #rate between intricsic and extrinsic reward
    'gamma' : 0.2,
    'max_episode_len' : 500,
    'min_progress' : 15,
    'action_repeats' : 3, #被選到的action 在訓練時會連做6次
    'frames_per_state' : 6 
}
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env,COMPLEX_MOVEMENT)
reset_env()
done = True 

Qmodel = QNetwork().to(device)
encoder = Phi().to(device)
forward_model = Fnet().to(device)
inverse_model = Gnet().to(device)
qloss = torch.nn.MSELoss()
forward_loss = torch.nn.MSELoss(reduction='none')
inverse_loss = torch.nn.CrossEntropyLoss(reduction='none')

Qmodel.load_state_dict(torch.load("Qmodel.pth"))
Qmodel.eval() # Set to evaluation mode
encoder.load_state_dict(torch.load("encoder.pth"))
encoder.eval() # Set to evaluation mode
inverse_model.load_state_dict(torch.load("inverse_model.pth"))
inverse_model.eval() # Set to evaluation mode
forward_model.load_state_dict(torch.load("forward_model.pth"))
forward_model.eval() # Set to evaluation mode



# import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import pygame
import numpy as np

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1200, 600))
pygame.display.set_caption("Super Mario Bros Control")
scale_factor = 2.5# Create the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0',)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
reset_env()
done = True
action = 0  # Initialize action to idle

# Create a clock to control the frame rate
clock = pygame.time.Clock()

step_counter = 0
while True:
    step_counter += 1
    
    current_frame = env.render('rgb_array')
    downscale_frame = downscale_obs(current_frame)
    if done or (stuckCounter == 500) :
        stuckCounter = 0
        env.reset()
        state1 = prepare_initial_state(current_frame,params['frames_per_state'])
    q_val_pred = Qmodel(state1)
    # action = int(policy(q_val_pred,eps))
    action = int(policy(q_val_pred,eps=0.3))
    
    
    
    
    
    for i in range(3):
        if done or (stuckCounter == 200) :
            stuckCounter = 0
            env.reset()
            state1 = prepare_initial_state(current_frame,params['frames_per_state'])
        state2, reward, done, info = env.step(action)
        if(x_pos == info['x_pos']):
            stuckCounter += 1
        else:
            stuckCounter = 0
        x_pos = info['x_pos']
    
        state2 = prepare_multi_state(state1,state2)
        state1=state2



        if (step_counter % 10 == 0):
            output_frame1 = downscale_frame
            output_frame1 = np.stack((output_frame1,)*3, axis=-1)
            output_frame1 = (output_frame1 * 255).astype(np.uint8)
            output_frame1 = np.fliplr(output_frame1)  # Flip the frame vertically
            output_frame1 = np.rot90(output_frame1)
            # plt.imshow(output_frame1)
            # plt.show()
            output_frame1 = pygame.surfarray.make_surface(output_frame1)
            output_frame1 = pygame.transform.scale(output_frame1, (600 , 600))
            
            
            output_frame2 = current_frame
            output_frame2 = np.fliplr(output_frame2)  # Flip the frame vertically
            output_frame2 = np.rot90(output_frame2)
            
            output_frame2 = pygame.surfarray.make_surface(output_frame2)
            output_frame2 = pygame.transform.scale(output_frame2, (600,600))
    
            screen.blit(output_frame1, (600, 0))
            screen.blit(output_frame2, (0, 0))
            pygame.display.update()  # Update the display
    
            # Control the frame rate
            clock.tick(30)  # Adjust this value to control the speed of the game
    