from Gridworld import Gridworld
import numpy as np
import torch 
from IPython.display import clear_output
import random
from matplotlib import pylab as plt
from typing import Type
from torchviz import make_dot

action_set = {
    0:'u',
    1:'d',
    2:'l',
    3:'r'
}
 
def percent_bar(i):
        progress = (i + 1) / epochs
        bar_length = 50
        progress_chars = int(progress * bar_length)
        bar = '[' + '#' * progress_chars + '-' * (bar_length - progress_chars) + ']'
        percent = '{:.2%}'.format(progress)
        print(f'\r{bar} {percent}', end='', flush=True)

def test_model(model,mode="static",display = True):
    i = 0
    game = Gridworld(4, mode)
    state_ = game.board.render_np().reshape(1,64)+ np.random.rand(1,64)/10.
    state = torch.from_numpy(state_).float()
    
    status = 1
    while(status == 1):
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)
        action = action_set[action_]
        
        if(display) :    
            print(game.display())
            print('action : ' + action)
        
        game.makeMove(action)
        state_ = game.board.render_np().reshape(1,64)+np.random.rand(1,64)/10.0
        state = torch.from_numpy(state_).float()
        reward = game.reward()
        if(reward != -1):
            status = 0
            win = True if reward == 10 else False
            if (display):
                print("result : win") if win else print("result : lose")            
            return win 
        i += 1
        if  i >= 15 :
            status = 0
            if(display):
                print("out of time")
            win = False
            return win 
    
L1 = 64
L2 = 150
L3 = 100
L4 = 4

model = torch.nn.Sequential(
    torch.nn.Linear(L1,L2),
    torch.nn.ReLU(),
    torch.nn.Linear(L2,L3),
    torch.nn.ReLU(),
    torch.nn.Linear(L3,L4),
)

lossfn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
# print(list(model.parameters()))
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

gamma = 0.9
epsilon = 1.0
epochs = 1000
losses = []

for i in range(epochs):
    percent_bar(i)
    game = Gridworld(size=4,mode = 'player')   
    state1_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64) / 10.0 #np array with noise
    state1 = torch.from_numpy(state1_).float() #torch tensor
    
    status = 1 #flag for each round
    
    while(status == 1):

         qval = model(state1)
         qval_ = qval.data.numpy()
         
         if np.random.rand() < epsilon :
             action_int = np.random.randint(4)
         else:
             action_int = np.argmax(qval_)
         
         action = action_set[action_int]
         
         if i % 100  == 0 :
            print(game.board.render())
            print(action)   
         
         game.makeMove(action)
         
         
         state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10
         state2 = torch.from_numpy(state2_).float()
         
         reward = game.reward()
         with torch.no_grad() :
             newQ = model(state2.reshape(1,64))
         maxQ = torch.max(newQ)   
         if reward == -1 : 
             Y = reward + (gamma * maxQ) 
         else :
             Y  = reward 
         Y = torch.Tensor([Y]).detach() #TD target
         X = qval.squeeze()[action_int] #TD predict
         loss = lossfn(X,Y) #TD error
         if i%100 == 0:
            print(i ,loss.item())
            clear_output(wait = True)
         optimizer.zero_grad()
         loss.backward() # calculate parital derivative and pass backward until the input
         optimizer.step()
         state1 = state2
         if abs(reward) == 10 :
             status = 0
    losses.append(loss.item())
    
    if epsilon > 0.1 :
        epsilon -=(1/epochs)
# make_dot(loss, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
# dot.render(f"model_graph_{i}")
    
plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("epochs")
plt.ylabel("loss")
test_model(model,"player",True)
plt.show()
