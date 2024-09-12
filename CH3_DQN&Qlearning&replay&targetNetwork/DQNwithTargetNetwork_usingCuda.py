from TestingEnvs.Gridworld import Gridworld
import numpy as np
import torch 
from IPython.display import clear_output
import random
from matplotlib import pylab as plt
from typing import Type
from collections import deque
import copy
import time
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda")
print(torch.cuda.get_device_name(0)) 


def percent_bar(i,epochs) :
    barSize = 30 
    progress = (i/epochs)
    completePart = int(progress * barSize)
    incompletePart = barSize - completePart
    barOutpString = f"\r[{'#' * completePart}{'-' * incompletePart}]" 
    print(barOutpString + '{:.2%}'.format(progress) ,end=' ', flush=True)
def test_model(model,game_size = 4,mode="random",display = True):
    i = 0
    game = Gridworld(4, mode)
    state_ = game.board.render_np().reshape(1,64)+ np.random.rand(1,64)/100.
    state = torch.from_numpy(state_).float().to(device)
    
    status = 1
    while(status == 1):
        qval = model(state)
        qval_ = qval.data.cpu().numpy()
        action_ = np.argmax(qval_)
        action = action_set[action_]
        
        if(display) :    
            print(game.display())
            print('action : ' + action)
        
        game.makeMove(action)
        state_ = game.board.render_np().reshape(1,64)+np.random.rand(1,64)/100.0
        state = torch.from_numpy(state_).float().to(device)
        reward = game.reward()
        if(reward != -1):
            status = 0
            win = True if reward == 10 else False
            if (display):
                print("result : win") if win else print("result : lose")            
            return win 
        i += 1
        if  i >= 20 :
            status = 0
            if(display):
                print("out of time")
            win = False
            return win 
action_set = {
    0 : 'u',
    1 : 'd',
    2 : 'l',
    3 : 'r',
}
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
model2 = copy.deepcopy(model)
model.to(device)
model2.to(device)
model2.load_state_dict(model.state_dict())
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

gamma = 0.9
epsilon = 1.0

epochs = 5000
losses = []
mem_size = 1000
batch_size = 200
replay = deque(maxlen=mem_size)
max_move = 50
sync_freq = 500

t1 = time.time()

step_counting = 0
for i in range(epochs):
    percent_bar(i,epochs)
    game = Gridworld(size = 4 , mode='random')
    state1_ = game.board.render_np().reshape(1,64)+np.random.rand(1,64)/100
    state1 = torch.from_numpy(state1_).float().to(device)
    status = 1
    move = 0
    while(status == 1):
        step_counting += 1
        move += 1
        qval : Type[torch.Tensor]= model(state1)
        qval_ = qval.data.cpu().numpy()
        if(np.random.rand() < epsilon):
            action_int = np.random.randint(0,4)
        else:
            action_int = np.argmax(qval_)
        action = action_set[action_int]
        
        game.makeMove(action)
        reward = game.reward()
        done = True if reward > 0 else False
        state2_ = game.board.render_np().reshape(1,64)+np.random.rand(64)/100.0
        state2 = torch.from_numpy(state2_).float().to(device)
        exp = (state1 , action_int , reward , state2 , done)
        replay.append(exp)
        state1 = state2
        if len(replay) >= batch_size :
            mini_batch = random.sample(replay,batch_size)

            state1_batch = torch.cat([state1 for (state1 , action_int, reward, state2 , done) in mini_batch]).to(device)
            action_batch = torch.Tensor([action_int for (state1 , action_int, reward, state2 , done) in mini_batch]).to(device)
            reward_batch = torch.Tensor([reward for (state1 , action_int, reward, state2 , done) in mini_batch]).to(device)
            state2_batch = torch.cat([state2 for (state1 , action_int, reward, state2 , done) in mini_batch]).to(device)
            done_batch = torch.Tensor([done for (state1 , action_int, reward, state2 , done) in mini_batch]).to(device)
            qval_batch : Type[torch.Tensor] = model(state1_batch)
            with torch.no_grad():
                qval_batch2 = model2(state2_batch)
            TD_target = gamma * ((1-done_batch) * torch.max(qval_batch2,dim=1)[0])
            TD_target += reward_batch
            TD_predict = qval_batch.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(TD_predict,TD_target.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step_counting % sync_freq == 0 :
                # update target
                model2.load_state_dict(model.state_dict())
        if abs(reward) == 10 or move >= max_move :
            status = 0    
            move = 0
    if epsilon >= 0.05 :
        epsilon -= (1/epochs)
t2 = time.time()
timeUsed = t2 - t1
round(timeUsed,2)
print("time :",timeUsed) 

plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("steps")
plt.ylabel("loss")
plt.show()
winCounter = 0
for i in range(1000):
    result = test_model(model,4,'random',True)
    if result :
        winCounter += 1
print('win rate : ' , winCounter/1000 )
plt.show()

