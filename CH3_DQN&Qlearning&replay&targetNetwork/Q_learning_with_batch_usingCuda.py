from TestingEnvs.Gridworld import Gridworld
import numpy as np
import torch 
from IPython.display import clear_output
import random
from matplotlib import pylab as plt
from typing import Type
from collections import deque
import os

action_set = {
    0 : 'u',
    1 : 'd',
    2 : 'l',
    3 : 'r',
}
def percent_bar(i,epochs):
        progress = (i + 1) / epochs
        bar_length = 50
        progress_chars = int(progress * bar_length)
        bar = '[' + '#' * progress_chars + '-' * (bar_length - progress_chars) + ']'
        percent = '{:.2%}'.format(progress)
        print(f'\r{bar} {percent}', end='', flush=True)

# discount :gamma
def training_loop(epochs,game_size,losses,mode = 'random', learning_rate = 1e-3 ,discount = 0.9, mem_list_size = 1000,batch_size = 200):
    L1 = game_size ** 2 * 4 
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
    
    model.to(device)
    
    
    
    epsilon = 1.0 #decrease from 1 , minus 1/epochs each round  
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    mem_list = deque(maxlen=mem_list_size)
    # esch mem tuple (state(t) , action ,reward(t+1) , state(t+1) , done )
    for i in range(epochs):
        percent_bar(i,epochs)
        game = Gridworld(size=game_size,mode=mode)
        state1_ = game.board.render_np().reshape(1,game_size ** 2 * 4)+ np.random.rand(1 , game_size ** 2 * 4)/100
        state1 = torch.from_numpy(state1_).float().to(device)
        status = 1
        move = 0
        while(status == 1) :
            move += 1
            qval:Type[torch.Tensor] = model(state1)
            qval_ = qval.data.cpu().numpy()

            # epsilon-greedy 
            if np.random.rand() < epsilon :
                action_int = np.random.randint(0,4)
            else :
                action_int = np.argmax(qval_)
            action = action_set[action_int]

            if i % 950 == 0 :
                print(game.board.render())
                print(action)
            
            game.makeMove(action)
            state2_ = game.board.render_np().reshape(1,game_size ** 2 * 4)+np.random.rand(game_size ** 2 * 4)/100
            state2 = torch.from_numpy(state2_).float().to(device)
            reward = game.reward()
            done = True if reward > 0 else False
            
            exp = (state1 , action_int , reward , state2 , done)
            mem_list.append(exp)
            # 重要: 更新state 忘記了
            state1 = state2

            if len(mem_list) >= batch_size :
                minibatch = random.sample(mem_list,batch_size)
                batch_state1 = torch.cat([ state1 for (state1 , action_int , reward , state2 , done) in minibatch]).to(device)
                batch_action_int = torch.Tensor([ action_int for (state1 , action_int , reward , state2 , done) in minibatch]).to(device)
                batch_reward = torch.Tensor([ reward for (state1 , action_int , reward , state2 , done) in minibatch]).to(device)
                batch_state2 = torch.cat([ state2 for (state1 , action_int , reward , state2 , done) in minibatch]).to(device)
                batch_done = torch.Tensor([ done for (state1 , action_int , reward , state2 , done) in minibatch]).to(device)
               
                batch_qval : Type[torch.Tensor]= model(batch_state1)
                
                with torch.no_grad() :
                    batch_TD_target = model(batch_state2) #input : Tensor[200,64] output : [200,4]
                batch_TD_target = batch_reward + discount * ((1-batch_done) * torch.max(batch_TD_target,dim=1)[0])
                #Tensor[200]      Tensor[200]    float      1-(0 or 1)       Tensor[200,64]->Tensor[200,2]->Tensor[200]
                batch_TD_predict = batch_qval.gather(dim=1,index=batch_action_int.long().unsqueeze(dim=1)).squeeze()
                loss = lossfn(batch_TD_predict,batch_TD_target.detach())
                # print(i,loss.item())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())        
                optimizer.step()
        
            if abs(reward) == 10 or move >= 50 :
                status = 0
                move = 0     
        if epsilon >= 0.1 :
            epsilon -= 1/epochs    
    losses = np.array(losses)
    
    return model
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

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda")
print(torch.cuda.get_device_name(0))

mode = 'random'
game_size = 4
losses = []
model = training_loop(epochs = 5000 , game_size = game_size , losses=losses,mode=mode)
plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("epochs",)
plt.ylabel("loss")
plt.show()


loops = 1000
win_count = 0.0
for i in range (loops):
    result = test_model(model=model,game_size= game_size,mode=mode,display=True)
    if result :
        win_count += 1.0
print(win_count/(loops))