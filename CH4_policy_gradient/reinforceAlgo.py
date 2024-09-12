# import gym
import gymnasium as gym
from gym.utils.play import play
import torch 
import numpy as np
from typing import Type
from matplotlib import pyplot as plt 




def DiscountRewardArray(rewards,gamma = 0.99):
    totalSteps = len(rewards)
    discReturn = torch.pow(gamma,torch.arange(totalSteps).float()) * rewards
    for i in range(totalSteps):
        
        Rewardt = 0
        for j in range(i,totalSteps):
            Rewardt += discReturn[j] 
        discReturn[i] = Rewardt  

    discReturn /= discReturn.max()
    # 正規化 reward 到 0~1 區間 提升訓練穩定性
    return discReturn

def loss_fn(preds,r):
    return -1 * torch.sum( r * torch.log(preds) )   

L1 = 4 
L2 = 150 
L3 = 2

model = torch.nn.Sequential(
    torch.nn.Linear(L1,L2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(L2,L3), 
    torch.nn.Softmax()
)
# 
lr = 0.009
optimizer = torch.optim.Adam(model.parameters(),lr)
# optimizer = torch.optim.SGD(model.parameters(),lr)
 



env = gym.make('CartPole-v1')

max_step = 300 # 200 steps of no filping stands for already learned
max_episodes = 1000 #max epochs to train


score = []

for episode in range(max_episodes):
    curr_state = env.reset()
    curr_state = curr_state[0]
    done = False
    transitions = []
    for step in range(max_step):
       
        act_prob = model(torch.from_numpy(curr_state).float())
        action = np.random.choice(np.array([0,1]),p=act_prob.data.numpy())
        prev_state = curr_state
        curr_state , _, done, _,info = env.step(action)
        
        #=======================
        if done == 0 : 
            reward = 1
        else :
            reward = -10
        #=======================
        
        
        transitions.append((prev_state,action,reward))   

        if done :
            print('step', episode,' :',step)
            break
    ep_len = len(transitions)
    score.append(ep_len)
    reward_batch = torch.Tensor([reward for (state,action,reward) in transitions])
    
    disc_returns = DiscountRewardArray(reward_batch)
 
    state_batch = torch.Tensor([state for (state,action,reward) in transitions])
    action_batch = torch.Tensor([action for (state,action,reward) in transitions])
    pred_batch : Type[torch.Tensor]= model(state_batch)
    prob_batch = pred_batch.gather(dim=1,index= action_batch.long().view(-1,1)).squeeze()
    
    loss = loss_fn(prob_batch,disc_returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
plt.figure(figsize=(10,7))
plt.scatter(np.arange(len(score)),score)
plt.show()


wining = 0
for i in range(100):
    state_test = env.reset()[0]
    status = 1 
    for i in range(200):
        action = model(torch.from_numpy(state_test).float())
        action = np.random.choice(np.array([0,1]),p=action.data.numpy())
        state_test , _, done, _,info = env.step(action)
        if done :
            break
    if i == 199 :
        wining += 1
print(wining)  