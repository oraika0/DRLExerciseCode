# import gym
import gymnasium as gym
from gym.utils.play import play
import torch 
import numpy as np
from typing import Type
from matplotlib import pyplot as plt 
def DiscountRewardArray(rewards,gamma = 0.995):
    totalSteps = len(rewards)
    # discReturn = torch.pow(gamma,torch.arange(totalSteps).float().flip(dims=(0,))) * rewards
    discReturn = torch.pow(gamma,torch.arange(totalSteps).float()) * rewards
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
    torch.nn.Softmax(dim = 0)
)

lr = 0.009
optimizer = torch.optim.Adam(model.parameters(),lr)


# Returns:
#             observation (object): this will be an element of the environment's :attr:`observation_space`.
#                 This may, for instance, be a numpy array containing the positions and velocities of certain objects.
#             reward (float): The amount of reward returned as a result of taking the action.
#             terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
#                 In this case further step() calls could return undefined results.
#             truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
#                 Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
#                 Can be used to end the episode prematurely before a `terminal state` is reached.
#             info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
#                 This might, for instance, contain: metrics that describe the agent's performance state, variables that are
#                 hidden from observations, or individual reward terms that are combined to produce the total reward.
#                 It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
#                 of returning two booleans, and will be removed in a future version.

# state
# Num Observation Min Max
# 0   Cart Position   -4.8    4.8
# 1   Cart Velocity   -Inf    Inf
# 2   Pole Angle  ~ -0.418 rad (-24°) ~ 0.418 rad (24°)
# 3   Pole Angular Velocity   -Inf    Inf


env = gym.make('CartPole-v1')
# state1 = env.reset()
# pred = model(torch.from_numpy(state1[0]).float())
# action = np.random.choice(np.array([0,1]),p=pred.data.numpy())
# state2 , reward , done , truncated ,info  = env.step(action)

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
        curr_state , r, done, _,info = env.step(action)
        transitions.append((prev_state,action,step+1))   

        if done :
            print('step', episode,' :',step)
            break
    ep_len = len(transitions)
    score.append(ep_len)
    reward_batch = torch.Tensor([reward for (state,action,reward) in transitions])
    
    # -------------------------------------------------------------------------
    reward_batch = reward_batch.flip(dims=(0,))
    disc_returns = DiscountRewardArray(reward_batch)
    # 原本的我覺的很怪 有問題 自己改成下面的
    
    # disc_returns = DiscountRewardArray(reward_batch)
    # disc_returns = disc_returns.flip(dims=(0,))
    # --------------------------------------------------------------------------
    
    state_batch = torch.Tensor([state for (state,action,reward) in transitions])
    action_batch = torch.Tensor([action for (state,action,reward) in transitions])
    pred_batch : Type[torch.Tensor]= model(state_batch)
    prob_batch = pred_batch.gather(dim=1,index= action_batch.long().view(-1,1)).squeeze()
    
    if episode == max_episodes-1:
        a = 1 
    
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