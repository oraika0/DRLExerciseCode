# finish 
# textbook code

import torch 
import numpy as np 
import gymnasium as gym
import torch.multiprocessing as mp
import matplotlib 
import matplotlib.pyplot as plt
from typing import Type
import os
import time

def percent_bar(i,epochs):
        progress = (i + 1) / epochs
        bar_length = 50
        progress_chars = int(progress * bar_length)
        bar = '[' + '#' * progress_chars + '-' * (bar_length - progress_chars) + ']'
        percent = '{:.2%}'.format(progress)
        print(f'\r{bar} {percent}', end='', flush=True)
        
class ActorCritic(torch.nn.Module):
    def __init__(self):
        super(ActorCritic,self).__init__()
        self.l1 = torch.nn.Linear(4,25)
        self.l2 = torch.nn.Linear(25,50)
        self.actor_l1 = torch.nn.Linear(50,2)
        self.l3 = torch.nn.Linear(50,25)
        self.critic_l1 = torch.nn.Linear(25,1)
#    4 -> 25 -> 50 -------> 2   actor : policy func
#                L--> 25 -> 1   critic : value func

    def forward(self,x):
        x = torch.nn.functional.normalize(x,dim=0)
        y = torch.nn.functional.relu(self.l1(x)) #linear calculate + relu
        y = torch.nn.functional.relu(self.l2(y))
        actor = torch.nn.functional.log_softmax(self.actor_l1(y),dim=0)
        c = torch.nn.functional.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_l1(c))
        return actor,critic

def worker(t,worker_model,counter,params):
    worker_env = gym.make('CartPole-v1')
    worker_env.reset()
    worker_opt = torch.optim.Adam(lr=1e-4,params=worker_model.parameters())
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        # percent_bar(i ,params['epochs'] )        
        worker_opt.zero_grad()
        tot_rew = torch.Tensor([0]) 
        values,logprobs,rewards,length,G,check = runEpisode(worker_env,worker_model)
        actor_loss,critic_loss,tot_rew = update_params(worker_opt,values,logprobs,rewards,G)
        while(check == 0):
            worker_opt.zero_grad()
            values,logprobs,rewards,length,G,check = runEpisode(worker_env,worker_model)
            actor_loss,critic_loss,eplen = update_params(worker_opt,values,logprobs,rewards,G)
            tot_rew += eplen
        
        counter.value = counter.value + 1
        
        if( (i+1) % 100 == 0):
            print("worker :",t,"=> ⌈","epoch",i+1,":",tot_rew,"⌋")
        buffer.put(tot_rew)
        print(f"\rsize : {buffer.qsize()}",end="",flush=True)
    print("worker",t,os.getpid(),"finish.")
        
def runEpisode(worker_env,worker_model):
    state = torch.from_numpy(np.array(worker_env.env.unwrapped.state)).float()
    values,logprobs,rewards = [],[],[]
    done = False
    j = 0 #not used now , only counting epochs now , can be used for j < n_Steps && done == False
    check = 1
    G = torch.Tensor([0])
    
    while(j < n_steps and done == False):
        j += 1
        # policy : actor R^2
        # value : critic -1 ~ 1
        policy,value = worker_model(state)
        values.append(value)
        # logits : a raw , unnormalized output data from each layer of nn before it is passed to the activatin function
        logits = policy.view(-1) #植基整理成一維向量 (-1 的為自己依照大小補齊的維度)
        # 但本來就是一維的了 我也不清楚有什麼用
        
        action_dist = torch.distributions.Categorical(logits=logits)
        # 用我的logits的情形作出一個分布情況
        action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_ , _, done, _,info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done : 
            reward = -10
            worker_env.reset() 
            check = 1
        else :
            reward = 1.0
            G = value.detach()
            check = 0
        rewards.append(reward)
    return values,logprobs,rewards,len(rewards),G,check
def testmodel(worker_env,worker_model):
    state = torch.from_numpy(np.array(worker_env.env.unwrapped.state)).float()
    values,logprobs,rewards = [],[],[]
    done = False
    j = 0 #not used now , only counting epochs now , can be used for j < n_Steps && done == False
    check = 1
    G = torch.Tensor([0])
    
    while(done == False):
        j += 1
        # policy : actor R^2
        # value : critic -1 ~ 1
        policy,value = worker_model(state)
        values.append(value)
        # logits : a raw , unnormalized output data from each layer of nn before it is passed to the activatin function
        logits = policy.view(-1) #植基整理成一維向量 (-1 的為自己依照大小補齊的維度)
        # 但本來就是一維的了 我也不清楚有什麼用
        
        action_dist = torch.distributions.Categorical(logits=logits)
        # 用我的logits的情形作出一個分布情況
        action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        # 不用np.choice的原因:保持在同一個框架不要亂跳 應該
        
        
        logprobs.append(logprob_)
        state_ , _, done, _,info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done : 
            reward = -10
            worker_env.reset() 
            check = 1
        else :
            reward = 1.0
            G = value.detach()
            check = 0
        rewards.append(reward)
    return values,logprobs,rewards,len(rewards),G,check

# 補充為什麼actor需要做log_softmax 而非softmax
# log_softmax 說明 : 做softmax 後的每一個數值都取log 
# => log(softmax(datas)) = log(e^datas / sigma(e^each_data) ) 
# == datas - log(sigma (eachdata )) 
# 此數學化簡可以在取對數時可以避免數據overflow or underflow ，提高數值穩定性

# 目前流程 : log_softmax -> Categorical(logits) 來輸入未正規化的資料 並輸出如softmax的結果
# 我的想法 : 只做softmax -> Categorical(probs) 輸入已經正規畫成機率的資料 直接輸出機率分布
# 我的想法容易使資料不穩定 雖然在數學式子裡面是等價的 但在程式中先取log會對穩定性更好


# opt : optimizer
def update_params(worker_opt,values,logprobs,rewards,G,clc = 0.1 , gamma = 0.95):
 
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    Returns = []
    ret_ = G    
    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma * ret_
        # ret_ =  gamma * ret_
        # a = rewards[r]
        # ret_ += a
        Returns.append(ret_)
        
    # Returns = torch.stack(Returns).view(-1)
    Returns = torch.stack(Returns)
    Returns = Returns.view(-1)
    Returns = torch.nn.functional.normalize(Returns,dim=0)
    
    # [ tensor([-10.]),
    #   tensor([-8.5000]),
    #   tensor([-7.0750]),
    #   tensor([-5.7212]), 
    #   tensor([-4.4352])  ]
    
    # to (stack)
    
    # tensor([[-10.0000],
    # [ -8.5000],
    # [ -7.0750],
    # [ -5.7212],
    # [ -4.4352]])
    
    # to (view)
    # tensor([-10.0000,  -8.5000,  -7.0750,  -5.7212,  -4.4352])
    
    actor_loss = -1 * logprobs * (Returns - values.detach())
    critic_loss = torch.pow(values - Returns , 2)
    loss = actor_loss.sum() + clc * critic_loss.sum()
    loss.backward()
    
    
    worker_opt.step()
    return actor_loss,critic_loss,len(rewards)



# ----------------------------------------------------------------------
# main function 

MasterNode = ActorCritic()
MasterNode.share_memory()

processes = []
params = {
    'epochs': 1000,
    'n_workers': 8,
}
n_steps = 80
counter = mp.Value('i',0)
buffer = mp.Manager().Queue()

# main training 
for i in range(params['n_workers']):
    p = mp.Process(target=worker,args = (i,MasterNode,counter,params))    
    p.start()
    processes.append(p)
    
print("showing all subprocess")
for p in processes:
    print(p)
print("-----------------------------------")


for p in processes:
    print("joining",p)
    p.join()
    print("\nstuck at process", p ,"end")

for p in processes:
    p.terminate()

# training finish
# -------------------------------------------------------------

# Correctly aggregate scores from all workers
n = params['n_workers']
score = []
running_mean = []

# Collect scores from the buffer
while not buffer.empty():
    score.append(buffer.get(timeout=10))

print("Total length of score:", len(score))

# Convert score to a list for processing
score = list(score)

for i in range(len(score)):
    if i >= 49:
        mean = sum(score[i - 49:i+1]) / 50
    else:
        mean = sum(score[:i+1]) / (i+1)
    running_mean.append(mean)


# Plot 1: Running mean of episode lengths
plt.figure(figsize=(17, 12))
plt.plot(running_mean, color='blue')
plt.title("Running Mean of Episode Lengths")
plt.xlabel("Training Episodes")
plt.ylabel("Mean Episode Length")
plt.show()




# graph and result
# n = params['n_workers']
# score = []
# running_mean = []
# total = torch.Tensor([0])
# mean = torch.Tensor([0])
# while not buffer.empty():
#     score.append(buffer.get(timeout=10))
# print("length :",len(score))
# for i in range( params['epochs']):
#     if (i >= 50):
#         total = total - sum(score[n*(i-50) : n*(i-50)+n])/n
#         total = total + sum(score[n*i : n*i + n])/n
#         mean = int(total/50)
#     else :
#         total = total + sum(score[n*i : n*i + n])/n
#         mean = int ( total/(i+1))
#     running_mean.append(mean)
    
# Plot 1: Running mean of episode lengths
# plt.figure(figsize=(17, 12))
# plt.plot(running_mean, color='blue')
# plt.title("Running Mean of Episode Lengths")
# plt.xlabel("Training Episodes")
# plt.ylabel("Mean Episode Length")
# plt.show()

# Plot 2: Individual episode lengths
plt.figure(figsize=(17, 12))
plt.plot(score, color='green')
plt.title("Episode Length per Episode")
plt.xlabel("Training Episodes")
plt.ylabel("Episode Length")
plt.show()       

# ------------------------------------------------------------
# test model
worker_envTest = gym.make('CartPole-v1')
worker_envTest.reset()
trainedModelscore= []
for i in range(500):    
    print("testing:",i)
    _,_,_,length,_,_ = testmodel(worker_envTest,MasterNode)
    trainedModelscore.append(length)
    
# Plot 2: Individual episode lengths
plt.figure(figsize=(17, 12))
plt.plot(trainedModelscore, color='red')
plt.title("Trained model test")
plt.xlabel("testing times")
plt.ylabel("Episode Length")
plt.show()   

# -------------------------------------------------------------    
    