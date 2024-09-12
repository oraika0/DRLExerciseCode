import numpy as np
import torch
import random
from typing import Type  # 導入Type類型
import matplotlib.pyplot as plt

class ContextBandit:
    def __init__(self,arms = 10):
        self.arms = arms
        self.init_distribution(arms)
        self.update_state()
    def init_distribution(self,arms):
        states = arms
        self.bandit_matrix = np.random.rand(states,arms)
        print(self.bandit_matrix)
    def reward(self,prob):
        reward = 0
        for i in range(self.arms):
            if random.random() < prob :
                reward += 1
        return reward
    def update_state(self):
        self.state = np.random.randint(0,self.arms)
    def get_state(self):
        return self.state
    def get_reward(self,arm):
        return self.reward(self.bandit_matrix[self.get_state(),arm])
    def choose_arm(self,arm):
        reward = self.get_reward(arm)
        self.update_state()
        return reward
def softmax(vals,tau): #vals : value array tau :temputure
    softm_pr = pow(np.e , vals/tau) / (sum( pow(np.e , vals/tau) ))
    return softm_pr
def one_hot(N,pos,val = 1): #N:onehot vector     length , pos : state是哪一個 v貓: one hot要設多少:正常定義是1 
    one_hot_vector = np.zeros(N,dtype=float)
    one_hot_vector[pos] = val
    return one_hot_vector
def training(env : Type[ContextBandit],epochs = 10000, learning_rate = 1e-2):
    cur_state = torch.Tensor(one_hot(arms,env.get_state()))
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    rewards = []
    for i in range(epochs):
        y_pred = model(cur_state)
        av_softmax = softmax(y_pred.detach().numpy(),tau = 1.12)
        choices = np.random.choice(arms,p=av_softmax)
        cur_reward = env.choose_arm(choices)
        one_hot_reward = y_pred.data.numpy().copy()
        one_hot_reward[choices] = cur_reward
        reward = torch.tensor(one_hot_reward)
        rewards.append(cur_reward)
        loss = loss_fn(y_pred,reward)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_state = torch.Tensor(one_hot(arms,env.get_state()))

    return np.array(rewards)
def running_mean(x,N):
    c = x.shape[0] - N
    y = np.zeros(c,dtype=float)
    conv = np.ones(N)
    for i in range(c): 
        y[i] = (x[i:i+N]@conv ) / N 
    return y

arms = 10
N = 1 # batch size
D_in = D_out = arms 
H = 100
model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out),
    torch.nn.ReLU(),
)
loss_fn = torch.nn.MSELoss()
env = ContextBandit(arms)
rewards = training(env)
plt.figure(figsize=(16,7))
plt.xlabel("Training Epochs",fontsize = 14)
plt.ylabel("average reward",fontsize = 14)

plt.plot(running_mean(rewards,N = 100))
plt.show()