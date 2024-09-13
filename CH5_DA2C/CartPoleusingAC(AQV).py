import torch 
import numpy as np 
import gymnasium as gym
import torch.multiprocessing as mp
import matplotlib 
import matplotlib.pyplot as plt
from typing import Type
from torchviz import make_dot, make_dot_from_trace
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
        
        self.l3Q = torch.nn.Linear(50,25)
        self.criticQ = torch.nn.Linear(25,1)
        
        self.l3V = torch.nn.Linear(50,25)
        self.criticV = torch.nn.Linear(25,1)
        #    4 -> 25 -> 50 -------> 2   actor : policy func
        #                L--> 25 -> 1   criticQ : Qvalue func
        #                L--> 25 -> 1   criticV : Vvalue func
        
        
    def forward(self,x):
        x = torch.nn.functional.normalize(x,dim=0) #A
        y = torch.nn.functional.relu(self.l1(x)) #linear calculate + relu
        y = torch.nn.functional.relu(self.l2(y))
        
        actor = torch.nn.functional.log_softmax(self.actor_l1(y),dim=0)
        cQ = torch.nn.functional.relu(self.l3Q(y.detach()))
        criticQ = torch.tanh(self.criticQ(cQ)) #B
        cV = torch.nn.functional.relu(self.l3V(y.detach()))
        criticV = torch.tanh(self.criticV(cV))
        return actor,criticQ,criticV
        {
        # A
        # torch.nn.functional.normalize : 
        # v : tensor input in size (n0 ,n1 , ... , nk)  
        # v = v / max( ||v|| )

        # B
        # Hyperbolic Tangent (Tanh)
        # output a value  -1~1
        # 單純cartpole 的 reward 就是 +1 or -1 符合情境所以拿來用
        # 更 : 之後又覺得不是 : 此 adventage為(  normalize(discounted total Reward) - hyperbolic tan(V-value)(critic output) )

        # 補充為什麼actor需要做log_softmax 而非softmax
        # log_softmax 說明 : 做softmax 後的每一個數值都取log 
        # => log(softmax(datas)) = log(e^datas / sigma(e^each_data) ) = log_softmax(datas)
        # == datas - log(sigma (eachdata )) 
        # 此數學化簡可以在取對數時可以避免數據overflow or underflow ，提高數值穩定性
        # page 5-23

        # 目前流程 : log_softmax -> Categorical(logits) 來輸入未正規化的資料 並輸出如softmax的結果
        # 我的想法 : 只做softmax -> Categorical(probs) 輸入已經正規畫成機率的資料 直接輸出機率分布
        # 我的想法容易使資料不穩定 雖然在數學式子裡面是等價的 但在程式中先取log會對穩定性更好
    
        }

# discrete  
def worker(t,worker_model,counter,params):
    worker_env = gym.make('CartPole-v1')
    worker_env.reset()
    worker_opt = torch.optim.Adam(lr=1e-4,params=worker_model.parameters())
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        # percent_bar(i ,params['epochs'] )        
        worker_opt.zero_grad()
        Qvalues,Vvalues,logprobs,rewards,length = runEpisode(worker_env,worker_model)
        actor_loss,critic_loss,eplen = update_params(worker_opt,Qvalues,Vvalues,logprobs,rewards)
        counter.value = counter.value + 1
        
        if( (i+1) % 100 == 0):
            print("worker :",t,"=> ⌈","epoch",i+1,":",len(rewards),"⌋")
        buffer.put(length)
        print(f"\rsize : {buffer.qsize()}",end="",flush=True)
    print("worker",t,os.getpid(),"finish.")
        
def runEpisode(worker_env,worker_model):
    state = torch.from_numpy(worker_env.env.unwrapped.state).float()
    Qvalues,Vvalues,logprobs,rewards = [],[],[],[]
    done = False
    j = 0 #not used now , only counting epochs now , can be used for j < n_Steps && done == False
    while(done == False):
        j += 1
        # policy : actor R^2
        # value : critic -1 ~ 1
        policy,Qvalue,Vvalue = worker_model(state)
        Qvalues.append(Qvalue)
        Vvalues.append(Vvalue)
        
        # logits : a raw , unnormalized output data from each layer of nn before it is passed to the activating function
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
        else :
            reward = 1.0
        rewards.append(reward)
        
    # V function(critic) output , picked action porb , reward , step length
    return Qvalues,Vvalues,logprobs,rewards,len(rewards)

# opt : optimizer
def update_params(worker_opt,Qvalues,Vvalues,logprobs,rewards,clc = 0.1 , gamma = 0.95):

    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1) # C
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    Returns = []
    ret_ = torch.Tensor([0]) # each return
    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)
        
    # Returns = torch.stack(Returns).view(-1)
    Returns = torch.stack(Returns)
    Returns = Returns.view(-1)
    Returns = torch.nn.functional.normalize(Returns,dim=0)
    
    {
        # C
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
    }
    
    actor_loss = -1 * logprobs * (Returns - values.detach())
    criticV_loss = torch.pow(Vvalues - Returns , 2)
    
    
    # 記得對齊 只會有
    criticQ_loss = torch.pow(Qvalues - (reward + Vvalues) )
    
    
    
    
    loss = actor_loss.sum() + clc * critic_loss.sum() # clc : actor critic 的重要程度調整
    loss.backward()
    worker_opt.step()
    return actor_loss,critic_loss,len(rewards)



# ----------------------------------------------------------------------
# main function 

MasterNode = ActorCritic()
MasterNode.share_memory()

processes = []
params = {
    'epochs': 3000,
    'n_workers': 1,
}

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


# graph and result
n = params['n_workers']
score = []
running_mean = []
total = torch.Tensor([0])
mean = torch.Tensor([0])
while not buffer.empty():
    score.append(buffer.get(timeout=10))
print("length :",len(score))
for i in range( params['epochs']):
    if (i >= 50):
        total = total - sum(score[n*(i-50) : n*(i-50)+n])/n
        total = total + sum(score[n*i : n*i + n])/n
        mean = int(total/50)
    else :
        total = total + sum(score[n*i : n*i + n])/n
        mean = int ( total/(i+1))
    running_mean.append(mean)
plt.figure(figsize=(17,12))
plt.ylabel("Mean episode length")
plt.xlabel("training epochs")
plt.plot(running_mean)
plt.show()        

print(counter.value,processes[0].exitcode)
    