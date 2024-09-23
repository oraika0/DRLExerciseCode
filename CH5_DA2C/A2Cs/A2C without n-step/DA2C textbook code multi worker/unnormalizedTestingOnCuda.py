import torch 
import numpy as np 
import gymnasium as gym
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import os

# 將模型、資料轉換到 GPU 上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def percent_bar(i, epochs):
    progress = (i + 1) / epochs
    bar_length = 50
    progress_chars = int(progress * bar_length)
    bar = '[' + '#' * progress_chars + '-' * (bar_length - progress_chars) + ']'
    percent = '{:.2%}'.format(progress)
    print(f'\r{bar} {percent}', end='', flush=True)
        
class ActorCritic(torch.nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = torch.nn.Linear(4, 25)
        self.l2 = torch.nn.Linear(25, 50)
        
        self.actor_l1 = torch.nn.Linear(50, 2)
        self.l3 = torch.nn.Linear(50, 25)
        self.critic_l1 = torch.nn.Linear(25, 1)
        
    def forward(self, x):
        y = torch.nn.functional.relu(self.l1(x))
        y = torch.nn.functional.relu(self.l2(y))
        
        actor = torch.nn.functional.log_softmax(self.actor_l1(y), dim=0)
        c = torch.nn.functional.relu(self.l3(y.detach()))
        critic = self.critic_l1(c)
        return actor, critic

def worker(t, worker_model, counter, params):
    worker_env = gym.make('CartPole-v1')
    worker_env.reset()
    worker_opt = torch.optim.Adam(lr=1e-4, params=worker_model.parameters())
    worker_model.to(device)  # 將模型轉移到 GPU
    worker_opt.zero_grad()
    
    for i in range(params['epochs']):    
        worker_opt.zero_grad()
        values, logprobs, rewards, length = runEpisode(worker_env, worker_model)
        actor_loss, critic_loss, eplen = update_params(worker_opt, values, logprobs, rewards)
        counter.value = counter.value + 1
        
        if (i+1) % 100 == 0:
            print("worker:", t, "=> ⌈", "epoch", i+1, ":", len(rewards), "⌋")
        buffer.put(length)
        print(f"\rsize: {buffer.qsize()}", end="", flush=True)
    print("worker", t, os.getpid(), "finish.")
        
def runEpisode(worker_env, worker_model):
    state = torch.from_numpy(worker_env.env.unwrapped.state).float().to(device)  # 轉到 GPU
    values, logprobs, rewards = [], [], []
    done = False
    while not done:
        policy, value = worker_model(state)
        values.append(value)
        
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        
        state_, _, done, _, _ = worker_env.step(action.detach().cpu().numpy())  # 將 action 轉回 CPU 執行
        state = torch.from_numpy(state_).float().to(device)  # 轉回 GPU
        reward = -10 if done else 1.0
        rewards.append(reward)
        if done:
            worker_env.reset()
            
    return values, logprobs, rewards, len(rewards)

def update_params(worker_opt, values, logprobs, rewards, clc=0.1, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1).to(device)  # 轉換到 GPU
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1).to(device)
    values = torch.stack(values).flip(dims=(0,)).view(-1).to(device)
    
    Returns = []
    ret_ = torch.Tensor([0]).to(device)
    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)
    
    Returns = torch.stack(Returns).view(-1).to(device)
    
    actor_loss = -1 * logprobs * (Returns - values.detach())
    critic_loss = torch.pow(values - Returns, 2)
    loss = actor_loss.sum() + clc * critic_loss.sum()
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)

# -------------------------------------------------------------
# 主程式部分

MasterNode = ActorCritic().to(device)  # 主模型也要放到 GPU
MasterNode.share_memory()

processes = []
params = {
    'epochs': 1000,
    'n_workers': 6,
}

counter = mp.Value('i', 0)
buffer = mp.Manager().Queue()

# main training 
for i in range(params['n_workers']):
    p = mp.Process(target=worker, args=(i, MasterNode, counter, params))
    p.start()
    processes.append(p)

print("showing all subprocess")
for p in processes:
    print(p)
print("-----------------------------------")

for p in processes:
    print("joining", p)
    p.join()
    print("\nstuck at process", p, "end")

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
print("scoring")
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
    _,_,_,length = runEpisode(worker_envTest,MasterNode)
    trainedModelscore.append(length)
    
# Plot 2: Individual episode lengths
plt.figure(figsize=(17, 12))
plt.plot(trainedModelscore, color='red')
plt.title("Trained model test")
plt.xlabel("testing times")
plt.ylabel("Episode Length")
plt.show()   

# -------------------------------------------------------------    
    