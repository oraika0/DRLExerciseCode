import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt
def get_reward(prob):
    # 用prob 跟 random 10抽出 0~10之間的某一個值 ，且期望值符合prob
    reward = 0;
    for i in range(10):
        if np.random.rand() < prob:
            reward += 1;
    return reward;
def update_record(record,action,r): 
    #input : record : record list , action(int : #action_type = #bandit) :which bandit , r(int 0~10) : return reward this time 
    record[action,1] = (record[action,0]*record[action,1]+r)/(record[action,0]+1);
    record[action,0] += 1 ;
    return record
def get_best_arm(record):
    best_arm_index = np.argmax(record[:,1]);
    return best_arm_index;

number = 10;
eps = 0.2; # epsilon

fig,ax = plt.subplots(1,1)
ax.set_xlabel("Plays")
ax.set_ylabel("Average Reward")
fig.set_size_inches(9,5)

record = np.zeros((number,2)) # 10個list 每個2 element (拉幾次,ave_value)
probs = np.random.rand(number) # each prob for each bandit
print(probs)
rewards = [0] # ave_reward each time
for i in range(500):
    if random.random() > eps:
        choice = get_best_arm(record);
    else :
        choice = np.random.randint(number);
        
    reward = get_reward(probs[choice]);
    record = update_record(record,choice,reward);

    # [0,2,4] => (reward[len-1]*len + reward)/len+1
    mean_reward_thistime = (len(rewards)*rewards[len(rewards)-1] + reward)/(len(rewards)+1);
    # or mean_reward = ((i+1) * rewards[-1] + r)/(i+2) in example 
    
    rewards.append(mean_reward_thistime);
print(rewards[-1]);
ax.scatter(np.arange(len(rewards)),rewards);
plt.show()
print(1)
    
# record = update_record(record,1,2)
# print(record)

