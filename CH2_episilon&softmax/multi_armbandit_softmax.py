import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt


fig,axis = plt.subplots(1,1) 
# we declare a 1*2 row*column grid , use axis[0,0] axis [0,1] to select where you wanna place the plot , like axis[0,1].scatter(np.random(10))
# axis for each plot 
# fig for the full window ,ex : fig.suptitle("all plots")
axis.set_xlabel("Plays")
axis.set_ylabel("Average Reward")
fig.set_size_inches(9,5)


def softmax(vals,tau): #vals : value array tau :temputure
    softm_pr = pow(np.e , vals/tau) / (sum( pow(np.e , vals/tau) ))
    return softm_pr
def get_reward(prob):
    # 用prob 跟 random 10抽出 0~10之間的某一個值 ，且期望值符合prob
    reward = 0;
    for i in range(10):
        if np.random.rand() < prob:
            reward += 1;
    return reward;


epochs = 500;
number = 10;
tau = 0.7;
# bandit_probs = np.random.rand(number)
bandit_probs = [0.33740031,0.68151394,0.1614269,0.0491745 , 0.82557627 ,0.63587592, 0.83411011 ,0.80015053 ,0.35110565 ,0.87850919]
print("bandit_probs : " , bandit_probs)
record = np.zeros((10,2))
rewards = [0]

counting_bar = np.zeros(number) 

for i in range(epochs):
    softmax_pr = softmax(record[:,1],tau)
    each_pick = np.random.choice(np.arange(number),p= softmax_pr)
    each_reward = get_reward(bandit_probs[each_pick])
    record[each_pick,1] = (record[each_pick,0]*record[each_pick,1] + each_reward) /( record[each_pick,0] +1)
    record[each_pick,0] += 1;
    each_ave_reward = (rewards[-1]*(len(rewards)-1) + each_reward)/len(rewards)
    rewards.append(each_ave_reward)
    counting_bar[each_pick] += 1;

print(rewards[-1])
axis.scatter(np.arange(len(rewards)),rewards)
# ax.bar(np.arange(number),counting_bar)
plt.show()