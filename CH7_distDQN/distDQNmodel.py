import numpy as np
import torch
from matplotlib import pyplot as plt    


def update_dist(reward, support,probs,lim=(-10.,10.) , gamma = 0.8) :
    # probs : tensor
    supNum = support.shape[0]
    supMin = lim[0]
    supMax = lim[1]
    # dz : supportStep (delta z)
    dz = (supMax - supMin) / (supNum-1.)
    # bj : bucket index(sepreate continuous space into several buckets(a part of continuous segment eg. [0.4,0.8) )
    bj = np.round(reward-supMin)/dz
    bj = int(np.clip(bj,0,supNum-1))

    m = probs.clone()
    
    j = 1
    for i in range(bj,1,-1):
        # update from reward index to 1
        m[i] += np.power(gamma,j) * m[i-1]
        j+=1
    
    j = 1
    for i in range(bj,supNum-1,1):
        # update from reward index to end 
        m[i] += np.power(gamma , j) * m[i+1]
        j+= 1
        
    m /= m.sum()
    # normalize to sum == 1
    return m


supMin = -10
supMax = 10
supNum = 51

support , supStep= np.linspace(supMin,supMax,supNum,retstep=True)
probs = np.ones(supNum)
probs = probs / probs.sum()
# z3 = torch.from_numpy(probs).float()

# ob_reward = -1 #假設觀測到的回饋值為-1
# Z = torch.from_numpy(probs).float()
# Z = update_dist(ob_reward,torch.from_numpy(support).float(),Z,lim=(supMin,supMax),gamma=0.8) #更新機率分佈
# plt.bar(support,Z)
# plt.show()


# ob_rewards = [ 10,10,10,0,1,0,1,-10,-10,10,10]
ob_rewards = [5]
ob_rewards = ob_rewards * 15

Z = torch.from_numpy(probs)
for i in range(len(ob_rewards)):
    Z = update_dist(ob_rewards[i],torch.from_numpy(support),Z,lim=(supMin,supMax),gamma=0.7)
plt.bar(support,Z)
plt.show()