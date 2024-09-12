import random 
import numpy as np
import torch
import gymnasium as gym
from matplotlib import pyplot as plt
env = gym.make("CartPole-v1")

device = torch.device("cuda")
print(torch.cuda.get_device_name(0))

{# x : cartPole state data(4) 
# 0 : pole pos
# 1 : velocity
# 2 : angle
# 3 : angluar velocity
}

def model (x,unpacked_params):
    l1,b1,l2,b2,l3,b3 = unpacked_params
    y = torch.nn.functional.linear(x,l1,b1)
    y = torch.relu(y)
    y = torch.nn.functional.linear(y,l2,b2)
    y = torch.relu(y)
    y = torch.nn.functional.linear(y,l3,b3)
    y = torch.log_softmax(y,dim= 0)
    return y

def unpack_params(params,layers=[(25,4),(10,25),(2,10)]):
    unpack_params = []
    e= 0
    for i ,layer in enumerate(layers):
        s,e = e, e+np.prod(layer)
        
        weights = params[s:e].view(layer)
        
        s,e = e,e+layer[0]
    
        bias = params[s:e]
        
        unpack_params.extend([weights,bias])
        
    return unpack_params

def spawn_population(N,size):
    # N : popSize
    # size : chromosome length 
    pop = []
    for i in range(N):
        vec = torch.randn(size).to(device) / 2.0
        fit = 0
        p = {'params':vec,'fitness' : fit}
        pop.append(p)
    return pop
    
def crossover(x1,x2):
    x1 = x1['params']
    x2 = x2['params']
    n = x1.size()[0]
    split_pt = np.random.randint(n)
    # may choose head(or tail) that cause not crossover
    child1 = torch.zeros(n).to(device)
    child2 = torch.zeros(n).to(device)
    
    child1[:split_pt] = x1[:split_pt]
    child1[split_pt:] = x2[split_pt:]
    child2[:split_pt] = x1[:split_pt]
    child2[split_pt:] = x2[split_pt:]
    
    c1 = {'params' : child1,'fitness' : 0.0}
    c2 = {'params' : child2,'fitness' : 0.0}
    
    return c1,c2

def mutation(x,mutProb):
    x_ = x['params']
    numToChange = np.random.binomial(x_.shape[0],mutProb)
    idx = np.random.randint(low=0,high=x_.shape[0],size=(numToChange))
    
    x_[idx] += torch.randn(len(idx)).to(device) / 10.0
    x['params'] = x_
    return x
    # editted version
    # from fixxed changeNum change to binomial num
    # from random randn to += randn 
    # from loop x_.shape[0] times to random.binomoal    

def test_model(agent):
    done = False
    state = torch.from_numpy(env.reset()[0]).float().to(device)
    score = 0
    while not done : 
        params = unpack_params(agent['params'])
        probs = model(state,params) 
        action = torch.distributions.Categorical(probs=probs).sample()
        #sample() outputs a torch     
        state_,reward,done,_,info = env.step(action.item())
        state = torch.from_numpy(state_).float().to (device)
        score += 1
    return score

def evaluatePopulation(pop):
    totFit = 0
    bestFit = 0
    bestAgent = None
    for agent in pop:
        score = test_model(agent)
        agent['fitness'] = score
        totFit += score
        if score > bestFit:
            bestFit = score
            bestAgent = agent
    avgFit = totFit / len(pop)
    return avgFit,bestAgent

def next_generation(pop,mutProb,tournamentSize):
    # using tournament selection
    # SGA (survives only generation)
    
    # tournamentSize : a rate of tournamentPop/wholePop 
    # float : (0,1)
    newPop = []
    popSize = len(pop)
    while len(newPop) < popSize:
        # rids : random indexs
        rids = np.random.randint(low = 0 ,high = popSize,size = int(popSize*tournamentSize))
        batch = np.array([[i,x['fitness']] for (i,x) in enumerate(pop) if i in rids])
        # enumerate returns a tuple that first stands for the counter of this iterable object(pop) and the second stands for the ith element in the object 
        # so the whole line means that : 
        #   in enuerate(pop) , track the whole pop and returns (index(i),agent(x)) in every loop
        #   if index(i) exists in rids , then append [index,agent['fitness']] to batch 
        #   so the final batch will looks like : [ [1,{params : "...", 'fit' : int}] , [...] , ....]

        scores = batch[batch[:,1].argsort()]
        # sort agent using agent['fitness']
        i0 , i1 = int(scores[-1][0]),int(scores[-2][0]) 
        # get the best 2 agent's params
        parent0 ,parent1 = pop[i0] , pop[i1]
        offSpring_ = crossover(parent0,parent1)
        child1 = mutation(offSpring_[0],mutProb=mutProb)
        child2 = mutation(offSpring_[1],mutProb=mutProb)
        offSpring = [child1,child2]
        newPop.extend(offSpring)
    return newPop

def GABP():    
    generations = 20
    popSize = 500
    mutProb = 0.008
    popAvgFit = []
    popBestFit = []
    tournamentSize = 0.2

    pop = spawn_population(popSize,407)
    
    # 407 = 2*25 + 25*10 + 10*2 (weight)+25 + 10 + 2(bias) 
    for i in range(generations):
        avgFit , bestAgent = evaluatePopulation(pop)
        pop = next_generation(pop,mutProb,tournamentSize)
        popAvgFit.append(avgFit)
        popBestFit.append(bestAgent['fitness'])
        print('generations :' ,i, bestAgent['fitness'])
            
    plt.plot(popAvgFit, label='AvgFitness', color='blue')
    plt.plot(popBestFit, label='BestFitness', color='green')
    plt.xlabel('generations')
    plt.ylabel('fitness')
    plt.title('CartPole using GABP')
    plt.legend()
    plt.show()    
    
GABP()
