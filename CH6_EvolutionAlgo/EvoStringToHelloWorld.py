import random 
import numpy as np
from matplotlib import pyplot as plt

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVXWYZ "
alphabetList = list(alphabet)
print(alphabetList)
target = "Hello World"
popSize = 500
generations = 3000

def minDistance(a,b):
    cnt = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            cnt += 1
    return cnt/len(a)
# 0/11 ~ 11/11 fitness
    return cnt

def minDistance2(word1, word2):
        editDistanceList = []
         
        for i in range(len(word1) + 1 ):
            eachList = []
            for j in range(len(word2) + 1 ):
                if i == 0:
                    eachList.append(j)
                else :
                    if j == 0 :
                        eachList.append(i)
                    else :
                        if word1[i-1] == word2[j-1] :
                            minDist = editDistanceList[i-1][j-1] #def : left
                        else :
                            minDist = editDistanceList[i-1][j-1] + 1 #def : left
                            if  ((editDistanceList[i-1][j] + 1) < minDist) :
                                minDist = editDistanceList[i-1][j] + 1
                            if ((eachList[j-1] + 1) < minDist) :
                                minDist = eachList[j-1] + 1
                        eachList.append(minDist)
            editDistanceList.append(eachList)
        return editDistanceList[len(word1)][len(word2)]

class chromosome:
    def __init__(self , string):
        self.string = string
        self.fitness = minDistance(self.string , target)
        
def initializeChromosomes(populationSize,length) :
    chromosomes = []
    for i in range(populationSize):
        EachChromosome = chromosome("".join(np.random.choice(alphabetList,length)))
        chromosomes.append(EachChromosome)
    return chromosomes

def crossover(chromosome1,chromosome2,crossoverProb = 0.8):
    if (np.random.rand() < crossoverProb):
        crossoverPoint = np.random.randint(1,len(chromosome1.string)) # for "abc" output 1 or 2
        child1 = "".join([chromosome1.string[:crossoverPoint],chromosome2.string[crossoverPoint:]])
        child2 = "".join([chromosome2.string[:crossoverPoint],chromosome1.string[crossoverPoint:]])
        return chromosome(child1), chromosome(child2)
    else:
        return chromosome1,chromosome2

def mutation(eachChromosome,mutationProb = 0.3):
    charList = list(eachChromosome.string)
    for i in range(len(eachChromosome.string)):
        if (np.random.rand() < mutationProb):
            charList[i] = np.random.choice(alphabetList)
    return chromosome("".join(charList))

def eachGeneration(chromosomes,popSize,maxChromosome):
    offSpring = []
    fitnessSum = sum([eachChromosome.fitness for eachChromosome in chromosomes])
    while(len(offSpring) < popSize - 1):
        p = [(eachChromosome.fitness/fitnessSum) for eachChromosome in chromosomes] 
        chromosome1 , chromosome2 = np.random.choice(chromosomes,2,p= p)
        chromosome1 , chromosome2 = crossover(chromosome1 , chromosome2)
        chromosome1 = mutation(chromosome1)
        chromosome2 = mutation(chromosome2)
        offSpring.append(chromosome1)
        offSpring.append(chromosome2)
    
    offSpring.append(maxChromosome)
    return offSpring[len(offSpring)-popSize:]

def GA(target , popSize , generations):
    maxFitnessInEachGeneration = []
    chromosomes = initializeChromosomes(popSize,len(target))
    for i in range(generations):
        maxFitness = max([eachChromosome.fitness for eachChromosome in chromosomes])
        maxChromosome = max(chromosomes,key=lambda chromo : chromo.fitness)
        chromosomes = eachGeneration(chromosomes,popSize,maxChromosome)
        maxFitnessInEachGeneration.append(maxFitness)
        print(f"generation{i} : {maxFitness} {maxChromosome.string}")
        if maxFitness == 11 :
            break
    
GA(target,popSize,generations)











# mutationProb : between 1/pop_size and 1/chromosome_length
# 1/100 ~ 1/11
