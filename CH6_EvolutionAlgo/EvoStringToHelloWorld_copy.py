import random 
import numpy as np
from matplotlib import pyplot as plt

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
alphabetList = list(alphabet)
target = "Hello World"
popSize = 300
generations = 5000

def minDistance(word1, word2):
    editDistanceList = []
    for i in range(len(word1) + 1):
        eachList = []
        for j in range(len(word2) + 1):
            if i == 0:
                eachList.append(j)
            else:
                if j == 0:
                    eachList.append(i)
                else:
                    if word1[i-1] == word2[j-1]:
                        minDist = editDistanceList[i-1][j-1]
                    else:
                        minDist = editDistanceList[i-1][j-1] + 1
                        if ((editDistanceList[i-1][j] + 1) < minDist):
                            minDist = editDistanceList[i-1][j] + 1
                        if ((eachList[j-1] + 1) < minDist):
                            minDist = eachList[j-1] + 1
                    eachList.append(minDist)
        editDistanceList.append(eachList)
    return editDistanceList[len(word1)][len(word2)]

class Chromosome:
    def __init__(self, string):
        self.string = string
        self.fitness = minDistance(self.string, target)

def initializeChromosomes(populationSize, length):
    chromosomes = []
    for _ in range(populationSize):
        eachChromosome = Chromosome("".join(np.random.choice(alphabetList, length)))
        chromosomes.append(eachChromosome)
    return chromosomes

def crossover(chromosome1, chromosome2, crossoverProb=0.8):
    if np.random.rand() < crossoverProb:
        crossoverPoint = np.random.randint(1, len(chromosome1.string))
        child1 = chromosome1.string[:crossoverPoint] + chromosome2.string[crossoverPoint:]
        child2 = chromosome2.string[:crossoverPoint] + chromosome1.string[crossoverPoint:]
        return Chromosome(child1), Chromosome(child2)
    else:
        return chromosome1, chromosome2

def mutation(eachChromosome, mutationProb=0.4):
    charList = list(eachChromosome.string)
    for i in range(len(eachChromosome.string)):
        if np.random.rand() < mutationProb:
            charList[i] = np.random.choice(alphabetList)
    return Chromosome("".join(charList))

def eachGeneration(chromosomes, popSize):
    offSpring = []
    fitnessSum = sum([1 / (eachChromosome.fitness + 1) for eachChromosome in chromosomes])
    p = [1 / (eachChromosome.fitness + 1) / fitnessSum for eachChromosome in chromosomes]
    while len(offSpring) < popSize:
        parents = np.random.choice(chromosomes, 2, p=p, replace=False)
        # print(parents[0].string,parents[1].string)
        child1, child2 = crossover(parents[0], parents[1])
        # print(child1.string,child2.string)
        child1 = mutation(child1)
        child2 = mutation(child2)
        # print(child1.string,child2.string)
        offSpring.append(child1)
        offSpring.append(child2)
    return offSpring[:popSize]

def GA(target, popSize, generations):
    chromosomes = initializeChromosomes(popSize, len(target))
    minLossInEachGeneration = []
    for i in range(generations):
        chromosomes = eachGeneration(chromosomes, popSize)
        minLoss = min([eachChromosome.fitness for eachChromosome in chromosomes])
        minLossInEachGeneration.append(minLoss)
        print(f"generation {i} : {minLoss}")
        if minLoss == 0:
            break
    plt.plot(minLossInEachGeneration)
    plt.xlabel('Generation')
    plt.ylabel('Minimum Loss')
    plt.title('Minimum Loss in Each Generation')
    plt.show()

GA(target, popSize, generations)
