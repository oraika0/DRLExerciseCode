import numpy as np 
def nn(x , w1 ,w2):
    l1 = x @ w1
    l1 = np.maximum(0, l1)
    l2 = l1 @ w2
    l2 = np.maximum(0 , l2)
    return l2 

w1  = np.random.randn(784,200)
w2  = np.random.randn(200,10)
x = np.random.randn(784)

print(nn(x,w1,w2))

