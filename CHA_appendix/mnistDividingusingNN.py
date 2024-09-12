import numpy as np
import torch
import torchvision as TV

from six.moves import urllib
opener = urllib.request.builder_opener()
opener.addheaders = [('User','Google_Chrome')]
urllib.request.install_opener(opener)

mnist_data = TV.datasets.MNIST("MNIST",train,train=TRUE,download = True)
lr = 0.0001
epochs = 1
batch_size = 1000
losses = []
lossfn = torch.nn.CrossEntropyLoss()
for i in range(epochs):
    rid = np.ramdom.randint(0,mnist_data.train_data.shape[0],size = batch_size)