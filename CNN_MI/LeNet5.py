import scipy
import torch
import numpy as np
import torch.nn as nn
from time import sleep
import torch.nn.init as init
from numpy import linalg as LA
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

# Class for the LeNet-5 network.

class LeNet5(nn.Module):
    def __init__(self, num_classes, activation):
        super(LeNet5, self).__init__()

        # First encoder
        self.layer1 = nn.Sequential(
                *([nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2, bias=True),
                   activation, ]))
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Second encoder
        self.layer2 = nn.Sequential(
                *([nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True),
                   activation, ]))
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        
        self.fc1 = nn.Sequential(*([
                nn.Dropout(),
                nn.Linear(16*5*5, 120),
                activation,]))
        self.fc2 = nn.Sequential(*([
                nn.Dropout(),
                nn.Linear(120, 84),
                activation,]))
        self.classifier = nn.Sequential(*([
                nn.Linear(84, num_classes),]))

        for m in self.modules(): # weight initialization.
            self.weight_init(m)

    def forward(self, x):                           # Forward pass for network.

        layers = []                                 # We collect the ouput of each layer in this list
        # Encoder 1
        layer1 = self.layer1(x)
        pool1 = self.pool1(layer1)

        layers.append(layer1)                       # Append the ouput of the first and second convolutional layer.
    
        # Encoder 2
        layer2 = self.layer2(pool1)
        pool2 = self.pool2(layer2)

        layers.append(layer2)

        # Classifier

        fc1 = self.fc1(pool2.view(pool2.size(0), -1)) # Reshape from (N, C, H, W) to (N, C*H*W) form.
        fc2 = self.fc2(fc1)
        classifier = self.classifier(fc2)

        layers.append(fc1)
        layers.append(fc2)
        layers.append(classifier)

        return classifier, layers                     # Returns the ouput of the network (Softmax not applied yet!)
                                                      # and a list containing the output of each layer.
        
    def weight_init(self, m):                         # Function for initializing wieghts.
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            init.constant_(m.bias.data, 1)
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight.data)
            init.constant_(m.bias.data, 1)
    
    def gram_matrix(self, x):                           # Calculate Gram matrix

#        sigma = 5*x.size(0)**(-1/(x.size(1)*x.size(2)))# Silverman's rule of Thumb

        if x.dim() == 2:                                # If the input is on matrix-form.
            k = x.data.cpu().numpy()
            k = squareform(pdist(k, 'euclidean'))       # Calculate Euclidiean distance between all samples.
            sigma = np.mean(np.mean(np.sort(k[:, :10], 1))) # Calculate kernel width based on 10 nearest neighbours.
            k = scipy.exp(-k ** 2 / sigma ** 2)         # RBF-kernel.
            k = k / np.float64(np.trace(k))             # Normalize kernel.
        if x.dim() == 4:                                # If the input is on tensor-form
            k = x[:, 0].view(x.size(0), -1).data.numpy()
            k = squareform(pdist(k, 'euclidean'))
            sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))
            k = scipy.exp(-k ** 2 / sigma ** 2)
            for i in range(x.size(1)-1):                # Loop through all feature maps.
                k_temp = x[:, i+1].view(x.size(0), -1).data.numpy()
                k_temp = squareform(pdist(k_temp, 'euclidean'))
                sigma = np.mean(np.mean(np.sort(k_temp[:, :10], 1)))
                k = np.multiply(k, scipy.exp(-k_temp ** 2 / sigma ** 2)) # Multiply kernel matrices together.

            k = k / np.float64(np.trace(k)) # Normalize final kernel matrix.
        return k # Return the kernel martix of x
    
    def renyi(self, x): # Matrix formulation of Renyi entropy.
        alpha = 1.01
        k = self.gram_matrix(x) # Calculate Gram matrix.
        l, v = LA.eig(k)        # Eigenvalue/vectors.
        lambda_k = np.abs(l)    # Remove negative/im values.

        return (1/(1-alpha))*np.log2(np.sum(lambda_k**alpha)) # Return entropy.

    def joint_renyi(self, x, y): # Matrix formulation of joint Renyi entropy.
        alpha = 1.01
        k_x = self.gram_matrix(x)
        k_y = self.gram_matrix(y)
        k = np.multiply(k_x, k_y)
        k = k / np.float64(np.trace(k))

        l, v = LA.eig(k)
        lambda_k = np.abs(l)

        return (1/(1-alpha))*np.log2(np.sum(lambda_k**alpha))

    def mutual_information(self, x, y):  # Matrix formulation of mutual Renyi entropy.

        return self.renyi(x)+self.renyi(y)-self.joint_renyi(x, y)

