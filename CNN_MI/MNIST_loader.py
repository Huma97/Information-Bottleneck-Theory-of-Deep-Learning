import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms


transform = transforms.ToTensor()


trainset = torchvision.datasets.MNIST(root='./data', train=True,         # Load Cifar10 training dataset.
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,        # Dataloader
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,        # Load Cifar10 test dataset.
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)     # Dataloader

import numpy as np


from LeNet5 import LeNet5           # Import network from LeNet script
#model = VGG16(10, nn.ReLU()).cuda()# Initialize model with 10 classes and relu activation function.
                                    # Note that if you run this scipt on a cpu you must remove the ".cuda()"
                                    # from this line.
model = LeNet5(10, nn.Sigmoid()).cuda()


#out, layers = model(images)

# Nevermind this part, it is just for prototyping and experimenting.
# %%
#
#mi_mat = np.zeros((1, 17, 17))
#j_mat = np.zeros((1, 17, 17))
#
#for i, layer in enumerate(layers, 0):
#
#    mi_mat[0, 0, 0] = model.mutual_information(images, images)
#    mi_mat[0, 0, i+1] = model.mutual_information(images, layer.cpu())
#    
#    j_mat[0, 0, 0] = model.joint_renyi(images, images)
#    j_mat[0, 0, i+1] = model.joint_renyi(images, layer.cpu())
#
#for i in range(16):
#    for j in range(i, 16):
#        mi_mat[0, i+1, j+1] = model.mutual_information(layers[i].cpu(), layers[j].cpu())
#        j_mat[0, i+1, j+1] = model.joint_renyi(layers[i].cpu(), layers[j].cpu()) 

# %%

mi_mat = np.zeros((1, 6, 6))   # Mutual information matrix (just zeros for now).
j_mat = np.zeros((1, 6, 6))    # Joint information matrix (just zeros for now).

y_mi = np.zeros((1, 6))

dataiter = iter(testloader)      # Data iterator, for running through the data.
inputs, labels = dataiter.next()
outputs, layers = model(inputs.cuda()) # Get the ouput of the network and each layer before any training.

for i, layer in enumerate(layers, 0): # Loop through the output of each layer.

    mi_mat[0, 0, 0] = model.mutual_information(inputs, inputs)       # MI between X and X
    mi_mat[0, 0, i+1] = model.mutual_information(inputs, layer.cpu())# Mi between X and all other feature maps.
    
    y_mi[0,0] = model.mutual_information(outputs, inputs)
    y_mi[0, i+1] = model.mutual_information(outputs, layer.cpu())

    j_mat[0, 0, 0] = model.joint_renyi(inputs, inputs)               # Joint between X and X
    j_mat[0, 0, i+1] = model.joint_renyi(inputs, layer.cpu())        # Joint between X and all other feature maps.

for i in range(5): # Loop through all layers. Maybe these two loops can be combined.
    for j in range(i, 5):
        mi_mat[0, i+1,j+1] = model.mutual_information(layers[i].cpu(), layers[j].cpu()) # MI between all feature maps.
        j_mat[0, i+1, j+1] = model.joint_renyi(layers[i].cpu(), layers[j].cpu()) # Joint between all feature maps.

criterion = nn.CrossEntropyLoss()               # Cross-entropy loss
optimizer = torch.optim.Adam(model.parameters())# ADAM optimizer.

cost = [] # List for keeping track of training error
test = [] # list for keeping track of test error.

num_epochs = 100

for epoch in range(num_epochs):     # Run for 10 epochs
    print('Epoch: ', epoch) 
    dataiter = iter(testloader)
    inputs, labels = dataiter.next()
    outputs, layers = model(inputs.cuda())

    mi_mat_temp = np.zeros((1, 6, 6)) # Temporary matrices for stroing MI and joint.
    j_mat_temp = np.zeros((1, 6, 6))
    
    y_mi_temp = np.zeros((1,6))
    
    cost_temp = []
    
    for i, layer in enumerate(layers, 0):
        
        mi_mat_temp[0, 0, 0] = model.mutual_information(inputs, inputs)    
        mi_mat_temp[0, 0, i+1] = model.mutual_information(inputs, layer.cpu())
        
        y_mi_temp[0,0] = model.mutual_information(outputs, inputs)
        y_mi_temp[0,i+1] = model.mutual_information(outputs, inputs)
        
        j_mat_temp[0, 0, 0] = model.joint_renyi(inputs, inputs)
        j_mat_temp[0, 0, i+1] = model.joint_renyi(inputs, layer.cpu())
    
    for i in range(5):
        for j in range(i,5):
            mi_mat_temp[0, i+1,j+1] = model.mutual_information(layers[i].cpu(), layers[j].cpu()) 
            j_mat_temp[0, i+1, j+1] = model.joint_renyi(layers[i].cpu(), layers[j].cpu()) 
            
    if epoch != 0: # We are not adding the temp matrices to the actual matrix for the first epoch, 
        mi_mat = np.concatenate((mi_mat, mi_mat_temp)) # since we already calcualted this outside the loop.
        j_mat = np.concatenate((j_mat, j_mat_temp))
        
        y_mi = np.concatenate((y_mi, y_mi_temp))

    for i, data in enumerate(trainloader, 0): # Run through entire training set.

        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, layers = model(inputs.cuda())  # If you wnat to run on cpu
        loss = criterion(outputs, labels.cuda())# remove ".cuda()"
        loss.backward()
        optimizer.step()
        cost_temp.append(loss.cpu().data.numpy()) # Append training cost
    cost.append(np.mean(cost_temp))               # Take mean cost of these batches
    print(cost[-1])              


print('Finished Training')
np.savez_compressed('X_MI_MNIST', a=mi_mat, b=j_mat, c=cost) # Save the final MI matrix, Joint matrix and cost as
                                                              # "MI_MNIST.npz"
    
np.savez_compressed('Y_MI_MNIST', a=y_mi)


