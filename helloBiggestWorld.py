# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:14:12 2018

@author: Joshua Ong
@summary simple cnn for classification between 4 classes
"""

import torch
from torchvision import transforms, datasets
import pdb

##- load data ##- load data ##- load data ##- load data ##- load data ##- load data
data_transform = transforms.Compose([
        transforms.RandomSizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])         
uo2_dataset = datasets.ImageFolder(root='sample/binaryTest/train',
                                           transform=data_transform)

##- weighted sample for class imbalance -- https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight                                

weights = make_weights_for_balanced_classes(uo2_dataset.imgs, len(uo2_dataset.classes)) 
weights = torch.DoubleTensor(weights) #cast me to tensor    
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) #sampler
##- weighted sample for class imbalance -- https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3

dataset_loader = torch.utils.data.DataLoader(uo2_dataset,
                                             batch_size=32,
                                             num_workers=0, sampler = sampler) #had to remove shuffle
uo2_evalset = datasets.ImageFolder(root='sample/binaryTest/eval',
                                           transform=data_transform)
evalset_loader = torch.utils.data.DataLoader(uo2_evalset,
                                             batch_size=10, shuffle=True,
                                             num_workers=0)
classes = ('AmUO3', 'U3O8', 'UO2f8', 'UO2fAm',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
#pdb.set_trace()

"""
##- show image ##- show image ##- show image ##- show image ##- show image 
import matplotlib.pyplot as plt
import numpy as np


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(dataset_loader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images.clamp(0,1)))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
"""

##-the structure of the convnet
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 9, 3)
        self.conv3 = nn.Conv2d(9, 12, 3)
        self.conv4 = nn.Conv2d(12, 15, 3)
        self.conv5 = nn.Conv2d(15, 18, 3)
        self.conv6 = nn.Conv2d(18, 21, 3)
        self.conv7 = nn.Conv2d(21, 24, 3)
        self.conv8 = nn.Conv2d(24, 27, 3)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(27 * 27 * 27, 10000)
        self.fc2 = nn.Linear(10000, 100)
        self.sm1 = nn.Softmax()
        self.fc3 = nn.Linear(100, 4)

    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = self.pool(x)
        #126x126x9
        x = (F.relu(self.conv3(x)))
        x = (F.relu(self.conv4(x)))
        x = (F.relu(self.conv5(x)))
        x = self.pool(x)
        #
        x = (F.relu(self.conv6(x)))
        x = (F.relu(self.conv7(x)))
        x = (F.relu(self.conv8(x)))
        x = self.pool(x)
        
        #voodoo?
        x = x.view(-1, 27 * 27 * 27)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.sm1()
        x = self.fc3(x)
        return x
"""
def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())
"""    
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x
    

net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(4):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataset_loader, start=0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        #net.conv1.register_forward_hook(printnorm)
        #net.conv2.register_forward_hook(printnorm)
        #net.pool.register_forward_hook(printnorm)
        outputs = net(inputs)
        #net.conv1.register_forward_hook(printnorm(net,inputs,outputs))
        #pdb.set_trace()
        #model:forward(input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in evalset_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in evalset_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(2):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(4):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


