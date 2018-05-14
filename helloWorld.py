# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:14:12 2018

@author: Owner
"""

import torch
from torchvision import transforms, datasets

##- load data
data_transform = transforms.Compose([
        transforms.RandomSizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
#train         
uo2_dataset = datasets.ImageFolder(root='sample/train',
                                           transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(uo2_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=0)
#eval
uo2_evalset = datasets.ImageFolder(root='sample/eval',
                                           transform=data_transform)
evalset_loader = torch.utils.data.DataLoader(uo2_dataset,
                                             batch_size=10, shuffle=True,
                                             num_workers=0)

classes = ('AmUO3', 'U3O8', 'UO2f8', 'UO2fAm',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

##- show image
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

##-the structure of the convnet
import torch.nn as nn
import torch.nn.functional as F
import pdb


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataset_loader, start=0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        net.conv1.register_forward_hook(printnorm)
        net.conv2.register_forward_hook(printnorm)
        net.pool.register_forward_hook(printnorm)
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

class_correct = list(0. for i in range(4))
class_total = list(0. for i in range(4))
with torch.no_grad():
    for data in evalset_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(4):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


