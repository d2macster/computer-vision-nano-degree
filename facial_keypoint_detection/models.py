## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net3(nn.Module):

    def __init__(self):
        super(Net3, self).__init__()
      
        
        self.conv1 = nn.Conv2d(1, 32, 3) 
        self.conv1_bn = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv4_bn = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(12*12*256, 4000)
        self.fc1_bn = nn.BatchNorm1d(4000)
        self.fc2 = nn.Linear(4000, 1000)
        self.fc2_bn = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000,136)
        
        self.dropout = nn.Dropout(0.2)
      

        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        
        x = x.view(x.size(0),-1)
        
        x = self.dropout(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.dropout(F.relu(self.fc2_bn(self.fc2(x))))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
      
        
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv4_bn = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(12*12*256, 1000)
        self.fc1_bn = nn.BatchNorm1d(1000)
        
        self.fc2 = nn.Linear(1000, 1000)
        self.fc2_bn = nn.BatchNorm1d(1000)
        
        self.fc3 = nn.Linear(1000,136)
        
        self.dropout = nn.Dropout(0.2)
      
    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        
        x = x.view(x.size(0),-1)
        
        x = self.dropout(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.dropout(F.relu(self.fc2_bn(self.fc2(x))))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # we will expect input images 224x224x1
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 512, 1)
        
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.6)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(6*6*512, 1000)
        self.fc2 = nn.Linear(1000, 136)

       
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # x.shape = 224x224x1
        x = self.drop1(self.pool(F.relu(self.conv1(x))))
        # print("1st layer shape: ", x.shape)
            
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        # print("2nd layer shape: ", x.shape)

        x = self.drop3(self.pool(F.relu(self.conv3(x))))
        # print("3rd layer shape: ", x.shape)

        x = self.drop4(self.pool(F.relu(self.conv4(x))))
        # print("4th layer shape: ", x.shape)
        
        x = self.drop5(self.pool(F.relu(self.conv5(x))))
        # print("5th layer shape: ", x.shape)

        x = x.view(x.size(0), -1)
        # print("flat layer shape: ", x.shape)
        
        x = self.drop6(F.relu(self.fc1(x)))
        # print("FC1 layer shape: ", x.shape)
        
        x = self.fc2(x)
        # 136
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
