import torch
import torch.nn as nn
from torch.nn.functional import relu
###
### YOUR CODE HERE
###

class Basic_cnn(nn.Module):
    def __init__(self):
        super(Basic_cnn, self).__init__()
        # Convolutional layer: 16 filters with size 5x5
        self.conv1 = nn.Conv2d(3, 16, 5) # in this convolutional layer, w = 32 (cifar-10 image size), wf = 5, stride = 1, pad = 0.
                                         # By default, convolution is "valid" so, the output size is (32 - 5 + 1) / 1 = 28
        
        # Max pooling layer with kernel (filter) size 2x2
        self.pool = nn.MaxPool2d(2, 2)   # in this max pooling layer, w = 28, wf = 2, stride = 2
                                         # So, the size of the max pool output is (28 - 2) / 2 + 1 = 14
        
        # Fully connected layer with size 16 x 14 x 14 (number of filters x width of filter x height of filter)
        self.fc1 = nn.Linear(16 * 14 * 14, 10)

    

    def forward(self, x):
        x = self.pool(relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class Better_cnn(nn.Module):
    def __init__(self, numfilter1, numfilter2):
        super(Better_cnn, self).__init__()
        
        # Convolutional layer: numfilter1 filters with size 7x7
        self.conv1 = nn.Conv2d(3, numfilter1, 7) # in this convolutional layer, w = 32 (cifar-10 image size), wf = 5, stride = 1, pad = 0.
                                         # By default, convolution is "valid" so, the output size is (32 - 7 + 1) / 1 = 26
        
        # Max pooling layer with kernel (filter) size 2x2
        self.pool = nn.MaxPool2d(4, 1)   # in this max pooling layer, w = 26, wf = 4, stride = 1
                                         # So, the size of the max pool output is (26 - 4) / 1 + 1 = 23

        # Convolutional layer: numfilter2 filters with size 5x5
        self.conv2 = nn.Conv2d(numfilter1, numfilter2, 2) # in this convolutional layer, w = 23 (cifar-10 image size), wf = 5, stride = 1, pad = 0.
                                         # By default, convolution is "valid" so, the output size is (23 - 2 + 1) / 1 = 22
        
        # Max pooling layer with kernel (filter) size 2x2
        self.pool2 = nn.MaxPool2d(2, 2)   # in this max pooling layer, w = 22, wf = 4, stride = 1
                                         # So, the size of the max pool output is (22 - 2) / 2 + 1 = 11

        
        # Fully connected layer with size 16 x 14 x 14 (number of filters x width of filter x height of filter)
        self.fc1 = nn.Linear(numfilter2*11*11, 10)

        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        
        x = self.pool(relu(self.conv1(x)))
        x = self.pool2(relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        return x


        