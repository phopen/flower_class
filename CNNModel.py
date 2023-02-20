import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=8,
                      kernel_size=5,
                      stride=1,
                      padding=2
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # batch norm 1
        self.conv1_bn = nn.BatchNorm2d(8)

        # Convolutional layer 3
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16,
                      kernel_size=5,
                      stride=1,
                      padding=2
                      ),
            nn.ReLU())

        # Convolutional layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # batch norm 2
        self.conv3_bn = nn.BatchNorm2d(32)

        # Convolutional layer 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU())

        # Fully connected layer
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):

        # convolution layers
        x = self.conv1_bn(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3_bn(self.conv3(x))
        x = self.conv4(x)

        # flatten the output of conv2 to (batch_size, 32*7*7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
