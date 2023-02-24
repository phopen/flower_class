import numpy as np
import os
import utils
import time
import cv2

from CNNModel import CNN
from torch import optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pdb

# train / test settings
trainSet = 1
testSet = 3
epochs = 2000

dataSet = 'Flowers'

# Load data
path = os.path.join(os.getcwd(), 'data/flowers')
data, labels = utils.load_images(path)
print('+++ Loading dataset: {} ({} images, {} labels)'.format(dataSet, len(data), len(labels)))

# Organize into numImages x numChannels x width x height
# x = data['x'].transpose([2, 0, 1])
# x = np.reshape(x, [x.shape[0], 1, x.shape[1], x.shape[2]])
# y = data['y']
#
# # Convert data into torch tensors
# x = torch.tensor(x).float()
# y = torch.tensor(y).long() # Labels are categorical
#
# # Define model
# model = CNN()
#
# # Define loss function and optimizer
# loss = nn.CrossEntropyLoss()
# opt = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # Start training
# xTrain = x[data['set'] == trainSet, :, :, :]
# yTrain = y[data['set'] == trainSet]
#
# # Loop over training data in some batches
# train_ds = TensorDataset(xTrain, yTrain)
# train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
#
# for epoch in range(epochs):
#     for xb, yb in train_dl:
#
#         model.train()
#         pred = model(xb)
#
#         it_loss = loss(pred, yb)
#
#         it_loss.backward()
#         opt.step()
#         opt.zero_grad()
#
# # Test model
# xTest = x[data['set'] == testSet, :, :, :]
# yTest = y[data['set'] == testSet]
#
# model.eval() # Set this to evaluation mode
#
# # Loop over xTest and compute labels (implement this)
# yPred = model.forward(xTest)
# yPred_n = yPred.detach().numpy()
# yPred_n = yPred_n - np.amax(yPred_n, axis=1, keepdims=True)
#
# q = np.exp(yPred_n)
# d = np.repeat(np.sum(q, axis=1)[:, np.newaxis], q.shape[1], axis=1)
# yPred_p = q / d
# yPred_v = np.argmax(yPred_p, axis=1)
#
# # Map it back to numpy to use our functions
# yTest = yTest.numpy()
# (acc, conf) = utils.evaluateLabels(yTest, yPred_v, False)
# print('Accuracy [testSet={}] {:.2f} %\n'.format(testSet, acc*100))
# accuracy[i] = acc
# #
# # Print the results in a table
# print('+++ Accuracy Table [trainSet={}, testSet={}]'.format(trainSet, testSet))
# print('--------------------------------------------------')
# print('dataset\t\t\t', end="")
# print('{}\t'.format('cnn'), end="")
# print()
# print('--------------------------------------------------')
# for i in range(len(dataTypes)):
#     print('{}\t'.format(dataTypes[i]), end="")
#     print('{:.2f}\t'.format(accuracy[i]*100))

