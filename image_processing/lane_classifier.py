import os
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
import math
from PIL import Image


class CNNPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 3
        self.layers = nn.ModuleList()
        input_size = 224
        kernel_size = 5
        stride = 2
        padding = 1
        self.layers.append(nn.Conv2d(3, 64, kernel_size, stride=stride, padding=padding))
        num_c = 64
        cnn_output_size = self.calc_output_size(input_size, kernel_size, stride, padding)
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Conv2d(num_c, num_c * 2, kernel_size, stride=stride, padding=padding))
            num_c = num_c * 2
            cnn_output_size = self.calc_output_size(cnn_output_size, kernel_size, stride, padding)
        self.fc1 = nn.Linear(cnn_output_size * cnn_output_size * num_c, 128)
        self.fc2 = nn.Linear(128, 3)

    def calc_output_size(self, input_size, kernel_size, stride, padding):
        output_size = ((input_size - kernel_size + 2 * padding) / stride) + 1
        return int(math.floor(output_size))

    def forward(self, X):
        '''
        inputs:
            X: [batch_size, num_channels, input_height, input_width]
        outputs:
            temp: [batch_size, num_classes]
        '''
        temp = X
        for i in range(self.num_layers - 1):
            temp = self.layers[i](temp)
            temp = F.relu(temp)
        temp = self.layers[-1](temp)
        temp = torch.flatten(temp, start_dim=1)
        temp = self.fc1(F.relu(temp))
        temp = self.fc2(F.relu(temp))
        return temp


class ConvNet(nn.Module):
    def __init__(self, f_size):
        super(ConvNet, self).__init__()
        self.f_size = f_size
        self.conv1 = nn.Conv2d(3, 16, 5, stride=1, padding='same')
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding='same')
        self.conv3 = nn.Conv2d(32, 64, 5, stride=1, padding='same')
        self.fc1 = nn.Linear(f_size * f_size * 64, 64)
        self.fc2 = nn.Linear(64, 3)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.f_size * self.f_size * 64)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


if __name__ == "__main__":
    model = torch.load("lane_classifier.model").to(torch.device("cpu"))
    print(model)
    # img = torch.randn(1, 3, 192, 192, dtype=torch.float).to(torch.device("cpu"))
    # torch.onnx.export(model, img, "lane_classifier.onnx")