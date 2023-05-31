import pkg_resources
pkg_resources.require("torch==1.8.1")

import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
import math
import gym
import numpy as np
import torch
import torch.nn as nn

import onnx
from onnx_tf.backend import prepare

import tensorflow as tf

from torch.autograd import Variable

from carla_rl.carla_env import CarlaEnv
from carla_rl.env_wrappers import PreprocessCARLAObs
from carla_rl.agent import MixedDQNAgent

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
    self.conv1 = nn.Conv2d(3, 16, 5, stride=1, padding=2)
    self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
    self.conv3 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
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
    print(torch.__version__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    """
    ########## Conversion of our ConvNet
    # Create ONNX model from our PyTorch model
    model = ConvNet(192 // 8)
    model.load_state_dict(torch.load("../Code/image_processing/lane_classifier.model").state_dict())
    model.eval()
    print(model)

    dummy_input = Variable(torch.randn(1, 3, 192, 192))
    output = model(dummy_input)

    torch.onnx.export(model, dummy_input, "lane_classifier.onnx", export_params=True,
                      opset_version=10, do_constant_folding=True,
                      input_names = ['input'], output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    
    # Create Tensorflow model from the ONNX model
    onnx_model = onnx.load("lane_classifier.onnx")
    tf_model = prepare(onnx_model)
    tf_model.export_graph("lane_classifier.h5")
    
    model = tf.saved_model.load("lane_classifier.h5")
    input_tensor = tf.random.uniform([1, 3, 192, 192])
    model(**{'input': input_tensor})
    print(model)

    # Create tf-lite model from our Tensorflow model
    converter = tf.lite.TFLiteConverter.from_saved_model("lane_classifier.h5")
    tflite_model = converter.convert()
    with open("lane_classifier.tflite", "wb") as file:
        file.write(tflite_model)
    """

    """
    ########## Conversion of our RL model
    Create ONNX model from our PyTorch model
    env = CarlaEnv(town=None, fps=20, im_width=1280, im_height=720, repeat_action=1, start_transform_type="random",
                   sensors="rgb", action_type="mixed", enable_preview=False, steps_per_episode=500, playing=False,
                   timeout=60)
    env = PreprocessCARLAObs(env)

    agent = MixedDQNAgent(env=env, epsilon=0.5)
    state, _ = env.reset()
    state = np.expand_dims(state, axis=0)
    print("STATE?", state)
    qvalues = agent.get_qvalue_params(np.concatenate([state, state]))
    print("QVALUES", qvalues)
    print("SAMPLED ACTIONS", agent.sample_actions(qvalues))
    agent .load_state_dict(torch.load("../Code/saved-models/temporary_saved_agent_470000.pth").state_dict())
    agent.eval()
    print(agent)

    dummy_input = Variable(torch.randn(1, 14))
    output = agent(dummy_input)

    torch.onnx.export(agent, dummy_input, "agent.onnx", export_params=True,
                      opset_version=10, do_constant_folding=True,
                      input_names = ['input'], output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    
    # Create Tensorflow model from the ONNX model
    onnx_model = onnx.load("agent.onnx")
    tf_model = prepare(onnx_model)
    tf_model.export_graph("agent.h5")
    
    model = tf.saved_model.load("agent.h5")
    print(model)

    # Create tf-lite model from our Tensorflow model
    converter = tf.lite.TFLiteConverter.from_saved_model("agent.h5")
    tflite_model = converter.convert()
    with open("agent.tflite", "wb") as file:
        file.write(tflite_model)
    """
