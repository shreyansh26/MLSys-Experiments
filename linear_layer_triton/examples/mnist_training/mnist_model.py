import time
from src.linear_layer_triton import LinearLayerTriton
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from src.mlp_models import MLP1, MLP2, MLP3, MLP4, MLP5
from src.patch_linear_layer import patch_linear_layer
from torch.fx import symbolic_trace
from misc.patch_model import patch_model

NUM_ITERS = 100
DEBUG = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        # self.fc1t = LinearLayerTriton(self.fc1.weight, self.fc1.bias, activation="relu")
        # self.fc2t = LinearLayerTriton(self.fc2.weight, self.fc2.bias)
        self.fc1t = LinearLayerTriton2(9216, 128, activation="relu")
        self.fc2t = LinearLayerTriton2(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1t(x)
        x = self.dropout2(x)
        x = self.fc2t(x)
        output = F.log_softmax(x, dim=1)
        return output
    
# model = Net().to('cuda')
# model = model.half()
# gm = symbolic_trace(model)
# gm_old = copy.deepcopy(gm)
# patch_linear_layer(gm, debug=DEBUG)

# print(gm_old.code)
# print("**"*100)
# print(gm.code)

def get_model(format="torch"):
    if format == "torch":
        model = Net().to('cuda')
        model = model.half()
        return model
    elif format == "triton":
        model = Net2().to('cuda')
        model = model.half()
        return model