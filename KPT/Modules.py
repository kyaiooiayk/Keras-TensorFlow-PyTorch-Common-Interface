# PyTorch modules
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torchinfo import summary
# Custom-made function
from .PyTorchTools import EarlyStopping

# Keras modules
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.layers import Dense

# Others
import time
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import functools
import logging

class PyTorchModules():
    def __init__(self):
        super(PyTorchModules, self).__init__()
        self.torch = torch
        self.nn = nn
        self.Variable = Variable
        self.F = F
        self.Data = Data
        self.EarlyStopping = EarlyStopping
        self.summary = summary


class KerasModules():
    def __init__(self):
        super(KerasModules, self).__init__()
        self.keras = keras
        self.layers = layers
        self.callbacks = callbacks
        self.Dense = Dense


class Others():
    def __init__(self):
        super(Others, self).__init__()
        self.plt = plt
        self.np = np
        self.seed = seed
        self.train_test_split = train_test_split
        self.r2_score = r2_score
        self.mean_squared_error = mean_squared_error
        self.mean_absolute_error = mean_absolute_error
        self.rcParams = rcParams
        self.sys = sys
        self.random = random
        self.time = time
        self.functools = functools
        self.logging = logging


"""
Alternatively, if you want to mantain 
the dot notation you can use the following.
If so, remove the super() from each closs.
"""
"""
class Modules_():
    def __init__(self):
        self.PT = PyTorchModules()
        self.K = KerasModules()
        self.O = Others()
"""


class Modules(PyTorchModules, KerasModules, Others):
    """Modules.

    Multi-inheritance of three base classes. Module is
    the children class which has access via self to all
    the base class instance attributes.

    This simplifies the import on each script. Instead
    of a having the same boiler plate element in all scripts,
    this is done once only and then called by a single line
    as:
    from KPT.Modules import Modules
    M = Modules() 
    """

    def __init__(self):
        super(Modules, self).__init__()
