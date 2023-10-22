#! /usr/bin/env python

import phero_turtlebot_turtlebot3_repellent_ppo
import numpy as np
import os
import sys
import multiprocessing
# Keras and TensorFlow imports are removed
import torch
# ... (you might need other PyTorch related imports such as torch.nn, torch.optim, etc.)
import random
from collections import deque
from utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from distributions import make_pdtype
import os.path as osp
import joblib
import tensorboard_logging
import timeit
import csv
import math
import time
import matplotlib.pyplot as plt
import scipy.io as sio

import logger

class AbstractEnvRunner(object):
    '''
    Basic class import env, model and etc (used in Runner class)
    '''
    def __init__(self, env, model, nsteps):
        self.env = env
        self.model = model
        nenv = 1
        self.obs = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        raise NotImplementedError
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class PheroTurtlebotPolicy(nn.Module):
    def __init__(self, ob_space, ac_space, nbatch, nsteps, reuse=False, deterministic=False):
        super(PheroTurtlebotPolicy, self).__init__()
        
        # Assuming make_pdtype is a function that provides information about the action space
        # This may need modification based on the actual function's functionality
        self.pdtype = make_pdtype(ac_space)

        self.fc1 = nn.Linear(13, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)

        # For value function
        self.value_head = nn.Linear(128, 1)

        # For action distribution
        # TODO: Modify this according to the actual action space. The below is just a placeholder
        self.action_head = nn.Linear(128, ac_space.shape[0])

    def forward(self, phero):
        x = F.relu(self.fc1(phero))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # You might need to adjust these based on the action space
        action_probs = F.softmax(self.action_head(x), dim=-1)
        value = self.value_head(x)
        
        return action_probs, value

    def step(self, ob):
        phero = torch.tensor(ob, dtype=torch.float32)
        action_probs, value = self.forward(phero)
        
        # Sample action
        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = torch.multinomial(action_probs, 1)

        # Calculate log probability
        neglogp = -torch.log(action_probs[range(len(action_probs)), action])
        
        return action, value, None, neglogp

    def value(self, ob):
        phero = torch.tensor(ob, dtype=torch.float32)
        _, value = self.forward(phero)
        return value
