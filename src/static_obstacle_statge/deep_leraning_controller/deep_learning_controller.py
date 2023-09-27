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