import torch
from torch.utils.data import DataLoader
import csv
import sys
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import glob
import configparser
import generator
from random import random, shuffle

class MyDataLoader:
    def __init__(self, loader):
        self.loader = loader
        self.n = 0
        self.iterator = iter(self.loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            # print("Reset iterator.")
            self.iterator = iter(self.loader)
            return next(self.iterator)

    def __len__(self):
        return len(self.loader)