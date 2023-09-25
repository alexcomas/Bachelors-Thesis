import torch
from torch.utils.data import DataLoader
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import random, shuffle

class dgl_DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.n = 0
        self.iterator = iter(self.dataset)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            # print("Reset iterator.")
            self.iterator = iter(self.dataset)
            return next(self.iterator)

    def __len__(self):
        return len(self.dataset)