import torch
from torch.utils.data import IterableDataset
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

class IDS2017_Dataset(IterableDataset):
    def __init__(self, path, window):
        super(IDS2017_Dataset).__init__()
        self.path = path
        self.window = window
        self.length = 0
        files = glob.glob(self.path + '/*.csv')
        
        for file in files:
            counter = 0
            with open(file, encoding="utf8", errors='ignore') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                counter +=  sum(1 for _ in reader)
            self.length += math.ceil(counter/self.window)

    def __iter__(self):
        # self.path = self.path.decode('utf-8')
        files = glob.glob(self.path + '/*.csv')

        for file in files:
            with open(file, encoding="utf8", errors='ignore') as csvfile:
                data = csv.reader(csvfile, delimiter=',', quotechar='|')
                
                current_time_traces = []
                counter = 0
                for row in data:
                    if len(row) > 1:
                        current_time_traces.append(row)
                        counter += 1
                        # remains to fix this criterion (for now we set the windows to be 200 connections big)
                        if counter >= 200:
                            G = generator.traces_to_graph(current_time_traces)
                            features, label =  generator.graph_to_dict(G)
                            # We do not need to do the undersampling here, since it was done during the preprocessing
                            yield (features, label)
                            counter = 0
                            current_time_traces = []

    def __len__(self):
        return self.length