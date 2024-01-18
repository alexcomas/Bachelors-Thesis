import torch
from torch.utils.data import IterableDataset
import csv
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import glob
from torch.utils.data import DataLoader
from ..IDS_Dataset_base import IDS_Dataset_base
from .MyDataLoader import MyDataLoader

class IDS_Dataset(IterableDataset, IDS_Dataset_base):
    def __init__(self, dataset_path: str, dataset:str='IDS2017', dataset_labels:str=None, window:int=200, data_treatment:str='none', data_treatment_params_path:str=None) -> None:
        IDS_Dataset_base.__init__(self, dataset_path=dataset_path, dataset=dataset, dataset_labels= dataset_labels, 
                                    for_framework='pytorch', data_treatment=data_treatment, data_treatment_params_path=data_treatment_params_path)
        IterableDataset.__init__(self)
        self.__setLength()

    def __iter__(self):
        return self.generate()

    def __len__(self):
        return self.length
    
    def __setLength(self):
        files = glob.glob(self.dataset_path + '/*.csv')
        self.length = 0
        for file in files:
            counter = 0
            with open(file, encoding="utf8", errors='ignore') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                counter +=  sum(1 for _ in reader)
            self.length += math.ceil(counter/self.window)

    def getLoader(self, loader_params):
        return MyDataLoader(DataLoader(self, **loader_params))
