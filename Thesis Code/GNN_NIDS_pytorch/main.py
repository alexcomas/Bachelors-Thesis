"""
   Copyright 2020 Universitat Politècnica de Catalunya
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import time
import numpy as np
import os
import configparser
from IDS2017_Dataset import IDS2017_Dataset
from MyDataLoader import MyDataLoader
from torch.utils.data import DataLoader
import GNN
import importlib
importlib.reload(GNN)
from GNN import GNN
from utils import datasetReport
import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('./config.ini')

# callbacks to save the model
path_logs = os.path.abspath(params['DIRECTORIES']['logs'])

train_ds =  IDS2017_Dataset(os.path.abspath(params["DIRECTORIES"]["train"]), int(params['HYPERPARAMETERS']['window']))
eval_ds = IDS2017_Dataset(os.path.abspath(params["DIRECTORIES"]["validation"]), int(params['HYPERPARAMETERS']['window']))

if not os.path.exists(params['DIRECTORIES']['logs'] + '/ckpt'):
    os.makedirs(params['DIRECTORIES']['logs'] + '/ckpt')

if params['RUN_CONFIG']['dataset_report'] == "True":
    print("Training dataset: ")
    datasetReport(train_ds, int(params['HYPERPARAMETERS']['classes']))
    print("Validation dataset: ")
    datasetReport(eval_ds, int(params['HYPERPARAMETERS']['classes']))

loader_params = {'batch_size': int(params['HYPERPARAMETERS']['batch_size'])}
training_generator = DataLoader(train_ds, **loader_params)
evaluating_generator = DataLoader(eval_ds, **loader_params)

model = GNN(params)

model.fit(MyDataLoader(training_generator), MyDataLoader(evaluating_generator)
            # ,steps_per_epoch= 1600
            # ,validation_steps = 200
        )
