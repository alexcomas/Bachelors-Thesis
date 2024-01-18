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
import numpy as np
import os
import configparser
import warnings
warnings.filterwarnings('ignore')
import Library.comaslib as comaslib
from Library.comaslib.utils.torch.torch_utils import datasetReport

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('./config.ini')

directory = 'DIRECTORIES_' + params['RUN_CONFIG']['dataset_data'] + '_LABELS_' + params['RUN_CONFIG']['dataset_labels']

def stringToBool(string):
    return string == 'True' or string == 'true'

# callbacks to save the model
path_logs = os.path.abspath(params[directory]['logs'])
(model, startingEpoch) = comaslib.model.torch.dict.GNN.make_or_restore_model(hyperparameters=params['HYPERPARAMETERS'], logs_dir=params[directory]['logs'],
                                                                             use_wandb=stringToBool(params['RUN_CONFIG']['wandb']),
                                                                             sweep=stringToBool(params['RUN_CONFIG']['sweep']),
                                                                             extended_metrics=stringToBool(params['RUN_CONFIG']['extended_metrics']),
                                                                             loadEpoch=None,
                                                                             loadBestEpoch=True,
                                                                             force_cpu=False)
train_ds =  comaslib.data.torch.IDS_Dataset(os.path.abspath(params[directory]["train"]),
                                            dataset=params['RUN_CONFIG']['dataset_data'],
                                            dataset_labels=params['RUN_CONFIG']['dataset_labels'],
                                            window = int(params['RUN_CONFIG']['window']),
                                            data_treatment=params['RUN_CONFIG']['data_treatment'],
                                            data_treatment_params_path=os.path.abspath(params[directory]["data_treatment"]))
eval_ds =   comaslib.data.torch.IDS_Dataset(os.path.abspath(params[directory]["validation"]), 
                                            dataset=params['RUN_CONFIG']['dataset_data'],
                                            dataset_labels=params['RUN_CONFIG']['dataset_labels'],
                                            window = int(params['RUN_CONFIG']['window']),
                                            data_treatment=params['RUN_CONFIG']['data_treatment'],
                                            data_treatment_params_path=os.path.abspath(params[directory]["data_treatment"]))

if not os.path.exists(path_logs + '/ckpt'):
    os.makedirs(path_logs + '/ckpt')

if params['RUN_CONFIG']['dataset_report'] == "True":
    print("Training dataset: ")
    datasetReport(train_ds, int(params['HYPERPARAMETERS']['classes']))
    print("Validation dataset: ")
    datasetReport(eval_ds, int(params['HYPERPARAMETERS']['classes']))

loader_params = {'batch_size': int(params['HYPERPARAMETERS']['batch_size'])}

model.fit(train_ds.getLoader(loader_params=loader_params), eval_ds.getLoader(loader_params=loader_params)
            ,steps_per_epoch= 200
            ,validation_steps = 50
        )
