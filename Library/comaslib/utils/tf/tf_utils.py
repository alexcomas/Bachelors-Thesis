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

from sympy import true
import tensorflow as tf
import tensorflow_addons as tfa
import sys
from ...model.tf.GNN import GNN
from ...data.generator import traces_to_graph, graph_to_dict, get_chosen_connection_features
import os


def _get_compiled_model(params):
    model = GNN(params)
    decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(float(params['HYPERPARAMETERS']['learning_rate']),
                                                                int(params['HYPERPARAMETERS']['decay_steps']),
                                                                float(params['HYPERPARAMETERS']['decay_rate']),
                                                                staircase=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=decayed_lr)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.SpecificityAtSensitivity(0.1),
               tf.keras.metrics.Recall(top_k=1,class_id=0, name='rec_0'), tf.keras.metrics.Precision(top_k=1, class_id=0, name='pre_0'),
               tf.keras.metrics.Recall(top_k=1,class_id=1, name='rec_1'), tf.keras.metrics.Precision(top_k=1,class_id=1, name='pre_1'),
               tf.keras.metrics.Recall(top_k=1,class_id=2, name='rec_2'), tf.keras.metrics.Precision(top_k=1,class_id=2, name='pre_2'),
               tf.keras.metrics.Recall(top_k=1,class_id=3, name='rec_3'), tf.keras.metrics.Precision(top_k=1,class_id=3, name='pre_3'),
               tf.keras.metrics.Recall(top_k=1,class_id=4, name='rec_4'), tf.keras.metrics.Precision(top_k=1,class_id=4, name='pre_4'),
               tf.keras.metrics.Recall(top_k=1,class_id=5, name='rec_5'), tf.keras.metrics.Precision(top_k=1,class_id=5, name='pre_5'),
               tf.keras.metrics.Recall(top_k=1,class_id=6, name='rec_6'), tf.keras.metrics.Precision(top_k=1,class_id=6, name='pre_6'),
               tf.keras.metrics.Recall(top_k=1,class_id=7, name='rec_7'), tf.keras.metrics.Precision(top_k=1,class_id=7, name='pre_7'),
               tf.keras.metrics.Recall(top_k=1,class_id=8, name='rec_8'), tf.keras.metrics.Precision(top_k=1,class_id=8, name='pre_8'),
               tf.keras.metrics.Recall(top_k=1,class_id=9, name='rec_9'), tf.keras.metrics.Precision(top_k=1,class_id=9, name='pre_9'),
               tf.keras.metrics.Recall(top_k=1,class_id=10, name='rec_10'), tf.keras.metrics.Precision(top_k=1, class_id=10, name='pre_10'),
               tf.keras.metrics.Recall(top_k=1,class_id=11, name="rec_11"), tf.keras.metrics.Precision(top_k=1, class_id=11, name="pre_11"),
               tf.keras.metrics.Recall(top_k=1,class_id=12, name="rec_12"), tf.keras.metrics.Precision(top_k=1, class_id=12, name='pre_12'),
               tf.keras.metrics.Recall(top_k=1,class_id=13, name='rec_13'), tf.keras.metrics.Precision(top_k=1, class_id=13, name='pre_13'),
               tf.keras.metrics.Recall(top_k=1,class_id=14, name='rec_14'), tf.keras.metrics.Precision(top_k=1, class_id=14, name='pre_14'),
               tfa.metrics.F1Score(15,average='macro',name='macro_F1'),tfa.metrics.F1Score(15,average='weighted',name='weighted_F1')]#, tfma.metrics.MultiClassConfusionMatrixPlot(name='multi_class_confusion_matrix_plot'),],

    model.compile(loss=loss_object,
                  optimizer=optimizer,
                  metrics= metrics,
                  run_eagerly=False)
    return model

import glob
from tensorflow.keras.models import load_model
def make_or_restore_model(params, forceRestore = None):
    # Either restore the latest model, or create a fresh one
    directory = 'DIRECTORIES_' + params['RUN_CONFIG']['dataset']
    checkpoint_dir = os.path.abspath(params[directory]['logs'] + '/ckpt')
    # if there is no checkpoint available.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoints = glob.glob(checkpoint_dir + "/weights*")
    
    startingEpoch = 0
    if checkpoints:
        if forceRestore == None:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
        else:
            latest_checkpoint = os.path.abspath(f'{checkpoint_dir}/{forceRestore}')
        print("Restoring from", latest_checkpoint)
        startingEpoch = int(latest_checkpoint.split('.')[1].split('-')[0])
        # gnn = load_model(latest_checkpoint, compile=True)
        gnn = _get_compiled_model(params)
        gnn.load_weights(latest_checkpoint)
        
    else:
        print("Creating a new model")
        gnn = _get_compiled_model(params)
    
    gnn.built = True
    return (gnn, startingEpoch)

import csv
from random import random
def generator(path, window, for_framework='tensorflow', data_treatment='none', dataset = None):
    if not isinstance(dataset, str):
        dataset = dataset.decode("utf-8")
    if dataset == '':
        dataset = None
    n_graphs = 0
    total_counter = 0
    path = path.decode('utf-8')
    files = glob.glob(path + '/*.csv')
    for file in files:
        print(f"\nOpening file in generator: {file}")
        print("Chosen features: ", ', '.join(get_chosen_connection_features()))
        print("Data treatment: ", data_treatment)
        with open(file, encoding="utf8", errors='ignore') as csvfile:
            data = csv.reader(csvfile, delimiter=',', quotechar='|')

            current_time_traces = []
            counter = 0
            for row in data:
                if len(row) > 1:
                    current_time_traces.append(row)
                    counter += 1
                    # remains to fix this criterion (for now we set the windows to be 200 connections big)
                    if counter >= window:
                        G = traces_to_graph(current_time_traces)
                        features, label = graph_to_dict(G, for_framework=for_framework, data_treatment=data_treatment, dataset=dataset)
                        n_graphs += 1
                        # We do not need to do the undersampling here, since it was done during the preprocessing
                        yield (features, label)
                        total_counter += counter
                        counter = 0
                        current_time_traces = []

def input_fn(data_path, window = 200, data_treatment='none',  validation=False, dataset = ''):
    ds = tf.data.Dataset.from_generator(generator,
                                        args=[data_path, window, 'tensorflow', data_treatment, dataset],
                                        output_types=(
                                            {'feature_connection':tf.float32,
                                             'n_i': tf.int64,
                                             'n_c': tf.int64,
                                             'src_ip_to_connection': tf.int64,
                                             'dst_ip_to_connection': tf.int64,
                                            'src_connection_to_ip': tf.int64,
                                            'dst_connection_to_ip': tf.int64}, tf.float32),
                                        output_shapes=(
                                            {
                                            'feature_connection':tf.TensorShape(None),
                                            'n_i': tf.TensorShape([]),
                                            'n_c': tf.TensorShape([]),
                                            'src_ip_to_connection': tf.TensorShape(None),
                                            'dst_ip_to_connection': tf.TensorShape(None),
                                            'src_connection_to_ip': tf.TensorShape(None),
                                            'dst_connection_to_ip': tf.TensorShape(None)}, tf.TensorShape(None))
                                        )

    #ds = ds.map(lambda x, y: standardization_function(x, y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    if not validation:
        ds = ds.repeat()
    
    return ds

def niceResults(result, dataset):
    dictKeys = ['loss', 'categorical_accuracy', 'specificity_at_sensitivity', 'rec_0', 'pre_0', 'rec_1', 'pre_1', 'rec_2', 'pre_2', 'rec_3', 'pre_3', 'rec_4', 'pre_4', 'rec_5', 'pre_5', 'rec_6', 'pre_6', 'rec_7', 'pre_7', 'rec_8', 'pre_8', 'rec_9', 'pre_9', 'rec_10', 'pre_10', 'rec_11', 'pre_11', 'rec_12', 'pre_12', 'rec_13', 'pre_13', 'rec_14', 'pre_14', 'macro_F1', 'weighted_F1']
    res = dict(zip(dictKeys, result))
    if dataset == "IDS2017":
        attack_names = ['SSH-Patator', 'DoS GoldenEye', 'PortScan', 'DoS Slowhttptest', 'Web Attack  Brute Force', 'Bot', 'Web Attack  Sql Injection', 
                        'Web Attack  XSS', 'Infiltration', 'DDoS', 'DoS slowloris', 'Heartbleed', 'FTP-Patator', 'DoS Hulk','BENIGN']
    elif dataset == "IDS2018":
        attack_names = ['FTP-BruteForce', 'SSH-BruteForce', 'DoS-GoldenEye', 'DoS-Slowloris', 'DoS-SlowHTTPTest', 'DoS-Hulk', 'DDoS attacks-LOIC-HTTP', 
                        'DDoS-LOIC-UDP', 'DDOS-HOIC', 'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection', 'Infiltration', 'Bot', 'BENIGN']
    indices = range(len(attack_names))
    zip_iterator = zip(attack_names,indices)
    attacks_dict = dict(zip_iterator)
    metrics = dict()
    for key, value in res.items():
        if key.startswith('pre'):
            i = int(key.split('_')[1])
            metrics['Pre_' + attack_names[i]] = value
        elif key.startswith('rec'):
            i = int(key.split('_')[1])
            metrics['Rec_' + attack_names[i]] = value
        else:
            metrics[key] = value
    return metrics