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

import csv
import sys

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import glob
import configparser
import os

generalParams = configparser.ConfigParser()
generalParams.read('D:\TFG\datasets\IDS2018\\config.ini')

directory = 'DIRECTORIES_' + generalParams['RUN_CONFIG']['dataset']

normalization = configparser.ConfigParser()
normalization._interpolation = configparser.ExtendedInterpolation()
normalization.read(generalParams[directory]['data_treatment'] + 'normalization_parameters.ini')

special_normalization = configparser.ConfigParser()
special_normalization._interpolation = configparser.ExtendedInterpolation()
special_normalization.read(generalParams[directory]['data_treatment'] + 'special_normalization_parameters.ini')

robust_normalization = configparser.ConfigParser()
robust_normalization._interpolation = configparser.ExtendedInterpolation()
robust_normalization.read(generalParams[directory]['data_treatment'] + 'robust_normalization_parameters.ini')

standardization = configparser.ConfigParser()
standardization._interpolation = configparser.ExtendedInterpolation()
standardization.read(generalParams[directory]['data_treatment'] + 'standardization_parameters.ini')

# --------------------------------------
# IDS 2017

# MAP THAT TELLS US, GIVEN A FEATURE, ITS POSITION (IDS 2017)
features = ['Flow ID','Source IP','Source Port','Destination IP','Destination Port','Protocol','Timestamp','Flow Duration','Total Fwd Packets',
            'Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets','Fwd Packet Length Max','Fwd Packet Length Min',
            'Fwd Packet Length Mean','Fwd Packet Length Std','Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std',
            'Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std',
            'Fwd IAT Max','Fwd IAT Min','Bwd IAT Total','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags',
            'Fwd URG Flags','Bwd URG Flags','Fwd Header Length','Bwd Header Length','Fwd Packets/s','Bwd Packets/s','Min Packet Length',
            'Max Packet Length','Packet Length Mean','Packet Length Std','Packet Length Variance','FIN Flag Count','SYN Flag Count','RST Flag Count',
            'PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count','Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size',
            'Avg Bwd Segment Size','Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk','Fwd Avg Bulk Rate','Bwd Avg Bytes/Bulk','Bwd Avg Packets/Bulk',
            'Bwd Avg Bulk Rate','Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes','Init_Win_bytes_forward',
            'Init_Win_bytes_backward','act_data_pkt_fwd','min_seg_size_forward','Active Mean','Active Std','Active Max','Active Min','Idle Mean',
            'Idle Std','Idle Max','Idle Min','Label']
indices = range(len(features))
zip_iterator = zip(features,indices)
features_dict = dict(zip_iterator)


attack_names_IDS2017 = ['SSH-Patator', 'DoS GoldenEye', 'PortScan', 'DoS Slowhttptest', 'Web Attack  Brute Force', 'Bot', 'Web Attack  Sql Injection', 
                        'Web Attack  XSS', 'Infiltration', 'DDoS', 'DoS slowloris', 'Heartbleed', 'FTP-Patator', 'DoS Hulk','BENIGN']
attack_names_IDS2018 = ['FTP-BruteForce', 'SSH-BruteForce', 'DoS-GoldenEye', 'DoS-Slowloris', 'DoS-SlowHTTPTest', 'DoS-Hulk', 'DDoS attacks-LOIC-HTTP', 
                        'DDoS-LOIC-UDP', 'DDOS-HOIC', 'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection', 'Infiltration', 'Bot', 'BENIGN']
# ATTACKS IDS 2017 i IDS 2018
if generalParams['RUN_CONFIG']['dataset'].endswith('LABELS_IDS2017'):
    attack_names = attack_names_IDS2017
elif generalParams['RUN_CONFIG']['dataset'].endswith('LABELS_IDS2018'):
    attack_names = attack_names_IDS2018
indices = range(len(attack_names))
zip_iterator = zip(attack_names,indices)
attacks_dict = dict(zip_iterator)


chosen_connection_features = ['Source Port', 'Destination Port', 'Bwd Packet Length Min', 'Subflow Fwd Packets',
                   'Total Length of Fwd Packets', 'Fwd Packet Length Mean', 'Total Length of Bwd Packets',
                   'Fwd Packet Length Std', 'Fwd IAT Min', 'Flow IAT Min', 'Flow IAT Mean', 'Bwd Packet Length Std',
                   'Subflow Fwd Bytes', 'Flow Duration', 'Flow IAT Std', 'Active Min','Active Mean', 'Bwd IAT Mean',
                   'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'ACK Flag Count','Fwd PSH Flags','SYN Flag Count',
                   'Flow Packets/s', 'PSH Flag Count', 'Average Packet Size']

# chosen_connection_features = ['Subflow Fwd Packets', 'Active Min']

indices = range(len(chosen_connection_features))
zip_iterator = zip(chosen_connection_features, indices)
chosen_features_dict = dict(zip_iterator)


possible_protocols = {'6':[0.0,0.0,1.0],'17':[0.0,1.0,0.0], '0':[1.0,0.0,0.0],'':[0.0,0.0,0.0]}

# --------------------------------------
def get_chosen_connection_features():
    return chosen_connection_features
# --------------------------------------

# --------------------------------------
def getDataTreatmentParamaters(data_treatment, dataset):
    if dataset not in ['IDS2017', 'IDS2018']:
        raise Exception(f'Dataset not valid: {dataset}')
    if dataset == 'IDS2017':
        my_params = configparser.ConfigParser()
        my_params._interpolation = configparser.ExtendedInterpolation()
        my_params.read(generalParams['DIRECTORIES_IDS2017_LABELS_IDS2017']['data_treatment'] + f'{data_treatment}_parameters.ini')
    elif dataset == 'IDS2018':
        my_params = configparser.ConfigParser()
        my_params._interpolation = configparser.ExtendedInterpolation()
        my_params.read(generalParams['DIRECTORIES_IDS2018_LABELS_IDS2018']['data_treatment'] + f'{data_treatment}_parameters.ini')
    return my_params
# --------------------------------------
def treat_traces(traces, feature_name, data_treatment = 'none', dataset=None):
    if data_treatment == 'standardization':
        # print("Standardizing features")
        traces =  standardize_traces(traces, feature_name, dataset)
    elif data_treatment == 'normalization':
        # print("Normalizing features")
        traces =  normalize_traces(traces, feature_name, dataset)
    elif data_treatment == 'special_normalization':
        # print("Normalizing specially features")
        traces =  special_normalize_traces(traces, feature_name, dataset)
    elif data_treatment == 'robust_normalization':
        # print("Normalizing robustly features")
        traces =  robust_normalize_traces(traces, feature_name, dataset)
    return traces

def treat_feature(column, feature_name, data_treatment = 'none', dataset=None):
    if data_treatment == 'standardization':
        # print("Standardizing features")
        column =  standardize_feature(column, feature_name, dataset)
    elif data_treatment == 'normalization':
        # print("Normalizing features")
        column =  normalize_feature(column, feature_name, dataset)
    elif data_treatment == 'special_normalization':
        # print("Normalizing specially features")
        column =  special_normalize_feature(column, feature_name, dataset)
    elif data_treatment == 'robust_normalization':
        # print("Normalizing robustly features")
        column =  robust_normalize_feature(column, feature_name, dataset)
    return column


### STANDARDIZATION
def standardization_function(feature, labels, dataset=None):
    my_standardization = standardization
    if dataset is not None:
        my_standardization = getDataTreatmentParamaters("standardization", dataset)
    for name in features:
        if name in chosen_connection_features and (name+'_mean') in my_standardization['PARAMS'] and float(my_standardization['PARAMS'][name + '_std']) != 0:
            idx = chosen_features_dict[name]
            feature['feature_connection'][:,idx] = (feature['feature_connection'][:,idx] - float(my_standardization['PARAMS'][name + '_mean'])) / float(my_standardization['PARAMS'][name + '_std'])
    return feature, labels

def standardize_traces(traces, feature_to_standardize, dataset=None):
    for name in feature_to_standardize:
        idx = features_dict[name]
        traces[:,idx] = standardize_feature(traces[:,idx], name, dataset)
    return traces

def standardize_feature(column, feature_name, dataset=None):
    my_standardization = standardization
    if dataset is not None:
        my_standardization = getDataTreatmentParamaters("standardization", dataset)
    if (feature_name+'_std') not in my_standardization['PARAMS'] or (feature_name+'_mean') not in my_standardization['PARAMS']:
        return column
    if float(my_standardization['PARAMS'][feature_name + '_std']) == 0:
            column = 0
    elif (feature_name+'_mean') in my_standardization['PARAMS']:
        column = (column - float(my_standardization['PARAMS'][feature_name + '_mean'])) / float(my_standardization['PARAMS'][feature_name + '_std'])
    return column
###
### NORMALIZATION
def normalization_function(feature, labels, dataset=None):
    my_normalization = normalization
    if dataset is not None:
        my_normalization = getDataTreatmentParamaters("normalization", dataset)
    for name in features:
        if name in chosen_connection_features and (name+'_max') in my_normalization['PARAMS'] and (float(my_normalization['PARAMS'][name + '_max']) - float(my_normalization['PARAMS'][name + '_min'])) != 0:
            idx = chosen_features_dict[name]
            feature['feature_connection'][:,idx] = (feature['feature_connection'][:,idx] - float(my_normalization['PARAMS'][name + '_min'])) / (float(my_normalization['PARAMS'][name + '_max']) - float(my_normalization['PARAMS'][name + '_min']))
    return feature, labels

def normalize_traces(traces, feature_to_standardize, dataset=None):
    for name in feature_to_standardize:
        idx = features_dict[name]
        temp = normalize_feature(traces[:,idx], name, dataset)
        traces[:,idx] = temp
    return traces

def normalize_feature(column, feature_name, dataset=None):
    my_normalization = normalization
    if dataset is not None:
        my_normalization = getDataTreatmentParamaters("normalization", dataset)
    if (feature_name+'_min') not in my_normalization['PARAMS'] or (feature_name+'_max') not in my_normalization['PARAMS']:
        return column
    if (feature_name+'_min') in my_normalization['PARAMS'] and (feature_name+'_max') in my_normalization['PARAMS']:
        if (float(my_normalization['PARAMS'][feature_name + '_max']) - float(my_normalization['PARAMS'][feature_name + '_min'])) == 0:
            column = [0 for _ in column]
        else:
            column = (column.astype(np.float) - float(my_normalization['PARAMS'][feature_name + '_min'])) / (float(my_normalization['PARAMS'][feature_name + '_max']) - float(my_normalization['PARAMS'][feature_name + '_min']))
    return column
###
### SPECIAL NORMALIZATION
def special_normalization_function(feature, labels, dataset=None):
    my_normalization = special_normalization
    if dataset is not None:
        my_normalization = getDataTreatmentParamaters("special_normalization", dataset)
    for name in features:
        if name in chosen_connection_features and (name+'_max') in my_normalization['PARAMS']  and (name+'_min') in my_normalization['PARAMS']:
            if (float(my_normalization['PARAMS'][name + '_max']) - float(my_normalization['PARAMS'][name + '_min'])) != 0:
                idx = chosen_features_dict[name]
                feature['feature_connection'][:,idx] = [((x - float(my_normalization['PARAMS'][name + '_min'])) / (float(my_normalization['PARAMS'][name + '_max']) - float(my_normalization['PARAMS'][name + '_min']))) 
                                                        if x < float(my_normalization['PARAMS'][name + '_max']) else 1 
                                                        for x in feature['feature_connection'][:,idx]]
        else:
             (feature, labels) = normalization_function(feature, labels)       
    return feature, labels

def special_normalize_traces(traces, feature_to_standardize, dataset=None):
    for name in feature_to_standardize:
        idx = features_dict[name]
        traces[:,idx] = special_normalize_feature(traces[:,idx], name, dataset)
    return traces

def special_normalize_feature(column, feature_name, dataset=None):
    my_normalization = special_normalization
    if dataset is not None:
        my_normalization = getDataTreatmentParamaters("special_normalization", dataset)
    if (feature_name+'_min') not in my_normalization['PARAMS'] or (feature_name+'_max') not in my_normalization['PARAMS']:
        return column
    if (feature_name+'_min') in my_normalization['PARAMS'] and (feature_name+'_max') in my_normalization['PARAMS']:
        if (float(my_normalization['PARAMS'][feature_name + '_max']) - float(my_normalization['PARAMS'][feature_name + '_min'])) == 0:
            column = [0 for _ in column]
        else:
            column = [((x - float(my_normalization['PARAMS'][feature_name + '_min'])) / (float(my_normalization['PARAMS'][feature_name + '_max']) - float(my_normalization['PARAMS'][feature_name + '_min']))) 
                                                        if x < float(my_normalization['PARAMS'][feature_name + '_max']) else 1 
                                                        for x in column]
    else:
        column = normalize_feature(column, feature_name, dataset)       
    return column
###
### ROBUST NORMALIZATION
def robust_normalization_function(feature, labels, dataset=None):
    my_normalization = robust_normalization
    if dataset is not None:
        my_normalization = getDataTreatmentParamaters("robust_normalization", dataset)
    for name in features:
        if name in chosen_connection_features and (name+'_max') in my_normalization['PARAMS']  and (name+'_min') in my_normalization['PARAMS'] and (name+'_mdn') in my_normalization['PARAMS']:
            if (float(my_normalization['PARAMS'][name + '_max']) - float(my_normalization['PARAMS'][name + '_min'])) != 0:
                idx = chosen_features_dict[name]
                feature['feature_connection'][:,idx] = ((feature['feature_connection'][:,idx] - float(my_normalization['PARAMS'][name + '_mdn'])) / (float(my_normalization['PARAMS'][name + '_max']) - float(my_normalization['PARAMS'][name + '_min']))) 
                                                        
        else:
             (feature, labels) = normalization_function(feature, labels)       
    return feature, labels

def robust_normalize_traces(traces, feature_to_standardize, dataset=None):
    for name in feature_to_standardize:
        idx = features_dict[name]
        traces[:,idx] = robust_normalize_feature(traces[:,idx], name, dataset)
    return traces

def robust_normalize_feature(column, feature_name, dataset=None):
    my_normalization = robust_normalization
    if dataset is not None:
        my_normalization = getDataTreatmentParamaters("robust_normalization", dataset)
    if (feature_name+'_min') not in my_normalization['PARAMS'] or (feature_name+'_max') not in my_normalization['PARAMS'] or (feature_name+'_mdn') not in my_normalization['PARAMS']:
        return column
    if (feature_name+'_min') in my_normalization['PARAMS'] and (feature_name+'_max') in my_normalization['PARAMS']:
        if (float(my_normalization['PARAMS'][feature_name + '_max']) - float(my_normalization['PARAMS'][feature_name + '_min'])) == 0:
            column = [0 for _ in column]
        else:
            column = [((x - float(my_normalization['PARAMS'][feature_name + '_mdn'])) / (float(my_normalization['PARAMS'][feature_name + '_max']) - float(my_normalization['PARAMS'][feature_name + '_min']))) for x in column]
    else:
        column = normalize_feature(column, feature_name, dataset)       
    return column
###

def transform_ips(ip):
    # transform it into a 12 bit string
    ip = ip.split('.')
    for i in range(len(ip)):
        ip[i] = '0'*(3 - len(ip[i])) + ip[i]

    ip = ''.join(ip)
    try:
        result = [float(v) for v in ip if v != '.']
    except:
        result = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    return result

def get_feature(trace, feature_name, parse=True, parseOnlyFloats = False):
    if parse:
        if feature_name == 'Label':
            attack = trace[-1]
            if parseOnlyFloats:
                return attack
            return attacks_dict.get(attack)
        else:
            idx = features_dict[feature_name]
            feature = trace[idx]

            if 'ID' in feature_name:
                if parseOnlyFloats:
                    return feature
                return feature
            elif 'IP' in feature_name:
                if parseOnlyFloats:
                    return feature
                return transform_ips(feature)
            elif feature_name == 'Protocol':
                if parseOnlyFloats:
                    return feature
                # Transform to a one-hot encoding
                return possible_protocols.get(feature)
            else:
                try:
                    value = float(feature)
                    if value != float('inf') and value != float('nan') and value != float('-inf'):
                        return value
                    else:
                        return 0
                except:
                    return 0
    else:
        idx = features_dict[feature_name]
        return trace[idx]

# constructs a dictionary with all the chosen features of the ids 2017
def get_connection_features(trace, final_feature, type):
    connection_features = {}
    
    for f in chosen_connection_features:
        connection_features[f] = get_feature(trace, f)

    connection_features['Label'] = final_feature
    connection_features['type'] = type
    return connection_features

def traces_to_graph(traces):
    G = nx.MultiDiGraph()
    #G = nx.MultiGraph()

    n = len(traces)
    for i in range(n):
        trace = traces[i]

        dst_name = 'Destination IP'
        src_name = 'Source IP'

        if get_feature(trace, dst_name, parse=False) not in G.nodes():
            G.add_node(get_feature(trace, dst_name, parse=False), ip = get_feature(trace,dst_name)[:-3], type=1)

        if get_feature(trace, src_name, parse=False) not in G.nodes():
            G.add_node(get_feature(trace, src_name, parse=False), ip=get_feature(trace, src_name)[:-3], type=1)

        label_num = get_feature(trace, 'Label')
        final_label = np.zeros(15)
        if label_num != -1: # if it is an attack
            final_label[label_num] = 1

        connection_features = get_connection_features(trace, final_label, 2)
        G.add_node('con_' + str(i), **connection_features)

        # these edges connect the ports with the IP node (connecting all the servers together)
        G.add_edge('con_' + str(i), get_feature(trace, dst_name, parse=False))
        G.add_edge('con_' + str(i), get_feature(trace, src_name, parse=False))
        G.add_edge(get_feature(trace, dst_name, parse=False), 'con_' + str(i))
        G.add_edge(get_feature(trace, src_name, parse=False), 'con_' + str(i))

    return G

def assign_indices(G):
    indices_ip = {}
    indices_connection = {}
    counter_ip = 0
    counter_connection = 0

    for v in G.nodes():
        if G.nodes()[v]['type'] == 1:
            if v not in indices_ip:
                indices_ip[v] = counter_ip
                counter_ip += 1
        else:
            if v not in indices_connection:
                indices_connection[v] = counter_connection
                counter_connection += 1
    return indices_ip, indices_connection, counter_ip, counter_connection

def process_adjacencies(G):
    indices_ip, indices_connection, counter_ip, counter_connection = assign_indices(G)
    src_ip_to_connection, dst_ip_to_connection, src_connection_to_ip, dst_connection_to_ip  = [], [], [], []

    for e in G.edges(): # each edge is a pair of the source, dst node
        if 'con' not in e[0]:
            ip_node = e[0] # ip
            connection_node = e[1]  # connection
        
            # connection to ip and ip to connection
            src_ip_to_connection.append(indices_ip[ip_node])
            dst_ip_to_connection.append(indices_connection[connection_node])
        
        else:
            ip_node = e[1]
            connection_node = e[0]
            
            # connection to ip and ip to connection
            src_connection_to_ip.append(indices_connection[connection_node])
            dst_connection_to_ip.append(indices_ip[ip_node])

    return src_ip_to_connection, dst_ip_to_connection, src_connection_to_ip, dst_connection_to_ip, counter_ip, counter_connection

def graph_to_dict(G, for_framework='tensorflow', data_treatment='none', dataset=None):
    if not isinstance(data_treatment, str):
        data_treatment = data_treatment.decode("utf-8")
    #edge features
    connection_features = np.array([])
    first = True

    for f in chosen_connection_features:
        aux = np.array(list(nx.get_node_attributes(G, f).values()))

        if first:
            connection_features = np.expand_dims(aux, axis=-1)
            first = False
        else:
            if len(aux.shape) == 1:
                aux = np.expand_dims(aux, -1)

            connection_features = np.concatenate([connection_features, aux], axis=1)

    # obtain the labels of the nodes (indicator r.v indicating whether it has been infected or not)
    label = np.array(list(nx.get_node_attributes(G, 'Label').values())).astype('float32')

    # obtain the adjacencies
    src_ip_to_connection, dst_ip_to_connection, src_connection_to_ip, dst_connection_to_ip, n_i, n_c = process_adjacencies(G)

    if for_framework == 'pytorch':
        features = {
            'feature_connection': torch.tensor(connection_features).float(),
            'n_i': n_i,
            'n_c': n_c,
            'src_ip_to_connection': torch.tensor(src_ip_to_connection),
            'dst_ip_to_connection': torch.tensor(dst_ip_to_connection),
            'src_connection_to_ip': torch.tensor(src_connection_to_ip),
            'dst_connection_to_ip': torch.tensor(dst_connection_to_ip)
        }
    else:
        features = {
            'feature_connection': connection_features,
            'n_i': n_i,
            'n_c': n_c,
            'src_ip_to_connection': src_ip_to_connection,
            'dst_ip_to_connection': dst_ip_to_connection,
            'src_connection_to_ip': src_connection_to_ip,
            'dst_connection_to_ip': dst_connection_to_ip
        }

    if data_treatment == 'standardization':
        features, label =  standardization_function(features, label, dataset)
    elif data_treatment == 'normalization':
        # print("Normalizing features")
        features, label =  normalization_function(features, label, dataset)
    elif data_treatment == 'special_normalization':
        # print("Normalizing specially features")
        features, label =  special_normalization_function(features, label, dataset)
    elif data_treatment == 'robust_normalization':
        # print("Normalizing robustly features")
        features, label =  robust_normalization_function(features, label, dataset)
    # print("Returning features")
    return (features, label)
