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
import networkx as nx
import matplotlib.pyplot as plt
import torch
import configparser

def list_to_dict(l):
    indices = range(len(l))
    zip_iterator = zip(l,indices)
    return dict(zip_iterator)

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


class Generator:
    # --------------------------------------
    # MAP THAT TELLS US, GIVEN A FEATURE, ITS POSITION (IDS 2017 and IDS 2018)
    FEATURES = ['Flow ID','Source IP','Source Port','Destination IP','Destination Port','Protocol','Timestamp','Flow Duration','Total Fwd Packets',
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
    
    FEATURES_DICT = list_to_dict(FEATURES)


    ATTACK_NAMES_IDS2017 = ['SSH-Patator', 'DoS GoldenEye', 'PortScan', 'DoS Slowhttptest', 'Web Attack  Brute Force', 'Bot', 'Web Attack  Sql Injection', 
                            'Web Attack  XSS', 'Infiltration', 'DDoS', 'DoS slowloris', 'Heartbleed', 'FTP-Patator', 'DoS Hulk','BENIGN']
    ATTACK_NAMES_IDS2018 = ['FTP-BruteForce', 'SSH-BruteForce', 'DoS-GoldenEye', 'DoS-Slowloris', 'DoS-SlowHTTPTest', 'DoS-Hulk', 'DDoS attacks-LOIC-HTTP', 
                            'DDoS-LOIC-UDP', 'DDOS-HOIC', 'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection', 'Infiltration', 'Bot', 'BENIGN']
    
    CHOSEN_CONNECTION_FEATURES = ['Source Port', 'Destination Port', 'Bwd Packet Length Min', 'Subflow Fwd Packets',
                        'Total Length of Fwd Packets', 'Fwd Packet Length Mean', 'Total Length of Bwd Packets',
                        'Fwd Packet Length Std', 'Fwd IAT Min', 'Flow IAT Min', 'Flow IAT Mean', 'Bwd Packet Length Std',
                        'Subflow Fwd Bytes', 'Flow Duration', 'Flow IAT Std', 'Active Min','Active Mean', 'Bwd IAT Mean',
                        'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'ACK Flag Count','Fwd PSH Flags','SYN Flag Count',
                        'Flow Packets/s', 'PSH Flag Count', 'Average Packet Size']
    
    CHOSEN_FEATURES_DICT = list_to_dict(CHOSEN_CONNECTION_FEATURES)

    # CHOSEN_CONNECTION_FEATURES = ['Subflow Fwd Packets', 'Active Min']

    POSSIBLE_PROTOCOLS = {'6':[0.0,0.0,1.0],'17':[0.0,1.0,0.0], '0':[1.0,0.0,0.0],'':[0.0,0.0,0.0]}


    def __init__(self, dataset='IDS2017', dataset_labels=None, for_framework='tensorflow', data_treatment='none', data_treatment_params_path=None) -> None:
        if dataset not in ['IDS2017', 'IDS2018']:
            raise Exception(f'Dataset not valid: {dataset}')
        if not isinstance(data_treatment, str):
            data_treatment = data_treatment.decode("utf-8")

        self.data_treatment = data_treatment
        self.dataset = dataset
        self.for_framework = for_framework

        match self.data_treatment:
            case 'standardization':
                self.standardization = configparser.ConfigParser()
                self.standardization._interpolation = configparser.ExtendedInterpolation()
                self.standardization.read(data_treatment_params_path + 'standardization_parameters.ini')
                self.data_treatment_function = self.standardization_function
            case 'normalization':
                self.normalization = configparser.ConfigParser()
                self.normalization._interpolation = configparser.ExtendedInterpolation()
                self.normalization.read(data_treatment_params_path + 'normalization_parameters.ini')
                self.data_treatment_function = self.normalization_function
            case 'special_normalization':
                self.special_normalization = configparser.ConfigParser()
                self.special_normalization._interpolation = configparser.ExtendedInterpolation()
                self.special_normalization.read(data_treatment_params_path + 'special_normalization_parameters.ini')
                self.data_treatment_function = self.special_normalization_function
            case 'robust_normalization':
                self.robust_normalization = configparser.ConfigParser()
                self.robust_normalization._interpolation = configparser.ExtendedInterpolation()
                self.robust_normalization.read(data_treatment_params_path + 'robust_normalization_parameters.ini')
                self.data_treatment_function = self.robust_normalization_function
            case _:
                self.data_treatment_function = self.none_function

        # ATTACKS IDS 2017 and IDS 2018
        if dataset_labels == 'IDS2017' or (dataset_labels is None and dataset == 'IDS2017'):
            self.attack_names = self.ATTACK_NAMES_IDS2017
        elif dataset_labels == 'IDS2018' or (dataset_labels is None and dataset == 'IDS2018'):
            self.attack_names = self.ATTACK_NAMES_IDS2018

        self.attacks_dict = list_to_dict(self.attack_names)

    # --------------------------------------
    def treat_traces(self, traces, feature_name):
        if self.data_treatment == 'standardization':
            # print("Standardizing features")
            traces =  self.standardize_traces(traces, feature_name)
        elif self.data_treatment == 'normalization':
            # print("Normalizing features")
            traces =  self.normalize_traces(traces, feature_name)
        elif self.data_treatment == 'special_normalization':
            # print("Normalizing specially features")
            traces =  self.special_normalize_traces(traces, feature_name)
        elif self.data_treatment == 'robust_normalization':
            # print("Normalizing robustly features")
            traces =  self.robust_normalize_traces(traces, feature_name)
        return traces

    def treat_feature(self, column, feature_name, data_treatment = 'none', dataset=None):
        if data_treatment == 'standardization':
            # print("Standardizing features")
            column =  self.standardize_feature(column, feature_name, dataset)
        elif data_treatment == 'normalization':
            # print("Normalizing features")
            column =  self.normalize_feature(column, feature_name, dataset)
        elif data_treatment == 'special_normalization':
            # print("Normalizing specially features")
            column =  self.special_normalize_feature(column, feature_name, dataset)
        elif data_treatment == 'robust_normalization':
            # print("Normalizing robustly features")
            column =  self.robust_normalize_feature(column, feature_name, dataset)
        return column

    ### NO DATA TREATMENT
    def none_function(self, feature, labels):
        return feature, labels


    ### STANDARDIZATION
    def standardization_function(self, feature, labels):
        my_standardization = self.standardization
        for name in self.FEATURES:
            if name in self.CHOSEN_CONNECTION_FEATURES and (name+'_mean') in my_standardization['PARAMS'] and float(my_standardization['PARAMS'][name + '_std']) != 0:
                idx = self.CHOSEN_FEATURES_DICT[name]
                feature['feature_connection'][:,idx] = (feature['feature_connection'][:,idx] - float(my_standardization['PARAMS'][name + '_mean'])) / float(my_standardization['PARAMS'][name + '_std'])
        return feature, labels

    def standardize_traces(self, traces, feature_to_standardize):
        for name in feature_to_standardize:
            idx = self.FEATURES_DICT[name]
            traces[:,idx] = self.standardize_feature(traces[:,idx], name)
        return traces

    def standardize_feature(self, column, feature_name):
        my_standardization = self.standardization
        if (feature_name+'_std') not in my_standardization['PARAMS'] or (feature_name+'_mean') not in my_standardization['PARAMS']:
            return column
        if float(my_standardization['PARAMS'][feature_name + '_std']) == 0:
                column = 0
        elif (feature_name+'_mean') in my_standardization['PARAMS']:
            column = (column - float(my_standardization['PARAMS'][feature_name + '_mean'])) / float(my_standardization['PARAMS'][feature_name + '_std'])
        return column
    ###
    ### NORMALIZATION
    def normalization_function(self, feature, labels):
        my_normalization = self.normalization
        for name in self.FEATURES:
            if name in self.CHOSEN_CONNECTION_FEATURES and (name+'_max') in my_normalization['PARAMS'] and (float(my_normalization['PARAMS'][name + '_max']) - float(my_normalization['PARAMS'][name + '_min'])) != 0:
                idx = self.CHOSEN_FEATURES_DICT[name]
                feature['feature_connection'][:,idx] = (feature['feature_connection'][:,idx] - float(my_normalization['PARAMS'][name + '_min'])) / (float(my_normalization['PARAMS'][name + '_max']) - float(my_normalization['PARAMS'][name + '_min']))
        return feature, labels

    def normalize_traces(self, traces, feature_to_standardize):
        for name in feature_to_standardize:
            idx = self.FEATURES_DICT[name]
            temp = self.normalize_feature(traces[:,idx], name)
            traces[:,idx] = temp
        return traces

    def normalize_feature(self, column, feature_name):
        my_normalization = self.normalization
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
    def special_normalization_function(self, feature, labels):
        my_normalization = self.special_normalization
        for name in self.FEATURES:
            if name in self.CHOSEN_CONNECTION_FEATURES and (name+'_max') in my_normalization['PARAMS']  and (name+'_min') in my_normalization['PARAMS']:
                if (float(my_normalization['PARAMS'][name + '_max']) - float(my_normalization['PARAMS'][name + '_min'])) != 0:
                    idx = self.CHOSEN_FEATURES_DICT[name]
                    feature['feature_connection'][:,idx] = [((x - float(my_normalization['PARAMS'][name + '_min'])) / (float(my_normalization['PARAMS'][name + '_max']) - float(my_normalization['PARAMS'][name + '_min']))) 
                                                            if x < float(my_normalization['PARAMS'][name + '_max']) else 1 
                                                            for x in feature['feature_connection'][:,idx]]
            else:
                (feature, labels) = self.normalization_function(feature, labels)       
        return feature, labels

    def special_normalize_traces(self, traces, feature_to_standardize):
        for name in feature_to_standardize:
            idx = self.FEATURES_DICT[name]
            traces[:,idx] = self.special_normalize_feature(traces[:,idx], name)
        return traces

    def special_normalize_feature(self, column, feature_name):
        my_normalization = self.special_normalization
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
            column = self.normalize_feature(column, feature_name)       
        return column
    ###
    ### ROBUST NORMALIZATION
    def robust_normalization_function(self, feature, labels):
        my_normalization = self.robust_normalization
        for name in self.FEATURES:
            if name in self.CHOSEN_CONNECTION_FEATURES and (name+'_max') in my_normalization['PARAMS']  and (name+'_min') in my_normalization['PARAMS'] and (name+'_mdn') in my_normalization['PARAMS']:
                if (float(my_normalization['PARAMS'][name + '_max']) - float(my_normalization['PARAMS'][name + '_min'])) != 0:
                    idx = self.CHOSEN_FEATURES_DICT[name]
                    feature['feature_connection'][:,idx] = ((feature['feature_connection'][:,idx] - float(my_normalization['PARAMS'][name + '_mdn'])) / (float(my_normalization['PARAMS'][name + '_max']) - float(my_normalization['PARAMS'][name + '_min']))) 
                                                            
            else:
                (feature, labels) = self.normalization_function(feature, labels)       
        return feature, labels

    def robust_normalize_traces(self, traces, feature_to_standardize):
        for name in feature_to_standardize:
            idx = self.FEATURES_DICT[name]
            traces[:,idx] = self.robust_normalize_feature(traces[:,idx], name)
        return traces

    def robust_normalize_feature(self, column, feature_name):
        my_normalization = self.robust_normalization
        if (feature_name+'_min') not in my_normalization['PARAMS'] or (feature_name+'_max') not in my_normalization['PARAMS'] or (feature_name+'_mdn') not in my_normalization['PARAMS']:
            return column
        if (feature_name+'_min') in my_normalization['PARAMS'] and (feature_name+'_max') in my_normalization['PARAMS']:
            if (float(my_normalization['PARAMS'][feature_name + '_max']) - float(my_normalization['PARAMS'][feature_name + '_min'])) == 0:
                column = [0 for _ in column]
            else:
                column = [((x - float(my_normalization['PARAMS'][feature_name + '_mdn'])) / (float(my_normalization['PARAMS'][feature_name + '_max']) - float(my_normalization['PARAMS'][feature_name + '_min']))) for x in column]
        else:
            column = self.normalize_feature(column, feature_name)       
        return column
    ###

    def get_feature(self, trace, feature_name, parse=True, parseOnlyFloats = False):
        if parse:
            if feature_name == 'Label':
                attack = trace[-1]
                if parseOnlyFloats:
                    return attack
                return self.attacks_dict.get(attack)
            else:
                idx = self.FEATURES_DICT[feature_name]
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
                    return self.POSSIBLE_PROTOCOLS.get(feature)
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
            idx = self.FEATURES_DICT[feature_name]
            return trace[idx]

    # constructs a dictionary with all the chosen features of the ids 2017
    def get_connection_features(self, trace, final_feature, type):
        connection_features = {}
        
        for f in self.CHOSEN_CONNECTION_FEATURES:
            connection_features[f] = self.get_feature(trace, f)

        connection_features['Label'] = final_feature
        connection_features['type'] = type
        return connection_features

    def traces_to_graph(self, traces):
        G = nx.MultiDiGraph()
        #G = nx.MultiGraph()

        n = len(traces)
        for i in range(n):
            trace = traces[i]

            dst_name = 'Destination IP'
            src_name = 'Source IP'

            if self.get_feature(trace, dst_name, parse=False) not in G.nodes():
                G.add_node(self.get_feature(trace, dst_name, parse=False), ip = self.get_feature(trace,dst_name)[:-3], type=1)

            if self.get_feature(trace, src_name, parse=False) not in G.nodes():
                G.add_node(self.get_feature(trace, src_name, parse=False), ip=self.get_feature(trace, src_name)[:-3], type=1)

            label_num = self.get_feature(trace, 'Label')
            final_label = np.zeros(15)
            if label_num != -1: # if it is an attack
                final_label[label_num] = 1

            connection_features = self.get_connection_features(trace, final_label, 2)
            G.add_node('con_' + str(i), **connection_features)

            # these edges connect the ports with the IP node (connecting all the servers together)
            G.add_edge('con_' + str(i), self.get_feature(trace, dst_name, parse=False))
            G.add_edge('con_' + str(i), self.get_feature(trace, src_name, parse=False))
            G.add_edge(self.get_feature(trace, dst_name, parse=False), 'con_' + str(i))
            G.add_edge(self.get_feature(trace, src_name, parse=False), 'con_' + str(i))

        return G

    def assign_indices(self, G):
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

    def process_adjacencies(self, G):
        indices_ip, indices_connection, counter_ip, counter_connection = self.assign_indices(G)
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

    def graph_to_dict(self, G):
        
        #edge features
        connection_features = np.array([])
        first = True

        for f in self.CHOSEN_CONNECTION_FEATURES:
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
        src_ip_to_connection, dst_ip_to_connection, src_connection_to_ip, dst_connection_to_ip, n_i, n_c = self.process_adjacencies(G)

        if self.for_framework == 'pytorch':
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

        features, label =  self.data_treatment_function(features, label)

        # print("Returning features")
        return (features, label)
