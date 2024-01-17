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

import tensorflow as tf
import sys
import os
import glob
from ...data.generator import Generator
import tensorflow_addons as tfa

class GNN(tf.keras.Model):

    def __init__(self, node_state_dim: int, t :int):
        super(GNN, self).__init__()

        self.node_state_dim = node_state_dim
        self.t = t

        # GRU Cells used in the Message Passing step
        self.ip_update = tf.keras.layers.GRUCell(self.node_state_dim, name='update_ip')
        self.connection_update = tf.keras.layers.GRUCell(self.node_state_dim, name='update_connection')

        self.message_func1 = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self.node_state_dim*2 ),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(self.node_state_dim,
                                      activation=tf.nn.relu)
            ]
        )
        self.message_func2 = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self.node_state_dim*2 ),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(self.node_state_dim,
                                      activation=tf.nn.relu)
            ]
        )


        # Readout Neural Network. It expects as input the path states and outputs the per-path delay
        self.readout = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.node_state_dim),
            tf.keras.layers.Dense(128,
                                  activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64,
                                  activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(15, activation = tf.nn.softmax)
        ])

    @tf.function
    def call(self, inputs):
        # connection features
        feature_connection = tf.squeeze(inputs['feature_connection'])

        # number of ip
        n_ips = inputs['n_i']

        # number of connections
        n_connections = inputs['n_c']

        # adjacencies
        src_ip_to_connection = tf.squeeze(inputs['src_ip_to_connection'])
        dst_ip_to_connection = tf.squeeze(inputs['dst_ip_to_connection'])
        src_connection_to_ip = tf.squeeze(inputs['src_connection_to_ip'])
        dst_connection_to_ip = tf.squeeze(inputs['dst_connection_to_ip'])

        # CREATE THE IP NODES
        #Encode only ones in the IP states
        ip_state = tf.ones((n_ips, self.node_state_dim))


        # CREATE THE CONNECTION NODES
        # Compute the shape for the  all-zero tensor for link_state
        shape = tf.stack([
            n_connections,
            self.node_state_dim - len(Generator.CHOSEN_CONNECTION_FEATURES)
        ], axis=0)

        # Initialize the initial hidden state for id nodes
        connection_state = tf.concat([
            feature_connection,
            tf.zeros(shape)
        ], axis=1)


        # MESSAGE PASSING: ALL with ALL simoultaniously
        # We simply use sum aggregation for all, RNN for the update. The messages are formed with the source and edge parameters (NN)
        # Iterate t times doing the message passing
        for _ in range(self.t):
            # IP to CONNECTION
            # compute the hidden-states
            ip_node_gather = tf.gather(ip_state, src_ip_to_connection)
            connection_gather = tf.gather(connection_state, dst_ip_to_connection)
            connection_gather = tf.squeeze(connection_gather)
            ip_gather = tf.squeeze(ip_node_gather)
            
            # apply the message function on the ip nodes
            nn_input = tf.concat([ip_gather, connection_gather], axis=1) #([port1, ... , portl, param1, ..., paramk])
            nn_input = tf.ensure_shape(nn_input,[None, self.node_state_dim*2])
            ip_message = self.message_func1(nn_input)

            # apply the aggregation function on the ip nodes
            ip_mean = tf.math.unsorted_segment_mean(ip_message, dst_ip_to_connection, n_connections)

            # CONNECTION TO IP
            # compute the hidden-states
            connection_node_gather = tf.gather(connection_state, src_connection_to_ip)
            ip_gather = tf.gather(ip_state, dst_connection_to_ip)
            ip_gather = tf.squeeze(ip_gather)
            connection_gather = tf.squeeze(connection_node_gather)

            # apply the message function on the connection nodes
            nn_input = tf.concat([connection_gather, ip_gather], axis=1)
            nn_input = tf.ensure_shape(nn_input, [None, self.node_state_dim * 2])
            connection_messages = self.message_func2(nn_input)

            # apply the aggregation function on the connection nodes
            connection_mean = tf.math.unsorted_segment_mean(connection_messages, dst_connection_to_ip, n_ips)


            # UPDATE (both IP and connection simoultaniously)
            #update of ip nodes
            connection_mean = tf.ensure_shape(connection_mean, [None, self.node_state_dim])
            ip_state, _ = self.ip_update(connection_mean, [ip_state])
            
            #update of connection nodes
            ip_mean = tf.ensure_shape(ip_mean, [None, self.node_state_dim])
            connection_state, _ = self.connection_update(ip_mean, [connection_state])

        # apply the feed-forward nn
        nn_output = self.readout(connection_state)
        return nn_output

    def _get_compiled_model(hyperparameters):
        model = GNN(int(hyperparameters['node_state_dim']),int(hyperparameters['t']))
        decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(float(hyperparameters['learning_rate']),
                                                                    int(hyperparameters['decay_steps']),
                                                                    float(hyperparameters['decay_rate']),
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

    def make_or_restore_model(hyperparameters, logs_dir, forceRestore = None):
        # Either restore the latest model, or create a fresh one
        checkpoint_dir = os.path.abspath(logs_dir + '/ckpt')
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
            gnn = GNN._get_compiled_model(hyperparameters)
            gnn.load_weights(latest_checkpoint)
            
        else:
            print("Creating a new model")
            gnn = GNN._get_compiled_model(hyperparameters)
        
        gnn.built = True
        return (gnn, startingEpoch)
