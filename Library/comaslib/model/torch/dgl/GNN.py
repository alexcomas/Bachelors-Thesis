"""
   Copyright 2022 Universitat Politècnica de Catalunya
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
import os
import glob
import sys
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch_scatter
from torch.nn import CrossEntropyLoss
from ....utils.torch.torch_utils import configWandB
import wandb
from ....utils.ProgressBar import ProgressBar
from ....data.generator import Generator

from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, precision_score, recall_score, confusion_matrix

class GNN(torch.nn.Module):

    def __init__(self, config, loadEpoch : int = None, loadBestEpoch=False):
        super(GNN, self).__init__()
        
        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file
        self.config = config
        if self.config['RUN_CONFIG']['wandb'] == "True":
            self.hyperparameters = configWandB(self.config['HYPERPARAMETERS'])
        else:
            self.hyperparameters = self.config['HYPERPARAMETERS']
        # GRU Cells used in the Message Passing step
        self.ip_update = torch.nn.GRUCell(int(self.hyperparameters['node_state_dim']), 
                int(self.hyperparameters['node_state_dim']))
        self.connection_update =  torch.nn.GRUCell(int(self.hyperparameters['node_state_dim']), 
                int(self.hyperparameters['node_state_dim']))
        self.GRUCellActivation = torch.nn.Tanh()
        
        self.message_func1 = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(in_features=int(self.hyperparameters['node_state_dim'])*2,
                                      out_features=int(self.hyperparameters['node_state_dim'])),
                torch.nn.ReLU(inplace = True)
        )
        self.message_func2 =  torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(in_features=int(self.hyperparameters['node_state_dim'])*2,
                                      out_features=int(self.hyperparameters['node_state_dim'])),
                torch.nn.ReLU(inplace = True)
        )

        # Readout Neural Network. It expects as input the path states and outputs the per-path delay
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 15)
        )

        self.extendedMetrics = self.config['RUN_CONFIG']['extended_metrics'] == 'True'

        useCuda = self.useCuda()

        if(useCuda):
            self.to(device=int(self.config['RUN_CONFIG']['cuda_device']))
        model_in_gpu = next(self.parameters()).is_cuda
        if(model_in_gpu):
            print("Model in the GPU.")
        else:
            print("Model not in the GPU.")

        # We load better checkpoint saved
        self.directory = 'DIRECTORIES_' + self.config['RUN_CONFIG']['dataset']
        self.ckpt_path = os.path.abspath(self.config[self.directory]['logs'])+ "\\ckpt"
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        files = glob.glob(self.ckpt_path + '/*.pt')
        self.optimizer = Adam(params=self.parameters() , lr=float(self.hyperparameters['learning_rate']), eps=1e-07, capturable=model_in_gpu)
        self.resume = bool(len(files) > 0) and (self.config['RUN_CONFIG']['sweep'] != "True")
        self.scheduler = ExponentialLR(self.optimizer, float(self.hyperparameters['gamma']))
        if self.config['RUN_CONFIG']['wandb'] == "True":
            wandb.init(project="TFG", entity="alexcomas", config=self.hyperparameters, resume=self.resume)
            self.hyperparameters = wandb.config
        if(self.resume):
            files = [(path, path.split('\\')[-1]) for path in files]
            files = [(path, '.'.join(el.split('.')[1:-1])) for (path, el) in files]
            files = [(path, el.split('-')) for (path, el) in files]
            files = [(path, {'Epoch': int(el[0]), 'Loss': float(el[1])}) for (path, el) in files]
            if loadEpoch != None:
                files = [(path, el) for (path, el) in files if el['Epoch'] == str(loadEpoch)]
            if loadBestEpoch:
                files.sort(key=lambda x: (x[1]['Loss'], -1*x[1]['Epoch']), reverse=False)
            else:
                files.sort(key=lambda x: (x[1]['Epoch'], x[1]['Loss']), reverse=True)
            filepath = files[0][0]
            
            if model_in_gpu:
                device=torch.device('cuda')
            else:
                device=torch.device('cpu')
            
            checkpoint = torch.load(filepath, device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loadedEpoch = checkpoint['epoch']
            self.startingLoss = checkpoint['loss']
            self.startingEpoch = self.loadedEpoch + 1

            print(f"Loaded epoch {self.loadedEpoch} with loss {self.startingLoss}.")
        else:
            self.startingEpoch = 1
            self.startingLoss = None
        
        for _ in range(self.startingEpoch-1):
            self.scheduler.step()

    def forward(self, dgl_graph):

        nn_output = []

        # adjacencies
        src_ip_to_connection = dgl_graph.edges(etype='Source IP->Connection')[0].type(torch.int64)
        dst_ip_to_connection = dgl_graph.edges(etype='Source IP->Connection')[1].type(torch.int64)
        src_connection_to_ip = dgl_graph.edges(etype='Connection->Destination IP')[0].type(torch.int64)
        dst_connection_to_ip =  dgl_graph.edges(etype='Connection->Destination IP')[1].type(torch.int64)

        # CREATE THE IP NODES
        #Encode only ones in the IP states
        ip_state = dgl_GNN.from_data_to_tensor(dgl_graph.nodes['IP'].data)

        # CREATE THE CONNECTION NODES
        # Initialize the initial hidden state for id nodes
        connection_state = dgl_GNN.from_data_to_tensor(dgl_graph.nodes['Connection'].data)
        
        # MESSAGE PASSING: ALL with ALL simoultaniously
        # We simply use sum aggregation for all, RNN for the update. The messages are formed with the source and edge parameters (NN)
        # Iterate t times doing the message passing
        for _ in range(int(self.hyperparameters['t'])):
            # IP to CONNECTION
            # compute the hidden-states
            ip_gather = torch.gather(ip_state, 0, src_ip_to_connection.unsqueeze(1).expand(-1, int(self.hyperparameters['node_state_dim'])))
            connection_gather = torch.gather(connection_state, 0, dst_ip_to_connection.unsqueeze(1).expand(-1, int(self.hyperparameters['node_state_dim'])))
            
            # apply the message function on the ip nodes
            nn_input = torch.concat([ip_gather, connection_gather], axis=1) #([port1, ... , portl, param1, ..., paramk])
            ip_message = self.message_func1(nn_input)

            # apply the aggregation function on the ip nodes
            ip_mean = torch_scatter.scatter_mean(ip_message, dst_ip_to_connection.unsqueeze(1).expand(-1, int(self.hyperparameters['node_state_dim'])), dim=0)

            # CONNECTION TO IP
            # compute the hidden-states
            connection_gather = torch.gather(connection_state, 0, src_connection_to_ip.unsqueeze(1).expand(-1, int(self.hyperparameters['node_state_dim'])))
            ip_gather = torch.gather(ip_state, 0, dst_connection_to_ip.unsqueeze(1).expand(-1, int(self.hyperparameters['node_state_dim'])))

            # apply the message function on the connection nodes
            nn_input = torch.concat([connection_gather, ip_gather], axis=1)
            connection_messages = self.message_func2(nn_input)

            # apply the aggregation function on the connection nodes
            connection_mean = torch_scatter.scatter_mean(connection_messages, dst_connection_to_ip, dim=0)

            ### UPDATE (both IP and connection simoultaniously)
            ## update of ip nodes
            # # Save sizes
            # ip_state_size = ip_state.size()
            # connection_mean_size = connection_mean.size()
            # # Flatten to two dimensions [N_batch*2, N_Hidden] and [N_batch*2, N_Hidden]
            # ip_state = torch.flatten(ip_state, start_dim=0, end_dim=1)
            # connection_mean = torch.flatten(connection_mean, start_dim=0, end_dim=1)
            # Update ip
            ip_state = self.ip_update(connection_mean, ip_state)
            ip_state = self.GRUCellActivation(ip_state)
            # # Recuperate original shape
            # ip_state = ip_state.reshape(ip_state_size)
            # connection_mean = connection_mean.reshape(connection_mean_size)

            ## update of connection nodes
            # # Save sizes
            # connection_state_size = connection_state.size()
            # ip_mean_size = ip_mean.size()
            # # Flatten to two dimensions [N_batch*2, N_Hidden] and [N_batch*N_graphs, N_Hidden]
            # connection_state = torch.flatten(connection_state, start_dim=0, end_dim=1)
            # ip_mean = torch.flatten(ip_mean, start_dim=0, end_dim=1)
            # Update ip
            connection_state = self.connection_update(ip_mean, connection_state)
            connection_state = self.GRUCellActivation(connection_state)
            # # Recuperate original shape
            # connection_state = connection_state.reshape(connection_state_size)
            # ip_mean = ip_mean.reshape(ip_mean_size)
        # apply the feed-forward nn
        nn_output = self.readout(connection_state)

        return nn_output

    def train_epoch(self, gen, epoch_index, steps_per_epoch = None):
        limit = len(gen)
        if steps_per_epoch != None:
            limit = steps_per_epoch
        self.train()
        profileOn = False
        loss = CrossEntropyLoss()
        running_loss = 0.
        total = 0
        labelsResult = []
        predictedResult = []
        progressbar = ProgressBar(limit, initial=0, unit="batch", type_prediction='linear', etaMode='remaining')
        if profileOn:
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=2,
                    warmup=2,
                    active=6,
                    repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./tensorboard-logs'),
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
            )
            profiler.start()
        for i in range(limit):
            batch = next(gen)
            self.optimizer.zero_grad(set_to_none=True)
            
            input = batch.to(self.device)

            labels = input.nodes['Connection'].data['Label']

            del input.nodes['Connection'].data['Label']

            # batch_size = labels.size(0)
            window_size = labels.size(0)
            number_classes = labels.size(1)
            
            output = self(input)

            # The label with the highest value will be our prediction 
            _, predicted = torch.max(output, 1)
            _, labelsIndex = torch.max(labels, 1)
            shape = [window_size, 1]
            
            # Add the original labels and predicted to a set so we can get metrics once the epoch is trained
            labelsResult.append(labelsIndex.view(-1))
            predictedResult.append(predicted.view(-1))

            # Adapt input and target for the cross entropy calculation
            output = output.permute([1,0])
            labels = labels.permute([1,0])
            # Calculate the loss
            train_loss = loss(output, labels)
            
            # Add the loss to the total
            running_loss += train_loss.item()
            total += window_size

            # train_loss.backward(create_graph=True)
            train_loss.backward()
            self.optimizer.step()
            if profileOn:
                profiler.step()

            # Update the output in the console
            progressbar.update(i+1, {'loss': running_loss/((i+1)*int(self.hyperparameters['batch_size']))})
        return (running_loss/(limit*int(self.hyperparameters['batch_size'])), torch.cat(labelsResult), torch.cat(predictedResult))

    def validate_epoch(self, gen, epoch_index, validation_steps = None):
        limit = len(gen)
        if validation_steps != None:
            limit = validation_steps
        loss = CrossEntropyLoss()

        running_vall_loss = 0.0
        labelsResult = []
        predictedResult = []
        progressbar = ProgressBar(limit, initial=0, unit="batch", type_prediction='linear', etaMode='remaining')
        total = 0
        
        with torch.no_grad(): 
            self.eval() 
            for i in range(limit):
                batch = next(gen)
                input = batch.to(self.device)

                labels = input.nodes['Connection'].data['Label']

                del input.nodes['Connection'].data['Label']

                # batch_size = labels.size(0)
                window_size = labels.size(0)
                number_classes = labels.size(1)
                
                output = self(input)

                # The label with the highest value will be our prediction 
                _, predicted = torch.max(output, 1)
                _, labelsIndex = torch.max(labels, 1)
                shape = [window_size, 1]
                
                # Add the original labels and predicted to a set so we can get metrics once the epoch is trained
                labelsResult.append(labelsIndex.view(-1))
                predictedResult.append(predicted.view(-1))

                # Adapt input and target for the cross entropy calculation
                output = output.permute([1,0])
                labels = labels.permute([1,0])

                # Calculate the loss
                eval_loss = loss(output, labels)

                # Add the loss to the total
                running_vall_loss += eval_loss.item()
                # total += batch_size*window_size
                
                # Update the output in the console
                progressbar.update(i+1, {'loss': running_vall_loss/((i+1)*int(self.hyperparameters['batch_size']))})
        return (running_vall_loss/(limit*int(self.hyperparameters['batch_size'])), torch.cat(labelsResult), torch.cat(predictedResult))

    def useCuda(self):
        useCuda = torch.cuda.is_available() and self.config['RUN_CONFIG']['force_cpu'] != 'True'
        if(useCuda):
            self.device = torch.cuda.current_device()
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = None
            torch.set_default_tensor_type('torch.FloatTensor')
        return useCuda

    def fit(self, training_generator, evaluating_generator, steps_per_epoch = None, validation_steps = None):
        start_time = time.time()
        for epoch in range(self.startingEpoch, int(self.hyperparameters['epochs'])+1):
            epoch_start_time = time.time()
            print("\nTraining epoch " + str(epoch))
            (train_loss, train_trueLabels, train_predLabels) = self.train_epoch(training_generator, epoch, steps_per_epoch=steps_per_epoch)
            print("\nValidating epoch " + str(epoch))
            (eval_loss, eval_trueLabels, eval_predLabels) = self.validate_epoch(evaluating_generator, epoch, validation_steps = validation_steps)
            print("")
            print("     Epoch stats:" + "    {:.2f}".format(time.time() - epoch_start_time) + "    {:.2f}".format(time.time() - start_time))
                 
            train_metrics = dgl_GNN.calculateMetrics(train_trueLabels, train_predLabels, extended=self.extendedMetrics)
            eval_metrics = dgl_GNN.calculateMetrics(eval_trueLabels, eval_predLabels, extended=self.extendedMetrics)

            print("             TRAIN ----- Loss: " + "{:.3f}".format(train_loss) + " - Accuracy: " + "{:.5f}".format(train_metrics['accuracy'])
                + " - Weighted F1: " + "{:.5f}".format(train_metrics['weighted_f1']) + " - Macro F1: " + "{:.5f}".format(train_metrics['macro_f1']))
            print("                 Labels (pred/true) +++++ " + ' - '.join([f"{Generator.attack_names[int(el[2])]}: {train_metrics.get('pred_count_'+ el[2],0)}/{train_metrics.get('true_count_'+ el[2], 0)}" for el in (key.split('_') for key in train_metrics.keys()) if len(el) > 2 and el[0] == 'pred' and el[1] == 'count']))
            
            print("             EVAL  ----- Loss: " + "{:.3f}".format(eval_loss) + " - Accuracy: " + "{:.5f}".format(eval_metrics['accuracy'])
                + " - Weighted F1: " + "{:.5f}".format(eval_metrics['weighted_f1']) + " - Macro F1: " + "{:.5f}".format(eval_metrics['macro_f1']))
            print("                 Labels (pred/true) +++++ " + ' - '.join([f"{Generator.attack_names[int(el[2])]}: {eval_metrics.get('pred_count_'+ el[2],0)}/{eval_metrics.get('true_count_'+ el[2], 0)}" for el in (key.split('_') for key in eval_metrics.keys()) if len(el) > 2 and el[0] == 'pred' and el[1] == 'count']))
            
            self.scheduler.step()

            if self.config['RUN_CONFIG']['sweep'] != "True":
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'lr':  self.scheduler.get_last_lr()[0],
                    'train_loss': train_loss,
                    'loss': eval_loss,
                    'accuracy': eval_metrics['accuracy'],
                    'f1_weighted': eval_metrics['weighted_f1'],
                    'f1_macro': eval_metrics['macro_f1'],
                    'class_accuracy': eval_metrics.get('class_accuracy', ''),
                    'class_precision': eval_metrics.get('class_precision', ''),
                    'class_recall': eval_metrics.get('class_recall', '')
                    }, self.ckpt_path + "\\weights." + "{0}-".format(str(epoch).zfill(4)) + "{:.2f}".format(eval_loss) + ".pt")
            
            ca_dic = dict()
            if self.extendedMetrics:
                for i, x in enumerate(eval_metrics['class_accuracy']):
                    ca_dic['_acc_'+str(i)] = x
                for i, x in enumerate(eval_metrics['class_precision']):
                    ca_dic['_pre_'+str(i)] = x
                for i, x in enumerate(eval_metrics['class_recall']):
                    ca_dic['_rec_'+str(i)] = x

            if self.config['RUN_CONFIG']['wandb'] == "True":
                wandb.log(dict({
                    '_epoch': epoch,
                    'train_f1_weighted': train_metrics['weighted_f1'],
                    'train_loss': train_loss,
                    'lr':  self.scheduler.get_last_lr()[0],
                    'loss': eval_loss,
                    'accuracy': eval_metrics['accuracy'],
                    'f1_weighted': eval_metrics['weighted_f1'],
                    'f1_macro': eval_metrics['macro_f1']
                }, **ca_dic))

    def evaluate(self, evaluating_generator, validation_steps = None):
        start_time = time.time()

        (eval_loss, eval_trueLabels, eval_predLabels) = self.validate_epoch(evaluating_generator, self.loadedEpoch, validation_steps = validation_steps)
        print("")
        print("     Epoch stats:" + "    {:.2f}".format(time.time() - start_time))
        
        eval_metrics = dgl_GNN.calculateMetrics(eval_trueLabels, eval_predLabels, extended=True)

        print("             EVAL  ----- Loss: " + "{:.3f}".format(eval_loss) + " - Accuracy: " + "{:.5f}".format(eval_metrics['accuracy'])
            + " - Weighted F1: " + "{:.5f}".format(eval_metrics['weighted_f1']) + " - Macro F1: " + "{:.5f}".format(eval_metrics['macro_f1']))
        print("                 Labels (pred/true) +++++ " + ' - '.join([f"{Generator.attack_names[int(el[2])]}: {eval_metrics.get('pred_count_'+ el[2],0)}/{eval_metrics.get('true_count_'+ el[2], 0)}" for el in (key.split('_') for key in eval_metrics.keys()) if len(el) > 2 and el[0] == 'pred' and el[1] == 'count']))
        return eval_metrics

    @staticmethod
    def calculateMetrics(trueLabels, predLabels, extended=True):
        result = dict()
        result['accuracy'] = accuracy_score(trueLabels.cpu(), predLabels.cpu())
        result['weighted_f1'] = f1_score(trueLabels.cpu(), predLabels.cpu(), average='weighted')
        result['macro_f1'] = f1_score(trueLabels.cpu(), predLabels.cpu(), average='macro')

        trueClasses, countTrueClasses = torch.unique(trueLabels, return_counts = True)
        for cl, count in zip(trueClasses.cpu().numpy(), countTrueClasses.cpu().numpy()):
            result[f'true_count_{cl}'] = count

        predClasses, countPredClasses = torch.unique(predLabels, return_counts = True)
        for cl, count in zip(predClasses.cpu().numpy(), countPredClasses.cpu().numpy()):
            result[f'pred_count_{cl}'] = count

        if extended:
            train_matrix = confusion_matrix(trueLabels.cpu(), predLabels.cpu())
            result['class_accuracy'] = train_matrix.diagonal()/train_matrix.sum(axis=1)

            result['class_precision'] = precision_score(trueLabels.cpu(), predLabels.cpu(), average=None, zero_division=0)
            result['class_recall'] = recall_score(trueLabels.cpu(), predLabels.cpu(), average=None, zero_division=0)
            result['class_f1'] = f1_score(trueLabels.cpu(), predLabels.cpu(), average=None, zero_division=0)
        
        return result

    @staticmethod
    def from_data_to_tensor(data):
        ret_list = []
        gen = (col for col in data if not col.startswith("h_"))
        for col in gen:
            ret_list.append(data[col])
        gen = (col for col in data if col.startswith("h_"))
        for col in gen:
            ret_list.append(data[col])
        return torch.stack(ret_list).transpose(0,1)