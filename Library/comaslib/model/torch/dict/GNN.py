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
import time
import torch
import numpy as np
from typing import NamedTuple
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch_scatter
from torch.nn import CrossEntropyLoss

from ....utils.torch.torch_utils import configWandB
from ....utils.ProgressBar import ProgressBar
from ....data.generator import Generator


from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, precision_score, recall_score, confusion_matrix

class Hyperparameters(NamedTuple):
    node_state_dim: int
    t: int
    epochs: int
    batch_size: int
    decay_rate: float
    decay_steps: int
    gamma: float

class GNN(torch.nn.Module):

    def __init__(self, node_state_dim: int, t :int, epochs: int = 20, batch_size: int = 1, decay_rate: float = 0.6, decay_steps:int = 50000, gamma:float = None):
        super(GNN, self).__init__()

        self.t = t
        self.node_state_dim = node_state_dim

        self.hyperparameters = Hyperparameters(
            node_state_dim= node_state_dim,
            t = t,
            epochs = epochs,
            batch_size = batch_size,
            decay_rate = decay_rate,
            decay_steps = decay_steps,
            gamma=gamma
        )

        self.ckpt_path = ""
        self.loss = None
        self.optimizer = None
        self.scheduler = None
        self.force_cpu = False
        self.is_cuda = False
        self.use_wandb = False
        self.sweep = False
        self.loadedEpoch = 0
        self.startingLoss = 1000
        self.startingEpoch = 1
        self.extended_metrics = False
        
        # GRU Cells used in the Message Passing step
        self.ip_update = torch.nn.GRUCell(self.node_state_dim, 
                self.node_state_dim)
        self.connection_update =  torch.nn.GRUCell(self.node_state_dim, 
                self.node_state_dim)
        self.GRUCellActivation = torch.nn.Tanh()
        
        self.message_func1 = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(in_features=self.node_state_dim*2,
                                      out_features=self.node_state_dim),
                torch.nn.ReLU(inplace = True)
        )
        self.message_func2 =  torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(in_features=self.node_state_dim*2,
                                      out_features=self.node_state_dim),
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
        
    
    def forward(self, inputs):

        nn_output = []

        # connection features
        feature_connection = inputs['feature_connection']
        # number of ip
        n_ips = inputs['n_i']

        # number of connections
        n_connections = inputs['n_c']

        # adjacencies
        src_ip_to_connection = inputs['src_ip_to_connection']
        dst_ip_to_connection = inputs['dst_ip_to_connection']
        src_connection_to_ip = inputs['src_connection_to_ip']
        dst_connection_to_ip = inputs['dst_connection_to_ip']

        # CREATE THE IP NODES
        #Encode only ones in the IP states
        ip_state = torch.nn.utils.rnn.pad_sequence(torch.ones((len(n_ips), int(max(n_ips).cpu().numpy()), self.node_state_dim)), batch_first=True)
        
        # tempList = []
        # for n in n_ips:
        #     tempList.append(torch.ones((n, self.node_state_dim)))
        # ip_state = torch.nn.utils.rnn.pad_sequence(tempList, batch_first=True)

        # CREATE THE CONNECTION NODES
        # Compute the shape for the  all-zero tensor for link_state
        shape = torch.stack([
            torch.tensor(feature_connection.size(0)),
            torch.max(n_connections),
            torch.tensor(self.node_state_dim - len(Generator.CHOSEN_CONNECTION_FEATURES))
        ], dim=0)

        # Initialize the initial hidden state for id nodes
        connection_state = torch.concat([
            feature_connection,
            torch.zeros(list(shape))
        ], dim=2)
        
        # MESSAGE PASSING: ALL with ALL simoultaniously
        # We simply use sum aggregation for all, RNN for the update. The messages are formed with the source and edge parameters (NN)
        # Iterate t times doing the message passing
        for _ in range(self.t):
            # IP to CONNECTION
            # compute the hidden-states
            ip_gather = torch.gather(ip_state, 1, src_ip_to_connection.unsqueeze(2).expand(-1 ,-1, self.node_state_dim))
            connection_gather = torch.gather(connection_state, 1, dst_ip_to_connection.unsqueeze(2).expand(-1 ,-1, self.node_state_dim))
            
            # apply the message function on the ip nodes
            nn_input = torch.concat([ip_gather, connection_gather], axis=2) #([port1, ... , portl, param1, ..., paramk])
            ip_message = self.message_func1(nn_input)

            # apply the aggregation function on the ip nodes
            ip_mean = torch_scatter.scatter_mean(ip_message, dst_ip_to_connection, dim=1)

            # CONNECTION TO IP
            # compute the hidden-states
            connection_gather = torch.gather(connection_state, 1, src_connection_to_ip.unsqueeze(2).expand(-1 ,-1, self.node_state_dim))
            ip_gather = torch.gather(ip_state, 1, dst_connection_to_ip.unsqueeze(2).expand(-1 ,-1, self.node_state_dim))

            # apply the message function on the connection nodes
            nn_input = torch.concat([connection_gather, ip_gather], axis=2)
            connection_messages = self.message_func2(nn_input)

            # apply the aggregation function on the connection nodes
            connection_mean = torch_scatter.scatter_mean(connection_messages, dst_connection_to_ip, dim=1)

            ### UPDATE (both IP and connection simoultaniously)
            ## update of ip nodes
            # Save sizes
            ip_state_size = ip_state.size()
            connection_mean_size = connection_mean.size()
            # Flatten to two dimensions [N_batch*2, N_Hidden] and [N_batch*2, N_Hidden]
            ip_state = torch.flatten(ip_state, start_dim=0, end_dim=1)
            connection_mean = torch.flatten(connection_mean, start_dim=0, end_dim=1)
            # Update ip
            ip_state = self.ip_update(connection_mean, ip_state)
            ip_state = self.GRUCellActivation(ip_state)
            # Recuperate original shape
            ip_state = ip_state.reshape(ip_state_size)
            connection_mean = connection_mean.reshape(connection_mean_size)

            ## update of connection nodes
            # Save sizes
            connection_state_size = connection_state.size()
            ip_mean_size = ip_mean.size()
            # Flatten to two dimensions [N_batch*2, N_Hidden] and [N_batch*N_graphs, N_Hidden]
            connection_state = torch.flatten(connection_state, start_dim=0, end_dim=1)
            ip_mean = torch.flatten(ip_mean, start_dim=0, end_dim=1)
            # Update ip
            connection_state = self.connection_update(ip_mean, connection_state)
            connection_state = self.GRUCellActivation(connection_state)
            # Recuperate original shape
            connection_state = connection_state.reshape(connection_state_size)
            ip_mean = ip_mean.reshape(ip_mean_size)
        # apply the feed-forward nn
        nn_output = self.readout(connection_state)

        return nn_output

    def train_epoch(self, gen, epoch_index, steps_per_epoch = None):
        limit = len(gen)
        if steps_per_epoch != None:
            limit = steps_per_epoch
        self.train()
        profileOn = False
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
            
            (input, labels) = batch

            batch_size = labels.size(0)
            window_size = labels.size(1)
            number_classes = labels.size(2)
            output = self(input)

            # The label with the highest value will be our prediction 
            _, predicted = torch.max(output, 2)
            _, labelsIndex = torch.max(labels, 2)
            shape = [batch_size, window_size, 1]
            
            # Add the original labels and predicted to a set so we can get metrics once the epoch is trained
            labelsResult.append(labelsIndex.view(-1))
            predictedResult.append(predicted.view(-1))

            # Adapt input and target for the cross entropy calculation
            output = output.permute([0,2,1])
            labels = labels.permute([0,2,1])
            # Calculate the loss
            train_loss = self.loss(output, labels)
            
            # Add the loss to the total
            running_loss += train_loss.item()
            total += batch_size*window_size

            # train_loss.backward(create_graph=True)
            train_loss.backward()
            self.optimizer.step()
            if profileOn:
                profiler.step()

            # Update the output in the console
            progressbar.update(i+1, {'loss': running_loss/((i+1)*int(self.hyperparameters.batch_size))})
        return (running_loss/(limit*int(self.hyperparameters.batch_size)), torch.cat(labelsResult), torch.cat(predictedResult))

    def validate_epoch(self, gen, epoch_index, validation_steps = None):
        limit = len(gen)
        if validation_steps != None:
            limit = validation_steps

        running_vall_loss = 0.0
        labelsResult = []
        predictedResult = []
        progressbar = ProgressBar(limit, initial=0, unit="batch", type_prediction='linear', etaMode='remaining')
        total = 0
        
        with torch.no_grad(): 
            self.eval() 
            for i in range(limit):
                batch = next(gen)
                (input, labels) = batch
                batch_size = labels.size(0)
                window_size = labels.size(1)
                number_classes = labels.size(2)
                output = self(input)
                
                # The label with the highest value will be our prediction
                _, predicted = torch.max(output, 2)
                _, labelsIndex = torch.max(labels, 2)

                # Add the original labels and predicted to a set so we can get metrics once the epoch is trained 
                labelsResult.append(labelsIndex.view(-1))
                predictedResult.append(predicted.view(-1))

                # Adapt input and target for the cross entropy calculation
                output = output.permute([0,2,1])
                labels = labels.permute([0,2,1])
                # Calculate the loss
                eval_loss = self.loss(output, labels)

                # Add the loss to the total
                running_vall_loss += eval_loss.item()
                # total += batch_size*window_size
                
                # Update the output in the console
                progressbar.update(i+1, {'loss': running_vall_loss/((i+1)*self.hyperparameters.batch_size)})
        return (running_vall_loss/(limit*self.hyperparameters.batch_size), torch.cat(labelsResult), torch.cat(predictedResult))

    def useCuda(self):
        useCuda = torch.cuda.is_available() and not self.force_cpu
        if(useCuda):
            self.device = torch.cuda.current_device()
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = None
            torch.set_default_tensor_type('torch.FloatTensor')
        return useCuda

    def fit(self, training_generator, evaluating_generator, steps_per_epoch = None, validation_steps = None):
        if self.hyperparameters.gamma is None:
            steps_per_epoch_local = steps_per_epoch
            if steps_per_epoch_local is None:
                steps_per_epoch_local = len(training_generator)
            self.scheduler.gamma = self.hyperparameters.decay_rate ** (steps_per_epoch_local / self.hyperparameters.decay_steps)
            print(f"Gamma has been calculated from Tensorflow's decay_rate({self.hyperparameters.decay_rate}), decay_steps ({self.hyperparameters.decay_steps}) "+
                   f"and steps_per_epoch ({steps_per_epoch_local})")
        
        print(f"Gamma: {self.scheduler.gamma}")
        start_time = time.time()
        for epoch in range(self.startingEpoch, int(self.hyperparameters.epochs)+1):
            epoch_start_time = time.time()
            print("\nTraining epoch " + str(epoch))
            (train_loss, train_trueLabels, train_predLabels) = self.train_epoch(training_generator, epoch, steps_per_epoch=steps_per_epoch)
            print("\nValidating epoch " + str(epoch))
            (eval_loss, eval_trueLabels, eval_predLabels) = self.validate_epoch(evaluating_generator, epoch, validation_steps = validation_steps)
            print("")
            print("     Epoch stats:" + "    {:.2f}".format(time.time() - epoch_start_time) + "    {:.2f}".format(time.time() - start_time))
                 
            train_metrics = GNN.calculateMetrics(train_trueLabels, train_predLabels, extended=self.extended_metrics)
            eval_metrics = GNN.calculateMetrics(eval_trueLabels, eval_predLabels, extended=self.extended_metrics)

            print("             TRAIN ----- Loss: " + "{:.3f}".format(train_loss) + " - Accuracy: " + "{:.5f}".format(train_metrics['accuracy'])
                + " - Weighted F1: " + "{:.5f}".format(train_metrics['weighted_f1']) + " - Macro F1: " + "{:.5f}".format(train_metrics['macro_f1']))
            print("                 Labels (pred/true) +++++ " + ' - '.join([f"{training_generator.generator.attack_names[int(el[2])]}: {train_metrics.get('pred_count_'+ el[2],0)}/{train_metrics.get('true_count_'+ el[2], 0)}" for el in (key.split('_') for key in train_metrics.keys()) if len(el) > 2 and el[0] == 'pred' and el[1] == 'count']))
            
            print("             EVAL  ----- Loss: " + "{:.3f}".format(eval_loss) + " - Accuracy: " + "{:.5f}".format(eval_metrics['accuracy'])
                + " - Weighted F1: " + "{:.5f}".format(eval_metrics['weighted_f1']) + " - Macro F1: " + "{:.5f}".format(eval_metrics['macro_f1']))
            print("                 Labels (pred/true) +++++ " + ' - '.join([f"{evaluating_generator.generator.attack_names[int(el[2])]}: {eval_metrics.get('pred_count_'+ el[2],0)}/{eval_metrics.get('true_count_'+ el[2], 0)}" for el in (key.split('_') for key in eval_metrics.keys()) if len(el) > 2 and el[0] == 'pred' and el[1] == 'count']))
            
            self.scheduler.step()

            if not self.sweep:
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
                    }, self.ckpt_path + "/weights." + "{0}-".format(str(epoch).zfill(4)) + "{:.2f}".format(eval_loss) + ".pt")
            
            ca_dic = dict()
            if self.extended_metrics:
                for i, x in enumerate(eval_metrics['class_accuracy']):
                    ca_dic['_acc_'+str(i)] = x
                for i, x in enumerate(eval_metrics['class_precision']):
                    ca_dic['_pre_'+str(i)] = x
                for i, x in enumerate(eval_metrics['class_recall']):
                    ca_dic['_rec_'+str(i)] = x
            if self.use_wandb == True:
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
        
        eval_metrics = GNN.calculateMetrics(eval_trueLabels, eval_predLabels, extended=self.extended_metrics)

        print("             EVAL  ----- Loss: " + "{:.3f}".format(eval_loss) + " - Accuracy: " + "{:.5f}".format(eval_metrics['accuracy'])
            + " - Weighted F1: " + "{:.5f}".format(eval_metrics['weighted_f1']) + " - Macro F1: " + "{:.5f}".format(eval_metrics['macro_f1']))
        print("                 Labels (pred/true) +++++ " + ' - '.join([f"{evaluating_generator.generator.attack_names[int(el[2])]}: {eval_metrics.get('pred_count_'+ el[2],0)}/{eval_metrics.get('true_count_'+ el[2], 0)}" for el in (key.split('_') for key in eval_metrics.keys()) if len(el) > 2 and el[0] == 'pred' and el[1] == 'count']))
        return eval_metrics

    def compile(self, loss, optimizer, scheduler, is_cuda):
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.is_cuda = is_cuda
        return

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

    def _get_compiled_model(hyperparameters):
        gamma = None
        if 'gamma' in hyperparameters.keys():
            gamma = float(hyperparameters['gamma'])

        model = GNN(node_state_dim=int(hyperparameters['node_state_dim']),t=int(hyperparameters['t']), epochs=int(hyperparameters['epochs']), batch_size=int(hyperparameters['batch_size']), 
                    decay_rate=float(hyperparameters['decay_rate']), decay_steps=int(hyperparameters['decay_steps']), gamma=gamma)
        useCuda = model.useCuda()

        if(useCuda):
            model.to(device=model.device)
        model_in_gpu = next(model.parameters()).is_cuda
        if(model_in_gpu):
            print("Model in the GPU.")
        else:
            print("Model not in the GPU.")

        loss = CrossEntropyLoss()
        optimizer = Adam(params=model.parameters() , lr=float(hyperparameters['learning_rate']), eps=1e-07, capturable=model_in_gpu)

        scheduler = ExponentialLR(optimizer, gamma if gamma is not None else 0.9)
        model.compile(loss=loss,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      is_cuda = model_in_gpu)

        return model
    
    def make_or_restore_model(hyperparameters, logs_dir, use_wandb:bool = False, sweep:bool = False, 
                              extended_metrics:bool=False, loadEpoch : int = None, 
                              loadBestEpoch:bool=True, force_cpu:bool=False):
        
        ckpt_path = os.path.abspath(logs_dir)+ "\\ckpt"
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        files = glob.glob(ckpt_path + '/*.pt')
        resume = bool(len(files) > 0) and not sweep
        
        if use_wandb:
            hyperparameters = configWandB(hyperparameters)
            wandb.init(project="TFG", entity="alexcomas", config=hyperparameters, resume=resume)
            hyperparameters = wandb.config
        
        model = GNN._get_compiled_model(hyperparameters)
        
        model.ckpt_path = ckpt_path
        model.extended_metrics = extended_metrics
        model.force_cpu = force_cpu
        model.sweep = sweep
        model.use_wandb = use_wandb

        if(resume):
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
            
            if model.is_cuda:
                device=torch.device('cuda')
            else:
                device=torch.device('cpu')
            
            checkpoint = torch.load(filepath, device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.loadedEpoch = checkpoint['epoch']
            model.startingLoss = checkpoint['loss']
            model.startingEpoch = model.loadedEpoch + 1

            print(f"Loaded epoch {model.loadedEpoch} with loss {model.startingLoss}.")
        else:
            model.startingEpoch = 1
            model.startingLoss = None
        
        for _ in range(model.startingEpoch-1):
            model.scheduler.step()

        return (model, model.startingEpoch)
