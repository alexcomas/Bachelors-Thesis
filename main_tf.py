import configparser
import os
import tensorflow as tf
import configparser
import warnings
warnings.filterwarnings('ignore')

import Library.comaslib as comaslib

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('./config.ini')

directory = 'DIRECTORIES_' + params['RUN_CONFIG']['dataset_data'] + '_LABELS_' + params['RUN_CONFIG']['dataset_labels']

generator = comaslib.data.Generator(dataset='IDS2017')
path_logs = os.path.abspath(params[directory]['logs'])
(model, startingEpoch) = comaslib.model.tf.GNN.make_or_restore_model(hyperparameters=params['HYPERPARAMETERS'], logs_dir=params[directory]['logs'])

# callbacks to save the model
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=  path_logs + "/ckpt/weights.{epoch:02d}-{loss:.3f}.tf", save_freq='epoch', 
                    monitor='val_loss', save_best_only=False, save_format="tf"), 
              tf.keras.callbacks.TensorBoard(log_dir=path_logs + "/logs", update_freq=1000, profile_batch = '100,120')]

train_dataset = comaslib.data.tf.IDS_Dataset(
    dataset_path=os.path.abspath(params[directory]["train"]),
    dataset=params['RUN_CONFIG']['dataset_data'],
    dataset_labels=params['RUN_CONFIG']['dataset_labels'],
    window = int(params['RUN_CONFIG']['window']),
    data_treatment='none',
    data_treatment_params_path=os.path.abspath(params[directory]["data_treatment"])
    )
val_dataset = comaslib.data.tf.IDS_Dataset(
    dataset_path=os.path.abspath(params[directory]["validation"]),
    dataset=params['RUN_CONFIG']['dataset_data'],
    dataset_labels=params['RUN_CONFIG']['dataset_labels'],
    window = int(params['RUN_CONFIG']['window']),
    data_treatment=params['RUN_CONFIG']['data_treatment'],
    data_treatment_params_path=os.path.abspath(params[directory]["data_treatment"])
    )
model.fit(train_dataset.getLoader({'validation': False}),
          validation_data= val_dataset.getLoader({'validation': True}),
        #   validation_steps = 249,
          steps_per_epoch = 1600,
          batch_size=1,
          epochs=2000,
          callbacks=callbacks,
          use_multiprocessing=True,
          initial_epoch=startingEpoch)