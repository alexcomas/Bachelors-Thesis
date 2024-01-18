import tensorflow as tf
import tensorflow_addons as tfa
import sys
import csv
import glob
import os
import os.path
from ..generator import Generator
from ..IDS_Dataset_base import IDS_Dataset_base

class IDS_Dataset(IDS_Dataset_base):
    def __init__(self, dataset_path: str, dataset:str='IDS2017', dataset_labels:str=None, window:int=200, data_treatment:str='none', data_treatment_params_path:str=None) -> None:
        super().__init__(dataset_path=dataset_path, dataset=dataset, dataset_labels=dataset_labels, for_framework='tensorflow',
                                    window=window, data_treatment=data_treatment, data_treatment_params_path=data_treatment_params_path)

    def getLoader(self, loader_params):
        ds = tf.data.Dataset.from_generator(self.__iter__,
                                            args=[],
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

        if not bool(loader_params['validation']):
            ds = ds.repeat()
        
        return ds
