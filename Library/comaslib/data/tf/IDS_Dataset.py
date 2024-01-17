import tensorflow as tf
import tensorflow_addons as tfa
import sys
import csv
import glob
import os
import os.path
from ..generator import Generator

class IDS_Dataset:
    def __init__(self, dataset_path: str, dataset='IDS2017', dataset_labels=None, for_framework='tensorflow', data_treatment='none', data_treatment_params_path=None) -> None:
        if not os.path.exists(dataset_path):
            raise Exception(f"File doesn't exist: {dataset_path}")
        if os.path.isfile(dataset_path) and not dataset_path.endswith('.csv'):
            raise Exception(f"File doesn't have a valid extension: {dataset_path}")
        self.generator = Generator(dataset=dataset, dataset_labels=dataset_labels, for_framework=for_framework, data_treatment=data_treatment, data_treatment_params_path=data_treatment_params_path)
        self.dataset_path = dataset_path
        # self.dataset_path = self.dataset_path.decode('utf-8')

    def generate(self, window: int):
        n_graphs = 0
        total_counter = 0
        if os.path.isdir(self.dataset_path):
            files = glob.glob(self.dataset_path + '/*.csv')
        else:
            files = [glob.glob(self.dataset_path)]
        for file in files:
            print(f"\nOpening file in generator: {file}")
            print("Chosen features: ", ', '.join(Generator.CHOSEN_CONNECTION_FEATURES))
            print("Data treatment: ", self.generator.data_treatment)
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
                            G = self.generator.traces_to_graph(current_time_traces)
                            features, label = self.generator.graph_to_dict(G)
                            n_graphs += 1
                            # We do not need to do the undersampling here, since it was done during the preprocessing
                            yield (features, label)
                            total_counter += counter
                            counter = 0
                            current_time_traces = []

    def input_fn(self, window:int = 200, validation=False):
        ds = tf.data.Dataset.from_generator(self.generate,
                                            args=[window],
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
