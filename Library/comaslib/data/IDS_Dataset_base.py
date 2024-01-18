import csv
import glob
import os.path
from .generator import Generator

class IDS_Dataset_base:
    def __init__(self, dataset_path: str, dataset:str='IDS2017', dataset_labels:str=None, for_framework:str='tensorflow', window:int=200, 
                 data_treatment:str='none', data_treatment_params_path:str=None) -> None:
        if not os.path.exists(dataset_path):
            raise Exception(f"File doesn't exist: {dataset_path}")
        if os.path.isfile(dataset_path) and not dataset_path.endswith('.csv'):
            raise Exception(f"File doesn't have a valid extension: {dataset_path}")
        self.generator = Generator(dataset=dataset, dataset_labels=dataset_labels, for_framework=for_framework, data_treatment=data_treatment, data_treatment_params_path=data_treatment_params_path)
        self.dataset_path = dataset_path
        # self.dataset_path = self.dataset_path.decode('utf-8')
        self.window = window

    def __iter__(self):
        return self.generate()

    def generate(self):
        n_graphs = 0
        total_counter = 0
        if os.path.isdir(self.dataset_path):
            files = glob.glob(self.dataset_path + '/*.csv')
        else:
            files = [glob.glob(self.dataset_path)]
        first_file_iteration_done = {f: False for f in files}
        for file in files:
            if not first_file_iteration_done[file]:
                print(f"\nOpening file in generator: {file}")
                print("Chosen features: ", ', '.join(Generator.CHOSEN_CONNECTION_FEATURES))
                print("Data treatment: ", self.generator.data_treatment)
                first_file_iteration_done[file] = True
            with open(file, encoding="utf8", errors='ignore') as csvfile:
                data = csv.reader(csvfile, delimiter=',', quotechar='|')

                current_time_traces = []
                counter = 0
                for row in data:
                    if len(row) > 1:
                        current_time_traces.append(row)
                        counter += 1
                        # remains to fix this criterion (for now we set the windows to be 200 connections big)
                        if counter >= self.window:
                            G = self.generator.traces_to_graph(current_time_traces)
                            features, label = self.generator.graph_to_dict(G)
                            n_graphs += 1
                            # We do not need to do the undersampling here, since it was done during the preprocessing
                            yield (features, label)
                            total_counter += counter
                            counter = 0
                            current_time_traces = []

    def getLoader(self):
        print("Method implemented in child classes")
        return