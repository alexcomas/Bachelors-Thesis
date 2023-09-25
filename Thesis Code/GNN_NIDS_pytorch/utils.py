import numpy as np
from generator import attack_names

intParams = ['node_state_dim', 't', 'readout_units', 'decay_steps', 
                'window', 'batch_size', 'epochs', 'classes']
floatParams = ['decay_rate', 'learning_rate']

def datasetReport(gen, n_classes):
    counter = []
    counter = [0 for _ in range(15)] 
    for _, labels in gen:
        for label in labels:
            i = np.argmax(label)
            counter[i] += 1
            
    length = np.sum(counter)
    print("     Number of examples in classes: ", [(x,y) for x,y in zip(attack_names, counter)])
    print("     %% of examples in classes: ",[(x,y) for x,y in  zip(attack_names, [np.round(el/length*100,2) for el in counter])])

def configWandB(params):
    result = dict()

    for var in intParams:
        result[var] = int(params[var])

    for var in floatParams:
        result[var] = float(params[var])

    return result