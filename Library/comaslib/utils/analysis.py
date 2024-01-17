from ..data.generator import Generator

def niceResults(result, generator: Generator, dictKeys = None):
    if dictKeys is None:
        dictKeys = ['loss', 'categorical_accuracy', 'specificity_at_sensitivity', 'rec_0', 'pre_0', 'rec_1', 'pre_1', 'rec_2', 'pre_2', 'rec_3', 
                    'pre_3', 'rec_4', 'pre_4', 'rec_5', 'pre_5', 'rec_6', 'pre_6', 'rec_7', 'pre_7', 'rec_8', 'pre_8', 'rec_9', 'pre_9', 'rec_10', 
                    'pre_10', 'rec_11', 'pre_11', 'rec_12', 'pre_12', 'rec_13', 'pre_13', 'rec_14', 'pre_14', 'macro_F1', 'weighted_F1']
    res = dict(zip(dictKeys, result))
    metrics = dict()
    for key, value in res.items():
        if key.startswith('pre'):
            i = int(key.split('_')[1])
            metrics['Pre_' + generator.attack_names[i]] = value
        elif key.startswith('rec'):
            i = int(key.split('_')[1])
            metrics['Rec_' + generator.attack_names[i]] = value
        else:
            metrics[key] = value
    return metrics