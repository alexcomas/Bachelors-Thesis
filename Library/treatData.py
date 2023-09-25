import sys
import pandas as pd
import comaslib.data.generator as generator
from comaslib.data.processing import treatDataset, convertDate
from datetime import datetime
from pathlib import Path
import os

in_features = ['Flow ID','Source IP','Source Port','Destination IP','Destination Port','Protocol','Timestamp','Flow Duration','Total Fwd Packets',
            'Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets','Fwd Packet Length Max','Fwd Packet Length Min',
            'Fwd Packet Length Mean','Fwd Packet Length Std','Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std',
            'Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std',
            'Fwd IAT Max','Fwd IAT Min','Bwd IAT Total','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags',
            'Fwd URG Flags','Bwd URG Flags','Fwd Header Length','Bwd Header Length','Fwd Packets/s','Bwd Packets/s','Min Packet Length',
            'Max Packet Length','Packet Length Mean','Packet Length Std','Packet Length Variance','FIN Flag Count','SYN Flag Count','RST Flag Count',
            'PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count','Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size',
            'Avg Bwd Segment Size','Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk','Fwd Avg Bulk Rate','Bwd Avg Bytes/Bulk','Bwd Avg Packets/Bulk',
            'Bwd Avg Bulk Rate','Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes','Init_Win_bytes_forward',
            'Init_Win_bytes_backward',
            # 
            'TO ELIMINATE', 
            # 
            'act_data_pkt_fwd','min_seg_size_forward','Active Mean','Active Std','Active Max','Active Min','Idle Mean',
            'Idle Std','Idle Max','Idle Min','Label']

def main():
    """mode == '0' takes SHUFFLED, mode == '1' takes original dataset. Example: 
        'D:\TFG\datasets\IDS2018\preprocessed_IDS2017\EVAL\SEPARATED\DDoS-Slowhttptest normalization 1'"""
    mode = '0'
    inner_folder = 'SHUFFLED/'
    if len(sys.argv) < 3:
        print("Missing data treatment argument.")
        return
    if len(sys.argv) > 3:
        mode = sys.argv[3]
        inner_folder = ''

    # dataset = sys.argv[1]
    dataset_path = sys.argv[1]
    treatment = sys.argv[2]
    if treatment not in ['standardization', 'normalization', 'special_normalization', 'robust_normalization']:
        print("Invalid treatment.")
        return
    features = generator.features
    if mode == '1':
        features = in_features
    dataset_path = os.path.abspath(dataset_path)
    df = pd.read_csv(f'{dataset_path}\\dataset.csv', header=None, names=features, low_memory=False)
    
    df = df[generator.features]
    
    if mode == '1':
        df = df[[el.strip() != 'Timestamp' for el in df['Timestamp']]]
        df['Timestamp'] = [convertDate(row['Timestamp']) for _, row in df.iterrows()]
        df['Timestamp'] = [el.strftime("%d/%m/%Y %I:%M:%S %p") for el in df['Timestamp']]

    print(f"Data treatment: {treatment}")
    out_path = Path(f'{dataset_path}\\{treatment}')
    out_path.mkdir(parents=True, exist_ok=True)
    df_treated = treatDataset(df, treatment=treatment, dataset='IDS2017')
    df_treated.to_csv(f'{out_path}\\dataset.csv', header=False, index=False)

if __name__ == '__main__':
    sys.exit(main())