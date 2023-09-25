import pandas as pd
from tracemalloc import start
from turtle import color
import numpy as np
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
import random
from imblearn.under_sampling import RandomUnderSampler
from ...utils.ProgressBar import ProgressBar
from typing import Callable

def convertDate(dateStr):
    try:
        ret = datetime.strptime(dateStr, "%d/%m/%Y %H:%M")
    except:
        ret = datetime.strptime(dateStr, "%d/%m/%Y %H:%M:%S")
    return ret

def csvReport(data, length = None, dataType='csv', percent = 1):
    attacks = dict()
    counter = 0
    total = 0
    if length != None:
        progress_bar = ProgressBar(length, unit="r", type_prediction='linear', etaMode='remaining')
    for row in data:
        counter+=1
        if random.random() > percent:
            continue
        total += 1
        if(dataType == 'iterrows'):
            row = row[1]
        if row[-1] == 'Label':
            continue
        label = row[-1]
        if not label in attacks.keys():
            attacks[label] = 0

        attacks[label]+=1
        if length != None:
            progress_bar.update(counter)
            
    print("     Number of examples in classes: ", attacks)
    print("     %% of examples in classes: ",[(x,y) for x,y in  zip(attacks.keys(), [np.round(el/total*100,2) for el in attacks.values()])])
    return counter

def makeCountGraphic(df: pd.DataFrame, targetCol, groupByCol, filterList = None, figsize = (13, 4)):
    temp = df[targetCol]
    if(filter != None):
        temp = temp[filterList]
    
    temp = temp.groupby(df[groupByCol]).count()
    fig, axs = plt.subplots(figsize=figsize)
    temp.plot(
        kind='bar', rot=0, ax=axs
    )

def printDistribution(df: pd.DataFrame, column, outdir = None):
    if (outdir != None):
        Path(outdir).mkdir(parents=True, exist_ok=True)

    values = df[column].value_counts().astype(float)
    for i, v in values.items():
        values[i] = 100*float(values[i])/float(len(df))

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        if outdir != None:
            values.to_csv(outdir)
        else:
            print(values)
    return values

def suffleDatasetByWindow(df: pd.DataFrame, window, features = None):
    if features == None:
        features = df.columns
    groups = []
    for i in range(len(df)):
        groups.append(int(i / window))
    df["Group"] = groups
    groups = [df2 for _, df2 in df.groupby('Group') if len(df2) == window]
    random.shuffle(groups)
    df_shuffled = pd.concat(groups).reset_index(drop=True)
    df_shuffled = df_shuffled[features]
    return df_shuffled

def sortDataset(df: pd.DataFrame, column = 'Timestamp', ascending=True, inplace=False, datetimeformat = "%d/%m/%Y %I:%M:%S %p"):
    if not inplace:
        df = df.copy()
    
    df[column] = [datetime.strptime(el, datetimeformat) for el in df[column]]
    df.sort_values(by=[column], inplace=True, ascending=ascending, ignore_index=True)
    df[column] = [el.strftime(datetimeformat) for el in df[column]]
    return df

def balanceDatasetByLabel(df: pd.DataFrame, value='BENIGN', p = 0.41520249, undersampler=RandomUnderSampler):
    n_benign = round(sum(df['Label'] != value)*p/(1-p))
    sampling_strategy = {value: n_benign}
    undersample = undersampler(sampling_strategy=sampling_strategy)
    X_cols = df.columns.copy().to_list()
    X_cols.remove('Label')
    X_under, y_under = undersample.fit_resample(df[X_cols], df['Label'])
    return pd.concat([X_under, y_under], axis=1)

def filterDatasetByAtttacks(df: pd.DataFrame, attacks_to_filter):
    return df[[el in attacks_to_filter for el in df['Label']]]

def separateAttacks(df: pd.DataFrame, attacks_to_filter, p_benign = 0.5, window = 200):
    print("Sorting dataset...")
    df = sortDataset(df)
    print("Done.")
    print("Filtering attacks...")
    df = filterDatasetByAtttacks(df, attacks_to_filter)
    print("Done.")
    print("Balancing dataset...")
    df = balanceDatasetByLabel(df, value='BENIGN', p = p_benign)
    print("Done.")
    print("Shuffling dataset...")
    df = suffleDatasetByWindow(df, window=window)
    print("Done.")
    return df

def formatDataset(df: pd.DataFrame, get_feature_func: Callable, keep_track = False, features = None):
    if features == None:
        features = df.columns
    if keep_track:
        progress_bar = ProgressBar(len(features), initial=0, unit="cols", etaMode='remaining')
    for i,col in enumerate(features):
        # print(f"Starting column: {col}")
        df[col] = [get_feature_func(row, col, parse=True, parseOnlyFloats=True) for _,row in df.iterrows()]
        if keep_track:
            progress_bar.update(i+1, {'col': col})
    return df

def removeNotValidValues(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

def dataTreatment(df: pd.DataFrame, treat_feature_func: Callable, features = None, treatment='none', dataset=None, keep_track = False):
    if features == None:
        features = df.columns
    if keep_track:
        progress_bar = ProgressBar(len(features), initial=0, unit="cols", etaMode='remaining')
    for i,col in enumerate(features):
        df[col] = treat_feature_func(df[col], col, treatment, dataset=dataset)
        if keep_track:
            progress_bar.update(i+1, {'col': col})
    return df

def treatDataset(df: pd.DataFrame, get_feature_func: Callable, treat_feature_func: Callable, treatment='none', dataset=None, window = 200):
    print("Sorting dataset...")
    df = sortDataset(df)
    print("Done.")
    print("Formatting dataset...")
    df = formatDataset(df, get_feature_func, keep_track=True)
    print("Done.")
    print("Removing invalid (NA/+-inf) values in the dataset...")
    df = removeNotValidValues(df)
    print("Done.")
    print("Treating data...")
    df = dataTreatment(df, treat_feature_func, treatment, dataset=dataset, keep_track=True)
    print("Done.")
    print("Shuffling dataset...")
    df = suffleDatasetByWindow(df, window=window)
    print("Done.")
    return df