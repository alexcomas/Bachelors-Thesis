import generator
import csv
import glob
import pandas as pd
import os
from random import random

def sortByTimestamp(inpath, outpath = None):
    files = glob.glob(inpath + '/*.csv')
    sorted = True
    count = 0

    for file in files:
        with open(file, encoding="utf8", errors='ignore') as csvfile:
            data = csv.reader(csvfile, delimiter=',', quotechar='|')
            
            rowList = []
            lastTime = None
            for row in data:
                count+=1
                error = False
                try:
                    thisTime = pd.to_datetime(row[generator.features_dict['Timestamp']], format='%d/%m/%Y %H:%M')
                except:
                    try:
                        thisTime = pd.to_datetime(row[generator.features_dict['Timestamp']], format='%d/%m/%Y %H:%M:%S')
                    except:
                        # print("Error converting: " + row[generator.features_dict['Timestamp']])
                        error = True
                if(lastTime != None and lastTime > thisTime):
                    # print("Not sorted: " + str(lastTime) + " < " + str(thisTime))
                    sorted = False
                if(not error):
                    row[generator.features_dict['Timestamp']] = thisTime
                    rowList.append(row)

                lastTime = thisTime

    df = pd.DataFrame(data=rowList, columns=generator.features)
    df.sort_values(by=['Timestamp'], inplace=True, ascending=True)
    if(outpath != None): 
        df.to_csv(outpath + 'out.csv', sep=',', header=False, index=False)
    
    return df

def checkSorted(df):
    sortedAscending = True
    sortedDescending = True
    lastTime = None
    for i, row in df.iterrows():
        error = False
        try:
            thisTime = pd.to_datetime(row[generator.features_dict['Timestamp']], format='%d/%m/%Y %H:%M')
        except:
            try:
                thisTime = pd.to_datetime(row[generator.features_dict['Timestamp']], format='%d/%m/%Y %H:%M:%S')
            except:
                print("Error converting: " + row[generator.features_dict['Timestamp']])
                error = True
        if(lastTime != None and lastTime > thisTime):
            # print("Not sorted: " + str(lastTime) + " < " + str(thisTime))
            sortedAscending = False
        if(lastTime != None and lastTime < thisTime):
            # print("Not sorted: " + str(lastTime) + " < " + str(thisTime))
            sortedDescending = False

        lastTime = thisTime

    return (sortedAscending, sortedDescending)


def shortenDataset(inpath, outpath = None, fraction = 0.5):
    files = glob.glob(inpath + '/*.csv')
    sorted = True
    count = 0

    for file in files:
        with open(file, encoding="utf8", errors='ignore') as csvfile:
            data = csv.reader(csvfile, delimiter=',', quotechar='|')
            
            rowList = []
            lastTime = None
            for row in data:
                count+=1
                error = False
                try:
                    thisTime = pd.to_datetime(row[generator.features_dict['Timestamp']], format='%d/%m/%Y %H:%M')
                except:
                    try:
                        thisTime = pd.to_datetime(row[generator.features_dict['Timestamp']], format='%d/%m/%Y %H:%M:%S')
                    except:
                        # print("Error converting: " + row[generator.features_dict['Timestamp']])
                        error = True
                if(not error and random() <  fraction):
                    row[generator.features_dict['Timestamp']] = thisTime
                    rowList.append(row)

    df = pd.DataFrame(data=rowList, columns=generator.features)
    if(outpath != None): 
        df.to_csv(outpath, sep=',', header=False, index=False)
    
    return df