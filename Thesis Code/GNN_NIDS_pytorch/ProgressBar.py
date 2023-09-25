from __future__ import division
import enum
import math
from tracemalloc import start
from turtle import color
import colorama
import numpy as np
from sqlalchemy import true
import time
from datetime import datetime, date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import json


ACCEPTABLE_PREDICTION_TYPES = ['linear', 'simple', 'polynomial']

class ProgressBar():

    def __init__(self, total, initial = 0, divisions = 200, color = colorama.Fore.YELLOW, unit = "unit", type_prediction='linear', recency = -1, day = None, degree = 2, etaMode = 'exact'):
        if type_prediction not in ACCEPTABLE_PREDICTION_TYPES:
            raise Exception('Not implemented prediction type.')
        self.divisions = min(total, divisions)
        self.total = total
        self.color = color
        self.unit = unit
        self.type_prediction = type_prediction
        if recency == -1:
            recency = self.divisions
        self.recency = recency
        self.day = day

        self.steps = [False for _ in range(divisions+1)]

        self.progress = [None]*(divisions+1)
        self.times = [None]*(divisions+1)
        self.velocity = [None]*(divisions+1)
        self.eta = [None]*(divisions+1)
        self.eta_times = [None]*(divisions+1)
        self.degree = degree
        self.etaMode = etaMode

        percent = 100 * (initial / float(self.total))
        step = int(percent*self.divisions/100)

        self.done = False
        self.current = step
        self.initial_step = step
        self.initial_time = time.time_ns()

        self.steps[step] = True
        self.progress[step] = 0
        self.times[step] = 0
        self.velocity[step] = 0
        self.eta_times[step] = time.time_ns()+30*1e9
        self.eta[step] = datetime.now() + timedelta(minutes=30)

        self.symbol_d = '█'
        self.symbol_n = '-'

    def update(self, progress: int, extraInfo = None):
        if self.done:
            return
        percent = round(100 * (progress / float(self.total)),2)
        step = int(percent*self.divisions/100)
        if(not self.steps[step]):
            self.steps[step] = True
            self.progress[step] = progress
            self.times[step] = time.time_ns() - self.initial_time
            self.velocity[step] = 1e9*(self.progress[step]-self.progress[self.current])/(self.times[step]-self.times[self.current])
            self.eta_times[step] = self.predictETA(step)
            self.eta[step] =  datetime.fromtimestamp((self.eta_times[step] + self.initial_time)/1e9)
            self.current=step
        bar = '█' * int(percent) + '-' * (100 - int(percent))
        if(step != self.initial_step):
            if self.etaMode == 'remaining':
                delta = self.eta[step] - datetime.now()
                minutes = int(delta.total_seconds()/60)
                seconds = int(delta.total_seconds()%60)
                eta = f"{str(minutes).zfill(2)}:{str(seconds).zfill(2)}"
            else:
                eta = self.eta[step].strftime("%H:%M:%S")
            extraInfoString = ""
            if(extraInfo != None):
                for key in extraInfo.keys():
                    value = round(extraInfo[key],2)
                    extraInfoString += f" | {key}: {value}" 
            print(self.color + f"\r|{bar}| {percent:.2f}% | {self.velocity[step]:.2f} {self.unit}/s | {eta}{extraInfoString}    " + f"", end="\r")
        else:
            print(self.color + f"\r|{bar}| {percent:.2f}%", end="\r")

        if percent >= 100:
            self.finish()

    def finish(self):
        self.done = True
        bar = self.symbol_d * 100
        print(colorama.Fore.GREEN + f"\r|{bar}| 100%\033[K", end="\r")
        print(colorama.Fore.RESET + "")
        if self.day != None:
            with open(f"Processed/ProgressBar/{self.day}_progress_bar.json", 'w') as outfile:
                json.dump(self.getData(), outfile)

    def predictETA(self, step):
        if self.type_prediction == 'linear':
            return self.linearETA(step)
        elif self.type_prediction == 'simple':
            return self.simpleETA(step)
        elif self.type_prediction == 'polynomial':
            return self.polynomialETA(step)
        else:
            return self.linearETA(step)
        
    def linearETA(self, step):
        indexes = [i for i, el in enumerate(self.steps) if el and i != self.initial_step and i > step-self.recency]
        y = np.array([self.times[i] for i in indexes])
        x = np.array([self.progress[i] for i in indexes]).reshape((-1,1))
        eta_model = LinearRegression().fit(x,y)
        return eta_model.predict(np.array([self.total]).reshape((-1,1)))[0]
    
    def polynomialETA(self, step):
        indexes = [i for i, el in enumerate(self.steps) if el and i != self.initial_step and i > step-self.recency]
        y = np.array([self.times[i] for i in indexes])
        prevx = np.array([self.progress[i] for i in indexes]).reshape((-1,1))

        x = PolynomialFeatures(degree=self.degree, include_bias=False).fit_transform(prevx)
        eta_model = LinearRegression().fit(x,y)

        predx = PolynomialFeatures(degree=self.degree, include_bias=False).fit_transform(np.array([self.total]).reshape((-1,1)))
        return eta_model.predict(predx)[0]

    def simpleETA(self, step):
        seconds_left = (self.total-self.progress[step])/self.velocity[step]
        return self.times[step] + seconds_left*1e9
    
    def getData(self):
        return {
            'steps': self.steps,
            'progress': self.progress,
            'times': self.times,
            'velocity': self.velocity,
            'eta_times': self.eta_times,
            'eta': self.times,
            'unit': self.unit
        }
        