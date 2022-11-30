from tqdm import tqdm
import pandas as pd
import sys
import os
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from functions import *
import tensorflow as tf 
import numpy as np
class Utilities():
    def __init__(self, name, n_data,columns, bin_cols):
        self.n_data = n_data
        self.columns=columns
        self.bin_cols = bin_cols
        self.syn_score = None
        self.real_score = None 
        self.name_ = name
        self.name_ = self.get_name()
        
        
    def get_name(self):
        return "super"      
    
    def get_score(self, data):
        scores = []
        for c in data.columns:
            scores.append(data[c].mean())
        return scores
    
    def get_distance(self,real_score,syn_score):
        distance = mean_absolute_error(real_score,syn_score)
        self.distance = distance
        return distance
    
    

class DWS(Utilities):

    def get_name(self):
        return "DWS_"+str(self.name_)

    def get_score(self, data):
        scores = []
        for c in self.bin_cols:
            scores.append(data[c].mean())
        return scores
    
class DWA(Utilities):
    
    def get_name(self):
        return "DWA_" + str(self.name_)

    def get_score(self, data):
        scores = []
        for c in data.columns:
            if not np.array_equal(data[c].unique(), data[c].unique().astype(bool)):
                scores.append(data[c].mean())
        return scores
    
    
    
class DWP(Utilities):
    
    def __init__(self, name, n_data,columns, bin_cols, ):
        self.n_data = n_data
        self.columns=columns
        self.bin_cols = bin_cols
        self.syn_mse = None
        self.syn_roc = None
        self.real_mse = None
        self.real_roc = None
        self.name_ = name
        self.name_ = self.get_name()
        
    def get_name(self):
        return "DWP_" + str(self.name_)

    def get_score(self, train_data, ho_data):
        mse = []
        rocauc = []
        for c in data.columns:
            if c in self.bin_cols:
                RF=RandomForestClassifier(n_jobs=64)
                RF.fit(train_data.drop([c],axis=1),data[c])
                pred=RF.predict_proba(ho_data.drop([c],axis=1))
        
                if pred.shape[1]==1: # 한종류의 클래스만 생성되는경우
                    score=roc_auc_score(ho_data[c].to_numpy(),pred[:,0])
                else:
                    score=roc_auc_score(ho_data[c].to_numpy(),pred[:,1])
                rocauc.append(score)
            else:
                RF=RandomForestRegressor(n_jobs=50)
                RF.fit(data.drop([c],axis=1),data_[c])
                pred=RF.predict(ho_data.drop([c],axis=1))
                score=mean_squared_error(ho_data[c],pred)                                                              
                mse.append(score)
        return rocauc, mse
    
    
    def get_distance(self):
        real_score = self.real_mse+self.real_roc
        syn_score = self.syn_mse+self.syn_roc
        distance = mean_absolute_error(real_score,syn_score)
        self.distance = distance 
        return distance
    
class Corr(Utilities):
    def get_name(self):
        return "Corr_"+str(self.name_)

    def get_score(self,data):
        return data.corr()
    
    def get_distance(self,real_corr,syn_corr):
        distance = np.abs(real_corr-syn_corr).mean().mean()
        self.distance = distance 
        return distance