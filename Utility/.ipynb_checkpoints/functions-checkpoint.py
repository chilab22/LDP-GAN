#### utility #########
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score,f1_score



def generate_syn_data(generator, bin_cols, n_data, columns, round_dict, latent_dim):
    z=tf.random.uniform((n_data,latent_dim),-1,1)
    syn=generator(z).numpy().reshape(n_data,-1)
    syn=pd.DataFrame(syn,columns=columns)
    for c in columns:
        if c in bin_cols:
            syn[c]=syn[c].round(0).astype('int')
        else:
            syn[c]=syn[c].round(round_dict[c])
    return syn


def get_DWS(data,bin_cols):
    DWS = []
    for c in bin_cols:
        DWS.append(data[c].mean())
    return DWS


def get_DWA(data,bin_cols):
    DWA = []
    ## 정확히는 c in count_cols:
    for c in data.columns:
        if c not in bin_cols:
            DWA.append(data[c].mean())
    return DWA               
    
def get_DWP(data, ho_data, bin_cols):
    mse = []
    rocauc = []
    for c in data.columns:
        if ho_data[c].unique().shape[0]==1:
            continue
        if c in bin_cols:
            RF=RandomForestClassifier(n_jobs=64)
            RF.fit(data.drop([c],axis=1),data[c])
            pred=RF.predict_proba(ho_data.drop([c],axis=1))
            
            if pred.shape[1]==1: # 한종류의 클래스만 생성되는경우
                score=roc_auc_score(ho_data[c].to_numpy(),pred[:,0])
            else:
                score=roc_auc_score(ho_data[c].to_numpy(),pred[:,1])
            rocauc.append(score)
        else:
            RF=RandomForestRegressor(n_jobs=64)
            RF.fit(data.drop([c],axis=1),data[c])
            pred=RF.predict(ho_data.drop([c],axis=1))
            score=mean_squared_error(ho_data[c],pred)                                                              
            mse.append(score)
    return rocauc, mse

def get_Corr(data):
    return data.corr()