from tqdm import tqdm
import pandas as pd
import sys
import os
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from functions import *
import tensorflow as tf 
import numpy as np
import tensorflow_probability as tfp

class Privacy():
    def __init__(self, name, threshold, n_target=None, latent_dim=100):
        self.name_ = name
        self.name_ = self.get_name()
        self.threshold = threshold
        self.n_target = n_target
        
        self.latent_dim = latent_dim

        self.TP = None
        self.FN = None
        self.target_index = None
        self.reconstructed_data = None
        self.min_distances = None
        
        
    def get_name(self):
        return "super"      
    
    def attack(self):
        return    

        
class FBA(Privacy):

    def get_name(self):
        return "FBA_"+str(self.name_)

    def attack(self,generator, data, n_syn):
        self.n_syn = n_syn
        TP=0
        FN=0
        min_distances = []
        reconstructed_data = []
        
        inds=list(range(data.shape[0]))
        np.random.shuffle(inds)
        
        self.target_index = inds[:self.n_target]
        
        for i in tqdm(inds[:self.n_target]):
            target=data.iloc[i].to_numpy().reshape(1,-1)
            target.dtype=np.float64
            
            z=tf.random.uniform((n_syn,self.latent_dim),-1,1)
            min_distance, reconst = get_FBA_result(generator, z,target)
                                    
            reconstructed_data.append(reconst)
            min_distances.append(min_distance)
            
            if min_distance<self.threshold:
                TP+=1
            else:
                FN+=1
        self.TP=TP
        self.FN=FN
        self.min_distances = min_distances
        self.reconstructed_data = reconstructed_data
        # return TP,FN,min_distances
          

    
    
class PBA(Privacy):

    def get_name(self):
        return "PBA_"+str(self.name_)

    def attack(self,generator, data, max_iterations):
        self.max_iterations=max_iterations
        TP=0
        FN=0
        min_distances = []
        reconstructed_data = []
        
        inds=list(range(data.shape[0]))
        np.random.shuffle(inds)
        
        self.target_index = inds[:self.n_target]
        
        for i in tqdm(inds[:self.n_target]):
            target=data.iloc[i].to_numpy().reshape(1,-1)
            target.dtype=np.float64         
        
            initial_z=tf.random.uniform((1,self.latent_dim,),-1,1,dtype=tf.float64)
            result = get_PBA_result(generator, max_iterations, initial_z,target)
            final_z = result.position
            
            reconst = generator(final_z).numpy().reshape(1,-1)
            
            min_distance = result.objective_value
                        
            reconstructed_data.append(reconst)
            min_distances.append(min_distance)
            
            if min_distance<self.threshold:
                TP+=1
            else:
                FN+=1
        self.TP=TP
        self.FN=FN
        self.min_distances = min_distances
        self.reconstructed_data = reconstructed_data
        # return TP,FN,min_distances

        


class WGA(Privacy):

    def get_name(self):
        return "WGA_"+str(self.name_)

    def attack(self,generator, data,max_iterations):
        self.max_iterations= max_iterations
        TP=0
        FN=0
        min_distances = []
        reconstructed_data = []
        
        inds=list(range(data.shape[0]))
        np.random.shuffle(inds)
        
        self.target_index = inds[:self.n_target]
        
        for i in tqdm(inds[:self.n_target]):
            target=data.iloc[i].to_numpy().reshape(1,-1)
            target.dtype=np.float64         

            initial_z=tf.random.uniform((1,self.latent_dim,),-1,1,dtype=tf.float64)
            
            result = get_WGA_result(generator, max_iterations, initial_z, target)
            
            final_z = result.position
            reconst = generator(final_z).numpy().reshape(1,-1)
            
            min_distance = result.objective_value
                        
            reconstructed_data.append(reconst)
            min_distances.append(min_distance)
            
            if min_distance<self.threshold:
                TP+=1
            else:
                FN+=1
        self.TP=TP
        self.FN=FN
        self.min_distances = min_distances
        self.reconstructed_data = reconstructed_data
        # return TP,FN,min_distances

class WDA(Privacy):

    def get_name(self):
        return "WDA_"+str(self.name_)

    def attack(self,discriminator, train_data,ho_data ):
        Top_N = get_WDA_result(discriminator=discriminator,train_data=train_data,ho_data=ho_data, n_data = self.n_target, threshold= self.threshold)
        
        TP = Top_N.loc[Top_N.sort =="Train"].shape[0]
        FN = Top_N.loc[Top_N.sort =="Holdout"].shape[0]
        self.TP=TP
        self.FN=FN
        # return TP,FN