import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from privacy import *
from functions import *
from tensorflow.keras.models import *
import pickle
import argparse 

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='') 

    parser.add_argument('--GPU', default = "0")
    parser.add_argument('--threshold', default = 300, type=float)
    parser.add_argument('--n_target', default = 300, type=int)
    parser.add_argument('--epsilons', nargs='*', type = float , default = [0.1, 0.25, 0.5 ,0.75, 1, 2.5 , 5, 7.5 , 10, 20])
    parser.add_argument('--Models', nargs='*', default = ['LDP' ,'DP'])                 
    parser.add_argument('--train', default = "train_data.csv")
    parser.add_argument('--ho', default = "ho_data.csv")
    
    args = parser.parse_args()
    
    GPU_to_use = args.GPU        
    threshold = args.threshold
    n_target = args.n_target
    epsilons = args.epsilons 
    Models = args.Models
    train = args.train
    ho = args.ho
    
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_to_use
    
    file_dir= os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_dir)
    
    train_data = pd.read_csv('../data/'+train)
    if 'Unnamed: 0' in train_data.columns:
        del train_data['Unnamed: 0']
        
        
    ho_data = pd.read_csv('../data/'+ho)
    if 'Unnamed: 0' in ho_data.columns:
        del ho_data['Unnamed: 0']
    
    
    columns = train_data.columns

    bin_cols = []
    for c in train_data.columns:
        if np.array_equal(train_data[c].unique(), train_data[c].unique().astype(bool)):
            bin_cols.append(c)
            
    discriminator_dir = "../save/critics/"
    
    epsilons = np.array(epsilons).astype(np.float32)
    
    for m in Models:
        for e in epsilons:
            
            name = m+"_"+str(e)
            discriminator = load_model(discriminator_dir+ name + "_critics.h5")
            WDA_model = WDA(name, n_target=n_target, threshold=threshold)
            WDA_model.attack(discriminator, train_data,ho_data)
            
            with open("privacy_models/WDA/{}.pickle".format(WDA_model.name_),"wb") as fw:
                pickle.dump(WDA_model, fw)