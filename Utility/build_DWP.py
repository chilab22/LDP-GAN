import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from utilites import *
from functions import *
from tensorflow.keras.models import *
import pickle
import argparse    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='') 

    parser.add_argument('--train', default = "train_data.csv")
    parser.add_argument('--ho', default = "ho_data.csv") # 
    parser.add_argument('--latent_dim', default = 100, type=int)
    parser.add_argument('--GPU', default = "0")
    parser.add_argument('--epsilons', nargs='*', type = float , default = [0.1, 0.25, 0.5 ,0.75, 1, 2.5 , 5, 7.5 , 10, 20])
    parser.add_argument('--Models', nargs='*', default = ['LDP' ,'DP'])                 
    
    args = parser.parse_args()

    epsilons = args.epsilons 
    GPU_to_use = args.GPU
    latent_dim = args.latent_dim
    Models = args.Models
    train = args.train 
    ho = args.ho 
    epsilons = args.epsilons
    
    file_dir= os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_dir)
    
    train_data = pd.read_csv('../data/'+train)
    if 'Unnamed: 0' in train_data.columns:
        del train_data['Unnamed: 0']
        
    ho_data = pd.read_csv('../data/'+ho)
    if 'Unnamed: 0' in ho_data.columns:
        del ho_data['Unnamed: 0']
        
    n_data = train_data.shape[0]
    columns = train_data.columns
    
    
    with open('../data/round_dict.pickle', 'rb') as f:
        round_dict = pickle.load(f)
        
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_to_use

    bin_cols = []
    for c in train_data.columns:
        if np.array_equal(train_data[c].unique(), train_data[c].unique().astype(bool)):
            bin_cols.append(c)
            
    generator_dir = "../save/generators/"
  
    epsilons = np.array(epsilons).astype(np.float32)
    
    for m in Models:
        for e in epsilons:
            
            name = m+"_"+str(e)
            generator = load_model(generator_dir+ name + "_generator.h5")
            DWP_model = DWP(name, n_data,columns, bin_cols)
                       
            syn_data =  generate_syn_data(generator, bin_cols, n_data, columns, round_dict,latent_dim) 
            DWP_model.syn_roc, DWP_model.syn_mse = get_DWP(syn_data, ho_data,bin_cols)            
            
            DWP_model.real_roc, DWP_model.real_mse  = get_DWP(train_data, ho_data, bin_cols)

            DWP_model.get_distance()
            with open("utility_models/DWP/{}.pickle".format(DWP_model.name_),"wb") as fw:
                pickle.dump(DWP_model, fw)