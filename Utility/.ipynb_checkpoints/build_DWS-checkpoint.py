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

    parser.add_argument('--data', default = "train_data.csv") # 
    parser.add_argument('--latent_dim', default = 100, type=int)
    parser.add_argument('--GPU', default = "0")
    parser.add_argument('--epsilons', nargs='*', type = float , default = [0.1, 0.25, 0.5 ,0.75, 1, 2.5 , 5, 7.5 , 10, 20])
    parser.add_argument('--Models', nargs='*', default = ['LDP' ,'DP'])                 
    
    args = parser.parse_args()

    epsilons = args.epsilons 
    GPU_to_use = args.GPU
    latent_dim = args.latent_dim
    Models = args.Models
    data = args.data 
    epsilons = args.epsilons
    
    file_dir= os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_dir)   
    
    data = pd.read_csv('../data/'+data)
    if 'Unnamed: 0' in data.columns:
        del data['Unnamed: 0']
    n_data = data.shape[0]
    columns = data.columns
    
    with open('../data/round_dict.pickle', 'rb') as f:
        round_dict = pickle.load(f)
        
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_to_use

    bin_cols = []
    for c in data.columns:
        if np.array_equal(data[c].unique(), data[c].unique().astype(bool)):
            bin_cols.append(c)
            
    generator_dir = "../save/generators/"
  
    epsilons = np.array(epsilons).astype(np.float32)
    
    for m in Models:
        for e in epsilons:
            
            name = m+"_"+str(e)
            generator = load_model(generator_dir+ name + "_generator.h5")
            DWS_model = DWS(name, n_data,columns, bin_cols)
            DWS_model.real_score = get_DWS(data, bin_cols)

            syn_data =  generate_syn_data(generator, bin_cols, n_data, columns, round_dict, latent_dim) 

            DWS_model.syn_score = get_DWS(syn_data, bin_cols)
            DWS_model.get_distance(DWS_model.real_score, DWS_model.syn_score)
            with open("utility_models/DWS/{}.pickle".format(DWS_model.name_),"wb") as fw:
                pickle.dump(DWS_model, fw)

        
    