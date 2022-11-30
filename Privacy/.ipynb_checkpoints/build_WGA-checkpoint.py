2import matplotlib.pyplot as plt
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
    parser.add_argument('--latent_dim', default = 100, type=int)
    parser.add_argument('--threshold', default = 0.005, type=float)
    parser.add_argument('--n_target', default = 600, type=int)
    parser.add_argument('--attack_on', default = "train_data.csv")
    parser.add_argument('--epsilons', nargs='*', type = float , default = [0.1, 0.25, 0.5 ,0.75, 1, 2.5 , 5, 7.5 , 10, 20])
    parser.add_argument('--Models', nargs='*', default = ['LDP' ,'DP'])                 
    parser.add_argument('--max_iterations', default = 500,type = int)                 

    args = parser.parse_args()

    
    latent_dim=args.latent_dim
    threshold = args.threshold
    n_target = args.n_target
    attack_on = args.attack_on
    GPU_to_use = args.GPU
    epsilons = args.epsilons 
    Models = args.Models
    max_iterations = args.max_iterations
    
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_to_use
    
    file_dir= os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_dir)
    
    data = pd.read_csv('../data/'+attack_on)
    if 'Unnamed: 0' in data.columns:
        del data['Unnamed: 0'] 
        
    columns = data.columns

    
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
            WGA_model = WGA(name, n_target=n_target,threshold=threshold, latent_dim=latent_dim)

            WGA_model.attack(generator=generator, data=data,max_iterations = max_iterations)
            
            with open("privacy_models/WGA/{}.pickle".format(WGA_model.name_),"wb") as fw:
                pickle.dump(WGA_model, fw)

        
    