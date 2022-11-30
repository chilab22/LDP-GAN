import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import pickle
import argparse 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='') 

    parser.add_argument('--epsilons', nargs='*', type = float , default = [0.1, 0.25, 0.5 ,0.75, 1, 2.5 , 5, 7.5 , 10, 20])
    parser.add_argument('--Models', nargs='*', default = ['LDP' ,'DP'])                 
    parser.add_argument('--fig_size',  default = 5, type = int)                 
    parser.add_argument('--facecolor',  default = '#eaeaf2')
    parser.add_argument('--color_DP',  default = '#fd625e')
    parser.add_argument('--color_LDP',  default = '#01b8aa')
    parser.add_argument('--GPU', default = "0")
    parser.add_argument('--model_color',  nargs='*',default = ['blue', 'red'])
    parser.add_argument('--utility_models',  nargs='*',default = ['DWS', 'DWA',"DWP", "Corr"])
    parser.add_argument('--attack_models',  nargs='*',default = ['FBA', 'PBA', 'WGA','WDA'])
    parser.add_argument('--markers',  nargs='*',default = ['o','^'])

    
    args = parser.parse_args()    

    epsilons = args.epsilons 
    fig_size = args.fig_size
    Models = args.Models
    facecolor = args.facecolor
    color_DP = args.color_DP
    color_LDP = args.color_LDP
    GPU_to_use = args.GPU
    model_color = args.model_color
    utility_models = args.utility_models                   
    attack_models = args.attack_models
    markers = args.markers
    n_row = len(attack_models)
    n_col = len(utility_models)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_to_use

    epsilons = np.sort(epsilons).astype(np.float32)
    
    file_dir= os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_dir)
    
    sys.path.append(r'../Utility/')
    utilites = {}    
    for m in Models:
        utilites[m] = {}
        for u in utility_models:
            utilites[m][u] = []
            for e in epsilons:
                with open("../Utility/utility_models/{}/{}.pickle".format(u, u+"_"+ m+"_"+str(e)),"rb") as fr:
                    util_model = pickle.load(fr)
                utilites[m][u].append(1/util_model.distance)
    
    
    sys.path.append(r'../Privacy/')                
    privacy = {}
    for c, m in zip(model_color,Models):
        privacy[m] = {}
        for a in attack_models:
            privacy[m][a] = []
            for e in epsilons:
                with open("../Privacy/privacy_models/{}/{}.pickle".format(a, a+"_"+ m+"_"+str(e)),"rb") as fr:
                    priv_model = pickle.load(fr)
                
                privacy[m][a].append( priv_model.FN )  
    
    
    fig,axes = plt.subplots(n_row,n_col,figsize=(fig_size,fig_size))
    for c, m, k in zip(model_color,Models,markers):
        for i,u in enumerate(utility_models):
            for j, a in enumerate(attack_models):            
                axes[i][j].plot(utilites[m][u], privacy[m][a] , color = c, marker = k,linewidth=fig_size/3, markersize=fig_size/3) 
                
                axes[i][j].set_xticklabels([])
                axes[i][j].set_yticklabels([])
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
                
                if j==0:
                    axes[i][j].set_ylabel(attack_models[i],color='black',labelpad=1.5, size=fig_size,weight='bold')
                if i==0:              
                    axes[i][j].set_title(utility_models[j],color='black', size=fig_size, pad=1,weight='bold') 
                    
    fig.savefig('figure/trade_off.png',dpi=300,bbox_inches='tight',pad_inches = 0.1)