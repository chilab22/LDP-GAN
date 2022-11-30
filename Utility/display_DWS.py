import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pickle
import math
import argparse 

def draw_DWS_scatter(DWS_model, i,j, m,e):
    real_score = DWS_model.real_score
    syn_score = DWS_model.syn_score
    distance = DWS_model.distance
    
    axes[i][j].plot([0,1],[0,1])
    axes[i][j].scatter(real_score,syn_score)
    
    if i==0 and j==0:
        axes[i][j].set_xlabel(xlabel=m,rotation=0,size=2*fig_size)
        axes[i][j].xaxis.set_label_coords(0.5,1.1)    
    if i==0 and j==1:
        axes[i][j].set_xlabel(xlabel=m,rotation=0,size=2*fig_size)
        axes[i][j].xaxis.set_label_coords(0.5,1.1)    
    if j==0: 
        axes[i][j].set_ylabel(e,size=2*fig_size,color='black',rotation=0)
        axes[i][j].yaxis.set_label_coords(-0.1,0.45)   
    axes[i][j].text(0.4,0.1, "distance : {}".format(np.round(distance,5)),size=2*fig_size)
    axes[i][j].set_xticklabels([])
    axes[i][j].set_yticklabels([])
    axes[i][j].set_xticks([])
    axes[i][j].set_yticks([])
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')    
    parser.add_argument('--epsilons', nargs='*', type = float , default = [0.1, 0.25, 0.5 ,0.75, 1, 2.5 , 5, 7.5 , 10, 20])
    parser.add_argument('--Models', nargs='*', default = ['LDP' ,'DP'])                 
    parser.add_argument('--fig_size', default = 5, type = int)
    
    args = parser.parse_args()

    fig_size = args.fig_size
    epsilons = args.epsilons  
    Models = args.Models    
    
    n_col = len(Models)
    n_row = len(epsilons)
    
    save_name = "DWS"

    epsilons = np.array(epsilons).astype(np.float32)

    file_dir= os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_dir)

    fig,axes=plt.subplots(n_row ,n_col,figsize=(n_col*fig_size,n_row*fig_size))
    axes = np.array([axes]).reshape(n_row,n_col)
    
    i=0
    for e in epsilons:
        j=0
        for m in Models:
            with open("utility_models/DWS/DWS_"+m+"_"+str(e) + ".pickle","rb") as fr:
                DWS_model = pickle.load(fr)
            draw_DWS_scatter(DWS_model,i,j, m, e)
            j+=1
        i+=1
        
    fig.savefig("figures/"+save_name+'.png',dpi=300,bbox_inches='tight',pad_inches = 0.1)