import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pickle
import math
import seaborn as sns
import argparse 

def draw_Corr_scatter(Corr_model, i,j, m,e):
    corr = Corr_model.syn_score
    distance = Corr_model.distance
    save_name = "Corr"    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plot = sns.heatmap(ax=axes[i][j],mask= mask, data=corr ,cbar=0,vmax = 1, vmin=-1,cmap='GnBu')
    plot.set(xticklabels=[])
    plot.set(yticklabels=[])
    plot.set_yticks([])
    plot.set_xticks([])
    for _, spine in plot.spines.items():
        spine.set_visible(True)
        spine.set_color("black")
    
    if i==0 and j==0:
        axes[i][j].set_xlabel(xlabel=m,rotation=0,size=fig_size*2)
        axes[i][j].xaxis.set_label_coords(0.5,1.1)    
    if i==0 and j==1:
        axes[i][j].set_xlabel(xlabel=m,rotation=0,size=fig_size*2)
        axes[i][j].xaxis.set_label_coords(0.5,1.1)    
    if j==0: 
        axes[i][j].set_ylabel(e,size=fig_size*2,color='black',rotation=0)
        axes[i][j].yaxis.set_label_coords(-0.1,0.45)   
    axes[i][j].text(data.shape[1]/2,data.shape[1]/4, "distance : {}".format(np.round(distance,5)),size=fig_size*2)

    
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='')    
    parser.add_argument('--epsilons', nargs='*', type = float , default = [0.1, 0.25, 0.5 ,0.75, 1, 2.5 , 5, 7.5 , 10, 20])
    parser.add_argument('--Models', nargs='*', default = ['LDP' ,'DP'])                 
    parser.add_argument('--fig_size', default = 5, type = int)
    parser.add_argument('--cmap', default ="GnBu")  
    parser.add_argument('--data', default ="train_data.csv")

    args = parser.parse_args()

    fig_size = args.fig_size
    epsilons = args.epsilons  
    Models = args.Models    
    cmap = args.cmap
    data = args.data
    
    
    
    n_col = len(Models)
    n_row = len(epsilons)
    save_name = "Corr"
    epsilons = np.array(epsilons).astype(np.float32)
    
    file_dir= os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_dir)    
    data = pd.read_csv("../data/"+data)
    
    
    
    #### display real data correlation
        
    plt.figure(figsize=(1.3*fig_size,1.7*fig_size))
    real_corr = data.corr()
    mask = np.triu(np.ones_like(real_corr, dtype=bool))
    ax=sns.heatmap(real_corr,cbar=1,vmax = 1, vmin=-1,mask=mask,cmap=cmap)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.5)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.set(xticks=[])
    ax.set(yticks=[])
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8,)
    plt.savefig("figures/real_Corr.png",dpi=300,bbox_inches='tight',pad_inches = 0.1)
    plt.show()

    #### display synthetic data correlation 
    fig,axes=plt.subplots(n_row ,n_col,figsize=(fig_size*n_col, n_row*fig_size))
    axes = np.array([axes]).reshape(n_row,n_col)
    i=0
    for e in epsilons:
        j=0
        for m in Models:
            
            with open("utility_models/Corr/Corr_"+m+"_"+str(e) + ".pickle","rb") as fr:
                Corr_model = pickle.load(fr)
            draw_Corr_scatter(Corr_model,i,j, m, e)
            j+=1
        i+=1
        
    fig.savefig("figures/"+"syn_"+save_name+'.png',dpi=300,bbox_inches='tight',pad_inches = 0.1)