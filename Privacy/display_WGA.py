import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pickle
import math
import argparse 

def draw_WGA_chart(WGA_model, m,e):
    
    axes[0].set_xlim([0,WGA_model.n_target])
    
    if "_LDP" in WGA_model.name_:
        axes[0].barh(str(e), WGA_model.TP ,hatch="///", align='center', color=color_LDP, zorder=10)
    axes[0].set_title("LDP-GAN", fontsize=3*fig_size, pad=3*fig_size, color=color_LDP)
    axes[0].grid()
    axes[0].invert_xaxis()
    axes[0].tick_params(left=False,right=False)

    
    axes[1].set_xlim([0,WGA_model.n_target])
    
    if "_DP" in WGA_model.name_:
        axes[1].barh(str(e), WGA_model.TP , align='center', color=color_DP, zorder=10)
    axes[1].set_title("DP-GAN", fontsize=3*fig_size, pad=3*fig_size, color=color_DP)
    axes[1].grid()
    axes[1].tick_params(left=False,right=False)
    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)

    
if __name__=="__main__":
    
    
    parser = argparse.ArgumentParser(description='') 

    parser.add_argument('--epsilons', nargs='*', type = float , default = [0.1, 0.25, 0.5 ,0.75, 1, 2.5 , 5, 7.5 , 10, 20])
    parser.add_argument('--Models', nargs='*', default = ['LDP' ,'DP'])                 
    parser.add_argument('--fig_size',  default = 5, type = int)                 
    parser.add_argument('--facecolor',  default = '#eaeaf2')
    parser.add_argument('--color_DP',  default = '#fd625e')
    parser.add_argument('--color_LDP',  default = '#01b8aa')
    parser.add_argument('--GPU', default = "0")

    args = parser.parse_args()    

    epsilons = args.epsilons 
    fig_size = args.fig_size
    Models = args.Models
    facecolor = args.facecolor
    color_DP = args.color_DP
    color_LDP = args.color_LDP
    GPU_to_use = args.GPU

    n_col = len(Models)
    n_row = len(epsilons)

    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_to_use

    epsilons = np.array(epsilons).astype(np.float32)

    file_dir= os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_dir)

    save_name = "WGA"

    fig,axes=plt.subplots(ncols = n_col,figsize=(2*fig_size,fig_size),facecolor=facecolor,sharey=True)
    
    for e in epsilons:
        for m in Models:
            with open("privacy_models/WGA/WGA_"+m+"_"+str(e) + ".pickle","rb") as fr:
                WGA_model = pickle.load(fr)
            draw_WGA_chart(WGA_model, m, e)
        
    fig.savefig("figures/"+save_name+'.png',dpi=300,bbox_inches='tight',pad_inches = 0.1)