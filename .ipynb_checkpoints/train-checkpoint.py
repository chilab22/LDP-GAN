import argparse 
import importlib
import pandas as pd
from model import *
import os


# # print(a.c)
parser = argparse.ArgumentParser(description='') 

parser.add_argument('--epochs', type = int , default = 100000)
parser.add_argument('--GPU', default = "0")
parser.add_argument('--epsilons', nargs='*', type = float , default = [0.1, 0.25, 0.5 ,0.75, 1, 2.5 , 5, 7.5 , 10, 20])
parser.add_argument('--Models', nargs='*', default = ['LDP' ,'DP'])                 
parser.add_argument('--lr', default = 0.00005, type=float)  # learning rate
parser.add_argument('--batch_size', default = 64, type=int) 
parser.add_argument('--latent_dim', default = 100, type=int)
parser.add_argument('--n_critics', default = 5, type=int)
parser.add_argument('--lambd', default = 10, type=int)      # lambda in wgan-gp loss
parser.add_argument('--log_step', default = 5000, type=int)    # every log_step, loss displayed
parser.add_argument('--decay', default = 0.995, type=float) # 
parser.add_argument('--data', default = "train_data.csv") # 

args = parser.parse_args()
# print(args.epsilons) 



epsilons = args.epsilons 
GPU_to_use = args.GPU
epochs = args.epochs
Models = args.Models
lr = args.lr
batch_size = args.batch_size
latent_dim = args.latent_dim
n_critics = args.n_critics
labmd = args.n_critics
log_step = args.log_step
decay = args.decay
data = args.data 

file_dir= os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)

train_data=pd.read_csv('data/'+data)
if 'Unnamed: 0' in train_data.columns:
    del train_data['Unnamed: 0'] 
data_dim = (train_data.shape[1],)

epsilons = np.array(epsilons).astype(np.float32)
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_to_use

for m in Models:
    for e in epsilons:
        print("############{} - {}##############".format(m,e))
        if m == "LDP":
            LDP_gan = LDP_GAN(data_dim, lr=lr,batch_size =batch_size, latent_dim=latent_dim, n_critics=n_critics, lambd = labmd, log_step = log_step, decay = decay,epsilon=e) 
            print(epochs)
            LDP_gan.train(train_data,epochs=epochs)
        if m == "DP":
            DP_gan = DP_GAN(data_dim, lr=lr,batch_size =batch_size, latent_dim=latent_dim, n_critics=n_critics, lambd = labmd, log_step = log_step, decay = decay,epsilon=e) 
            DP_gan.train(train_data,epochs=epochs)

