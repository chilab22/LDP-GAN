import numpy as np
import pandas as pd


def get_category_index(data):
    binary_index =[]
    continous_index=[]
    for i,c in enumerate(data.columns):
        if np.array_equal(data[c].unique(), data[c].unique().astype(bool)):
            binary_index.append(i)
        else:
            continous_index.append(i)
    return np.array(binary_index) , np.array(continous_index)


def get_syn_data(generator, n_data, latent_dim=100, rnd=5):
    z=tf.random.uniform((n_data,latent_dim),-1,1)
    syn=generator(z).numpy().reshape(n_data,-1)
    syn=sc.inverse_transform(syn)
    syn=pd.DataFrame(syn,columns=cols)
    for c in data.columns:
        syn[c]=syn[c].round(rnd)
    syn = pd.DataFrame(sc.transform(syn),columns=data.columns)
    return syn

