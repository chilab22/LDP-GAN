import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score,f1_score
import tensorflow as tf
import tensorflow_probability as tfp

def get_FBA_result(generator, z,target):

    syn_data=generator(z).numpy().reshape(z.shape[0],-1)
            
    distances = np.mean(np.sqrt(np.square(target-syn_data)),axis=1)

    min_distance=np.min(distances)
    min_idx = np.where(distances==min_distance)[0][0]

    reconst = syn_data[min_idx]
    
    return min_distance, reconst



def get_PBA_result(generator, max_iterations, initial_vertex,target):
    def target_funtion(x):
        MSE=mse(generator(x),target).numpy()[0]
#         print(MSE)
        return MSE
    mse=tf.keras.metrics.mean_squared_error

    result=tfp.optimizer.nelder_mead_minimize(
    target_funtion, initial_simplex=None, initial_vertex=initial_vertex, step_sizes=None,
    objective_at_initial_simplex=None, objective_at_initial_vertex=None,
    batch_evaluate_objective=False, func_tolerance=1e-06, position_tolerance=1e-06,
    parallel_iterations=1, max_iterations=max_iterations, reflection=None, expansion=None,
    contraction=None, shrinkage=None, name=None)
    return result 

def get_WGA_result(generator, max_iterations, initial_vertex,target):
    def target_funtion(x):
        return tfp.math.value_and_gradient(lambda x: mse(generator(x),target),x)
    mse=tf.keras.metrics.mean_squared_error
    
    result=tfp.optimizer.lbfgs_minimize(
    target_funtion,
    initial_vertex,
    num_correction_pairs=10,
    tolerance=1e-08,
    x_tolerance=0,
    f_relative_tolerance=0,
    initial_inverse_hessian_estimate=None,
    max_iterations=max_iterations,
    parallel_iterations=1,
    stopping_condition=None,
    name=None)
    return result


def get_WDA_result(discriminator,train_data, ho_data, n_data, threshold):
    np.random.shuffle(train_data.to_numpy())
    sampled_train=train_data.iloc[:n_data].to_numpy()  
    train_df=pd.DataFrame(sampled_train,columns=train_data.columns)
    train_df['sort']='Train'
    train_df['score']=discriminator.predict(train_df.iloc[:,:-1])
    
    np.random.shuffle(ho_data.to_numpy())
    sampled_ho=ho_data.iloc[:n_data].to_numpy()
    ho_df=pd.DataFrame(ho_data,columns=train_data.columns)
    ho_df['sort']='Holdout'
    ho_df['score']=discriminator.predict(ho_df.iloc[:,:-1]) 
    
    concat_df=pd.concat([train_df,ho_df])
    Top_N=concat_df.sort_values(['score'],ascending=False).iloc[:threshold]
    return Top_N