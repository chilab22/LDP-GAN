import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm
import os
import random
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop,Adam
import numpy as np
from tensorflow.keras.models import *
from utils import *
import argparse

class WGAN_GP():
    def __init__(self, data_dim,epsilon, lr=0.00005,batch_size =64, latent_dim=100, n_critics=5, lambd = 10, log_step = 5000, decay = 0.995):
        self.lr = lr # learning late
        self.batch_size = batch_size 
        self.epsilon=epsilon
        self.log_step = log_step # every log step, log displayed
        self.n_critics = n_critics  # number of critic training step per generator step
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.lambd = lambd  #lambda in wgan-gp
        self.decay = decay
        self.critics = self.build_critics()
        self.generator = self.build_generator()
        self.name = self.get_name()
        
    def get_name(self):
        return "pure"

    ########################build_critics(discriminator)#######################
    def build_critics(self):
        model=Sequential()
        model.add(Dense(480,input_shape=self.data_dim))
        model.add(LeakyReLU(alpha=0.3))
        #14,14
    #     model.add(Dropout(0.3))
        model.add(Dense(360))
        model.add(LeakyReLU(alpha=0.3))
        #7,7
    #     model.add(Dropout(0.3))
        model.add(Dense(240))
        model.add(LeakyReLU(alpha=0.3))

        model.add(Dense(120))
        model.add(LeakyReLU(alpha=0.3))

        model.add(Dense(60))
        model.add(LeakyReLU(alpha=0.3))
        #4,4
    #     model.add(Dropout(0.3))
        model.add(Dense(30))
        model.add(LeakyReLU(alpha=0.3))
    #     model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(1))

        inputs=Input(shape=self.data_dim)
        outputs=model(inputs)
        return Model(inputs,outputs)


    ########################build_generator#######################
    def build_generator(self):

        model=Sequential()

        model.add(Dense(120,input_shape=(self.latent_dim,)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization())
        model.add(Dense(240))    
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization())
        model.add(Dense(360))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization())

        model.add(Dense(480))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization())
        model.add(Dense(self.data_dim[0],activation='sigmoid'))

        inputs=Input(shape=(self.latent_dim,))
        outputs=model(inputs)
        return Model(inputs,outputs)
    
    
    ####################gradient GP loss ####################
    def wasserstein_GP_loss(self, x, x_):
        lambd=10
        e=np.random.uniform(0,1)
        x_hat=e*x+(1-e)*x_    
        with tf.GradientTape() as tape:
            tape.watch(x_hat)
            loss=self.critics(x_hat)
        gradient=tape.gradient(loss,x_hat)
        gradient_norm2=tf.sqrt(tf.reduce_sum(tf.square(gradient),axis=(1)))
        gradient_penalty=lambd*tf.square((gradient_norm2-1))
        return self.critics(x_)-self.critics(x)+gradient_penalty
    ########################train_model#######################
    def train(self,train_data,epochs):
        train_data= train_data.to_numpy()
        loss=0
        acc=0
        optimizer=Adam(learning_rate=self.lr,beta_1=0,beta_2=0.9)
        for i in tqdm(range(epochs)):
            #######################
            ## train discrimator ##
            #######################
            for _ in range(self.n_critics):
                with tf.GradientTape() as tape:
                    ind=np.random.randint(0,train_data.shape[0], size=self.batch_size)
                    x=train_data[ind]
                    x=tf.constant(x,dtype=tf.float32)
                    z=tf.random.uniform(minval=-1,maxval=1,shape=(self.batch_size,self.latent_dim))
                    x_=self.generator(z)
                    critic_loss=self.wasserstein_GP_loss(x,x_)

                gradient=tape.gradient(critic_loss,self.critics.trainable_variables)
                optimizer.apply_gradients(zip(gradient,self.critics.trainable_variables))
                loss+=tf.reduce_mean(critic_loss)
            loss/=self.n_critics
            log="{} : [discriminator loss : {}]".format(i,loss)
            #######################
            ## train generator   ##
            #######################
            with tf.GradientTape() as tape:
                z=tf.random.uniform(minval=-1,maxval=1,shape=(self.batch_size,self.latent_dim))
                x_=self.generator(z)
                generator_score1=self.critics(x_)
                generator_loss1=tf.reduce_mean(-generator_score1)
            gradient=tape.gradient([generator_loss1],self.generator.trainable_variables)
            optimizer.apply_gradients(zip(gradient,self.generator.trainable_variables))
            log="%s : [adversarial loss : %f]"%(log,generator_loss1)
            if i%self.log_step==0:
                print(log)
                self.lr*=self.decay
                optimizer=Adam(learning_rate=self.lr,beta_1=0,beta_2=0.9)
        # save trained model
        self.critics.save("save/critics/{}_critics.h5".format(self.name))
        self.generator.save("save/generators/{}_generator.h5".format(self.name))
        print("saved")
        

class LDP_GAN(WGAN_GP):  
    
    def get_name(self):
        return "LDP_" + str(self.epsilon)
    
    def get_p(self):
        return np.e**self.epsilon/(np.e**self.epsilon+1)
    
    def LDP_noise(self,x):
                
        # Randomized response
        p = self.get_p()
        zeros=np.zeros(shape=x.shape)
        coin=np.random.uniform(0,1,size=(x.shape[0],self.binary_index.shape[0]))
        coin[coin<=p]=0   # head면 그대로
        coin[coin>p]=1    # tail이면 반대로 대답
        zeros[:,self.binary_index]=coin

        ones=tf.ones(shape=x.shape,dtype=tf.float32)
        negative = 2*tf.cast(x>0,tf.float32)
        negative = negative-ones
        x=tf.cast(tf.math.abs(x-zeros),dtype=tf.float32)
        x=tf.multiply(x,negative)
        # #     add noise to continous cols
        zeros=np.zeros(shape=(x.shape[0],x.shape[1]))
        for i in self.continous_index:
            noise=np.random.laplace(0,1/self.epsilon,size=(x.shape[0]))
            zeros[:,i]+=noise
        x=tf.math.add(x,zeros.reshape(x.shape[0],x.shape[1]))
        return x
    
    
    
    ########################train_model#######################
    def train(self,train_data,epochs):
        self.binary_index, self.continous_index = get_category_index(train_data)
        train_data= train_data.to_numpy()
        
        disturbed_data = self.LDP_noise(train_data).numpy()
        loss=0
        acc=0
        optimizer=Adam(learning_rate=self.lr,beta_1=0,beta_2=0.9)
        for i in tqdm(range(epochs)):
            #######################
            ## train discrimator ##
            #######################
            for _ in range(self.n_critics):
                with tf.GradientTape() as tape:
                    ind=np.random.randint(0,disturbed_data.shape[0], size=self.batch_size)
                    x=disturbed_data[ind]
                    x=tf.constant(x,dtype=tf.float32)
                    z=tf.random.uniform(minval=-1,maxval=1,shape=(self.batch_size,self.latent_dim))
                    x_=self.generator(z)
                    x_=Lambda(self.LDP_noise)(x_)
                    critic_loss=self.wasserstein_GP_loss(x,x_)

                gradient=tape.gradient(critic_loss,self.critics.trainable_variables)
                optimizer.apply_gradients(zip(gradient,self.critics.trainable_variables))
                loss+=tf.reduce_mean(critic_loss)
            loss/=self.n_critics
            log="{} : [discriminator loss : {}]".format(i,loss)
            #######################
            ## train generator   ##
            #######################
            with tf.GradientTape() as tape:
                z=tf.random.uniform(minval=-1,maxval=1,shape=(self.batch_size,self.latent_dim))
                x_=self.generator(z)
                x_=Lambda(self.LDP_noise)(x_)
                generator_score1=self.critics(x_)
                generator_loss1=tf.reduce_mean(-generator_score1)

            gradient=tape.gradient([generator_loss1],self.generator.trainable_variables)
            optimizer.apply_gradients(zip(gradient,self.generator.trainable_variables))
            log="%s : [adversarial loss : %f]"%(log,generator_loss1)
            if i%self.log_step==0:
                print(log)
                self.lr*=self.decay
                optimizer=Adam(learning_rate=self.lr,beta_1=0,beta_2=0.9)
        # save trained model
        self.critics.save("save/critics/{}_critics.h5".format(self.name))
        self.generator.save("save/generators/{}_generator.h5".format(self.name))
        print("saved")

        
        
        
        
        
        
class DP_GAN(WGAN_GP):
   
    def get_name(self):
        return "DP_" + str(self.epsilon)
    
    def DP_noise(self,x):
                
        # Randomized response
        p = self.get_p()
        zeros=np.zeros(shape=x.shape)
        coin=np.random.uniform(0,1,size=(x.shape[0],self.binary_index.shape[0]))
        coin[coin<=p]=0   # head면 그대로
        coin[coin>p]=1    # tail이면 반대로 대답
        zeros[:,self.binary_index]=coin

        ones=tf.ones(shape=x.shape,dtype=tf.float32)
        negative = 2*tf.cast(x>0,tf.float32)
        negative = negative-ones
        x=tf.cast(tf.math.abs(x-zeros),dtype=tf.float32)
        x=tf.multiply(x,negative)
        # #     add noise to continous cols
        zeros=np.zeros(shape=(x.shape[0],x.shape[1]))
        for i in self.continous_index:
            noise=np.random.laplace(0,1/self.epsilon,size=(x.shape[0]))
            zeros[:,i]+=noise
        x=tf.math.add(x,zeros.reshape(x.shape[0],x.shape[1]))
        return x
    
    
    ########################train_model#######################
    def train(self,train_data,epochs):
        sig=2*self.batch_size/train_data.shape[0]*np.sqrt(self.n_critics*np.log(10**5))/self.epsilon
        train_data= train_data.to_numpy()
        loss=0
        acc=0
        optimizer=Adam(learning_rate=self.lr,beta_1=0,beta_2=0.9)
        for i in tqdm(range(epochs)):
            #######################
            ## train discrimator ##
            #######################
            for _ in range(self.n_critics):
                with tf.GradientTape() as tape:
                    ind=np.random.randint(0,train_data.shape[0], size=self.batch_size)
                    x=train_data[ind]
                    x=tf.constant(x,dtype=tf.float32)
                    z=tf.random.uniform(minval=-1,maxval=1,shape=(self.batch_size,self.latent_dim))
                    x_=self.generator(z)
                    critic_loss=self.wasserstein_GP_loss(x,x_)

                gradient=tape.gradient(critic_loss,self.critics.trainable_variables)
                
                for ii in range(len(gradient)-1):
                    cp=np.maximum(gradient[ii].numpy().max(),np.abs(gradient[ii].numpy().min()))
                    gradient[ii]+=np.random.laplace(0.0,(sig**2)*(cp**2),size=gradient[ii].shape)
                
                optimizer.apply_gradients(zip(gradient,self.critics.trainable_variables))
                loss+=tf.reduce_mean(critic_loss)
            loss/=self.n_critics
            log="{} : [discriminator loss : {}]".format(i,loss)
            #######################
            ## train generator   ##
            #######################
            with tf.GradientTape() as tape:
                z=tf.random.uniform(minval=-1,maxval=1,shape=(self.batch_size,self.latent_dim))
                x_=self.generator(z)
                generator_score1=self.critics(x_)
                generator_loss1=tf.reduce_mean(-generator_score1)

            gradient=tape.gradient([generator_loss1],self.generator.trainable_variables)
            optimizer.apply_gradients(zip(gradient,self.generator.trainable_variables))
            log="%s : [adversarial loss : %f]"%(log,generator_loss1)
            if i%self.log_step==0:
                print(log)
                self.lr*=self.decay
                optimizer=Adam(learning_rate=self.lr,beta_1=0,beta_2=0.9)
        # save trained model
        self.critics.save("save/critics/{}_critics.h5".format(self.name))
        self.generator.save("save/generators/{}_generator.h5".format(self.name))
        print("saved")
        