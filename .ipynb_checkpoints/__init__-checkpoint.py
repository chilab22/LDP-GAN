import pandas as pd
class WGAN_GP():
    def __init__(self, data_dim, epochs=100000, lr=0.00005,batch_size =64, latent_dim=100, n_critics=5, lambd = 10, log_step = 5000, decay = 0.995):
        self.epochs = epochs
        self.lr = lr # learning late
        self.batch_size = batch_size 
        self.log_step = log_step # every log step, log displayed
        self.n_critics = n_critics  # number of critic training step per generator step
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.lambd = lambd  #lambda in wgan-gp
        self.decay = decay
        
        self.critics = self.build_critics()
        self.generator = self.build_generator()
        

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
        return self.critics(x_)-cself.ritics(x)+gradient_penalty
    ########################train_model#######################
    def train(self,train_data):
        loss=0
        acc=0
        optimizer=Adam(learning_rate=self.lr,beta_1=0,beta_2=0.9)
        for i in tqdm(range(step)):
            #######################
            ## train discrimator ##
            #######################
            for _ in range(self.n_critics):
                with tf.GradientTape() as tape:
                    ind=np.random.randint(0,train_data.shape[0], size=self.batch_size)
                    x=train_data[ind]
                    x=tf.constant(x,dtype=tf.float32)
                    z=tf.random.uniform(minval=-1,maxval=1,shape=(batch_size,latent_dim))
                    x_=generator(z)
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

                z=tf.random.uniform(minval=-1,maxval=1,shape=(batch_size,latent_dim))
                x_=generator(z)
                generator_score1=critics(x_)
                generator_loss1=tf.reduce_mean(-generator_score1)

            gradient=tape.gradient([generator_loss1],generator.trainable_variables)
            optimizer.apply_gradients(zip(gradient,generator.trainable_variables))
            log="%s : [adversarial loss : %f]"%(log,generator_loss1)
            if i%self.log_step==0:
                print(log)
                lr*=self.decay
                optimizer=Adam(learning_rate=lr,beta_1=0,beta_2=0.9)
    