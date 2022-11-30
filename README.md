# Generative Adversarial Network with Local Differential Privacy for patient data sysnthesis

description

# Require
----------------------

* python 3.8
* tensorflow 2.6
* sklearn 0.24.2
* tensorflow_probability 0.14.1
* matplotlib 3.4.2

# Module
-------------------------------------------


* ```train.py``` train the GAN models

* ```Utility/build_DWS.py``` Build Dimension Wise Statistics(DWS) model. 
In this module, the DWS scores and distances of each GAN model are calculated.

* ```Utility/build_DWA.py``` Build Dimension Wise Average(DWA) model. In this module, the DWA scores and distances of each GAN model are calculated.

* ```Utility/build_DWP.py``` Build Dimension Wise Prediction(DWP) model. 
In this module, the DWP scores and distances of each GAN model are calculated.

* ```Utility/build_Corr.py``` Build Correlation model. In this module, the Correlation scores and distances of each GAN model are calculated.



* ```Privacy/build_FBA.py``` Build Full Black-box Attack(FBA) model. In this module, FBA for each model is performed.

* ```Privacy/build_PBA.py``` Build Partial Black-box Attack(PBA) model. In this module, PBA for each model is performed.

* ```Privacy/build_WGA.py``` Build White-box Attack(WGA) model. In this module, WGA for each model is performed.

* ```Privacy/build_WDA.py``` Build Whitebox-Discriminator Attack WDA model. In this module, WDA for each model is performed.

# Usage
-------------------------------------------

## Model training

#### usage with default

```shell
python3 train.py
```

#### options

* <span style="background-color:#0000FF">--epochs``` : Number of training epochs(int, defaul = 100000)

* ```--GPU``` : Index of the GPU to use (int, default = 0)

* ```--epsilons``` : Epsilons to use (float, default = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20])

* ```--Models``` Kind of model to train. 
There are two types of selectable models, "LDP" and "DP". (text , default = ['LDP', 'DP'])

* ```--lr``` Learning rate(float, default = 0.00005)

* ```--latent_dim``` Dimension of latent vector (int, default = 100)

* ```--n_critics``` : Training epochs of Critics per Generator training epoch (int, defalut = 5)

* ```--lambd``` : Lambda of Wasserstain-loss (int, default = 10)

* ```--log_step``` : The training status is displayed at every log step. (int, default = 5000)

* ```--decay``` : Proportion of the learning rate that will change at each log step (float, default = 0.995)

* ```--data``` : The name of the data to use for training file(csv).The data should be in the "/data" directory. (text, default = "train_data.csv)

####  optional usage

```shell
python3 train.py --epochs 200000 --lr 0.0001 --epsilons 1.0 3.0 5.0 7.0 9.0 11.0 13.0 15.0 --GPU 1 --data train_data2.csv
```

  



## Build Utility model

####  usage with default

```shell

python3 Utility/build_DWS.py
```
```shell

python3 Utility/build_DWA.py

```
```shell

python3 Utility/build_DWP.py

```

```shell

python3 Utility/build_Corr.py

```

#### options

* ```--data``` Name of real data file(csv). This option is not used for "build_DWP.py". The data should be in the "/data" directory  (text, defaul = "train_data.csv")

* ```--latent_dim``` Dimension of latent vector (int, default = 100)

* ```--GPU``` Index of the GPU to use (int, default = 0)

* ```--epochs``` Number of training epochs(int, defaul = 100000)

* ```--epsilons``` Epsilons to use (float, default = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20])

* ```--Models``` Kind of model to train. 
There are two types of selectable models, "LDP" and "DP". (text , default = ['LDP', 'DP'])

* ```--train``` Name of train data file(csv). This options is only for "build_DWP.py" (text , default = 'train_data.csv')

* ```--ho```  Name of train data file(csv). This option is only for "build_DWP.py" (text , default = 'ho_data.csv')

#### optional usage
```shell

python3 Utility/build_DWS.py  --data train_data.csv --GPU 1 --epochs 200000  


```
```shell

python3 Utility/build_DWA.py --data train_data.csv --GPU 1 --epochs 200000 


```
```shell

python3 Utility/build_DWP.py --train train_data.csv --ho ho_data.csv --GPU 1 --epochs 200000 

```

```shell

python3 Utility/build_Corr.py --data train_data.csv --GPU 1 --epochs 200000 

```

## Display Utility Result

#### usage with default

```shell

python3 Utility/display_DWS.py

```
```shell

python3 Utility/display_DWA.py

```
```shell

python3 Utility/display_DWP.py

```

```shell

python3 Utility/display_Corr.py

```

#### options 

* ```--epsilons``` Epsilons to use (float, default = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20])

* ```--Models``` Kind of model to train. 
There are two types of selectable models, "LDP" and "DP". (text , default = ['LDP', 'DP'])

* ```--fig_size``` size of figures  (float, default = 5.0)

* ```--cmap``` Color of correlation matrix. This option is only for "display_Corr.py" (text, default = "GnBu")

* ```--data``` Name of real data file(csv). This option is only for "display_corr.py". The data should be in the "/data" directory  (text, defaul = "train_data.csv")

#### optional usage

```shell

python3 Utility/display_DWS.py --fig_size 3 

```
```shell

python3 Utility/display_DWA.py --fig_size 3

```
```shell

python3 Utility/display_DWP.py --fig_size 3 

```

```shell

python3 Utility/display_Corr.py --data ho_data.csv --cmap "BuPu"

```

#### Perform all utility experiments
You can adjust the options by editing the "Utility/run_utility.sh" file.

```shell

sh Utility/run_utility.sh

```


## Build Privacy model

#### usage with default

```shell

python3 Privacy/build_FBA.py 

```
```shell

python3 Privacy/build_PBA.py 

```
```shell

python3 Privacy/build_WGA.py

```

```shell

python3 Privacy/build_WDA.py 

```

#### options

* ```--GPU``` Index of the GPU to use (int, default = 0)

* ```--epsilons``` Epsilons to use (float, default = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20])

* ```--Models``` Kind of model to train. There are two types of selectable models, "LDP" and "DP". (text , default = ['LDP', 'DP'])
* ```--latent_dim``` Dimension of latent vector.Must match the model's latent_dim. (int, default = 100)

* ```--threshold``` the boundary of the distance. If the distance is less than the threshold, it is assumed that the target data is used for the train. (float, default = 0.05/0.025/0.005/300)

* ```--n_target``` The number of targets to infer. (int, default = 600)

* ```--attack_on``` File name of the data to be attacked(csv).The data should be in the "/data" directory.  (text, default = "train_data.csv")

* ```--n_syn``` The number of data FBA generates for attacks. This option is only for "build_FBA.py" (int, default = 500)

* ```--max_iterations``` max_iterations of nelder_mead/l-bfgs algorithms. This option is only for "build_PBA.py" and "build_WGA.py".(text, default = 500)

#### optional usage

```shell

python3 Privacy/build_FBA.py --n_syn 1000 --attack_on ho_data.csv

```
```shell

python3 Privacy/build_PBA.py --max_iterations 300 --n_target 1000

```
```shell

python3 Privacy/build_WGA.py --max_iterations 300 --n_target 1000

```

```shell

python3 Privacy/build_WDA.py  --max_iterations 300 --n_target 1000 --threshold=400

```

## Display Attack Result

#### usage with default

```shell

python3 Privacy/build_FBA.py 

```
```shell

python3 Privacy/build_PBA.py 

```
```shell

python3 Privacy/build_WGA.py

```
```shell

python3 Privacy/build_WDA.py 

```

#### options 

* ```--epsilons``` Epsilons to use (float, default = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20])

* ```--Models``` Kind of model to train. 
There are two types of selectable models, "LDP" and "DP". (text , default = ['LDP', 'DP']')

* ```--fig_size``` size of figures  (float, default = 5.0)

* ```--facecolor``` Base color of figure. (text, default = '#eaeaf2')

* ```--color_DP``` Color of DP chart. (text, default = "#fd625e")

* ```--color_LDP``` Color of LDP chart. (text, default = "#01b8aa")

* ```--GPU``` Color of LDP chart. (int, default = 0)


#### optional usage

```shell

python3 Privacy/build_FBA.py  --color_DP '#01b8aa'  --color_LDP '#fd625e' --fig_size 3.0

```
```shell

python3 Privacy/build_PBA.py --color_DP '#01b8aa'  --color_LDP '#fd625e' --fig_size 3.0

```
```shell

python3 Privacy/build_WGA.py --color_DP '#01b8aa'  --color_LDP '#fd625e' --fig_size 3.0

```
```shell

python3 Privacy/build_WDA.py --color_DP '#01b8aa'  --color_LDP '#fd625e' --fig_size 3.0

```

#### Perform all utility experiments
You can adjust the options by editing the "Utility/run_privacy.sh" file.

```shell

sh Utility/run_privacy.sh

```

## Display trade-off

#### usage with default

```shell

python3 trade-off/trade-off.py 

```

#### options

* ```--epsilons``` Epsilons to use (float, default = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20])

* ```--Models``` Kind of model to train. 
There are two types of selectable models, "LDP" and "DP". (text , default = ['LDP', 'DP'])

* ```--fig_size``` size of figures  (float, default = 5.0)

* ```--facecolor``` Base color of figure. (text, default = '#eaeaf2')

* ```--color_DP``` Color of DP chart. (text, default = "#fd625e")

* ```--color_LDP``` Color of LDP chart. (text, default = "#01b8aa")

* ```--GPU``` Color of LDP chart. (int, default = 0)

* ```--utility_models``` Utility metrics to display trade-off.(text, default = ['DWS','DWA','DWP','Corr'])

* ```--attack_models``` Attack metrics to display trade-off (text,  default = ['FBA','PBA','WGA','WDA'])  

#### optional usage

```shell

python3 trade-off/trade-off.py  --fig_size --utility_models DWS DWA DWP --privacy_models FBA PBA WGA

```
