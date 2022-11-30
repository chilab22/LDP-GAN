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
==글자파란색==
<mark>형광펜</mark>
<span style="color:red">빨간</span>

* <mark>train.py</mark> train the GAN models

* <span style="background-color:#D3D3D3">Utility/build_DWS.py</span> Build Dimension Wise Statistics(DWS) model. 
In this module, the DWS scores and distances of each GAN model are calculated.

* <span style="background-color:#D3D3D3">Utility/build_DWA.py</span> Build Dimension Wise Average(DWA) model. In this module, the DWA scores and distances of each GAN model are calculated.

* <span style="background-color:#D3D3D3">Utility/build_DWP.py</span> Build Dimension Wise Prediction(DWP) model. 
In this module, the DWP scores and distances of each GAN model are calculated.

* <span style="background-color:#D3D3D3">Utility/build_Corr.py</span> Build Correlation model. In this module, the Correlation scores and distances of each GAN model are calculated.



* <span style="background-color:#D3D3D3">Privacy/build_FBA.py</span> Build Full Black-box Attack(FBA) model. In this module, FBA for each model is performed.

* <span style="background-color:#D3D3D3">Privacy/build_PBA.py</span> Build Partial Black-box Attack(PBA) model. In this module, PBA for each model is performed.

* <span style="background-color:#D3D3D3">Privacy/build_WGA.py</span> Build White-box Attack(WGA) model. In this module, WGA for each model is performed.

* <span style="background-color:#D3D3D3">Privacy/build_WDA.py</span> Build Whitebox-Discriminator Attack WDA model. In this module, WDA for each model is performed.

# Usage
-------------------------------------------

## Model training

#### usage with default

```shell
python3 train.py
```

#### options

* <span style="background-color:#0000FF">--epochs</span> : Number of training epochs(int, defaul = 100000)

* <span style="background-color:#D3D3D3">--GPU</span> : Index of the GPU to use (int, default = 0)

* <span style="background-color:#D3D3D3">--epsilons</span> : Epsilons to use (float, default = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20])

* <span style="background-color:#D3D3D3">--Models</span> Kind of model to train. 
There are two types of selectable models, "LDP" and "DP". (text , default = ['LDP', 'DP'])

* <span style="background-color:#D3D3D3">--lr</span> Learning rate(float, default = 0.00005)

* <span style="background-color:#D3D3D3">--latent_dim</span> Dimension of latent vector (int, default = 100)

* <span style="background-color:#D3D3D3">--n_critics</span> : Training epochs of Critics per Generator training epoch (int, defalut = 5)

* <span style="background-color:#D3D3D3">--lambd</span> : Lambda of Wasserstain-loss (int, default = 10)

* <span style="background-color:#D3D3D3">--log_step</span> : The training status is displayed at every log step. (int, default = 5000)

* <span style="background-color:#D3D3D3">--decay</span> : Proportion of the learning rate that will change at each log step (float, default = 0.995)

* <span style="background-color:#D3D3D3">--data</span> : The name of the data to use for training file(csv).The data should be in the "/data" directory. (text, default = "train_data.csv)

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

* <span style="background-color:#D3D3D3">--data</span> Name of real data file(csv). This option is not used for "build_DWP.py". The data should be in the "/data" directory  (text, defaul = "train_data.csv")

* <span style="background-color:#D3D3D3">--latent_dim</span> Dimension of latent vector (int, default = 100)

* <span style="background-color:#D3D3D3">--GPU</span> Index of the GPU to use (int, default = 0)

* <span style="background-color:#D3D3D3">--epochs</span> Number of training epochs(int, defaul = 100000)

* <span style="background-color:#D3D3D3">--epsilons</span> Epsilons to use (float, default = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20])

* <span style="background-color:#D3D3D3">--Models</span> Kind of model to train. 
There are two types of selectable models, "LDP" and "DP". (text , default = ['LDP', 'DP'])

* <span style="background-color:#D3D3D3">--train</span> Name of train data file(csv). This options is only for "build_DWP.py" (text , default = 'train_data.csv')

* <span style="background-color:#D3D3D3">--ho</span>  Name of train data file(csv). This option is only for "build_DWP.py" (text , default = 'ho_data.csv')

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

* <span style="background-color:#D3D3D3">--epsilons</span> Epsilons to use (float, default = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20])

* <span style="background-color:#D3D3D3">--Models</span> Kind of model to train. 
There are two types of selectable models, "LDP" and "DP". (text , default = ['LDP', 'DP'])

* <span style="background-color:#D3D3D3">--fig_size</span> size of figures  (float, default = 5.0)

* <span style="background-color:#D3D3D3">--cmap</span> Color of correlation matrix. This option is only for "display_Corr.py" (text, default = "GnBu")

* <span style="background-color:#D3D3D3">--data</span> Name of real data file(csv). This option is only for "display_corr.py". The data should be in the "/data" directory  (text, defaul = "train_data.csv")

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

* <span style="background-color:#D3D3D3">--GPU</span> Index of the GPU to use (int, default = 0)

* <span style="background-color:#D3D3D3">--epsilons</span> Epsilons to use (float, default = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20])

* <span style="background-color:#D3D3D3">--Models</span> Kind of model to train. There are two types of selectable models, "LDP" and "DP". (text , default = ['LDP', 'DP'])
* <span style="background-color:#D3D3D3">--latent_dim</span> Dimension of latent vector.Must match the model's latent_dim. (int, default = 100)

* <span style="background-color:#D3D3D3">--threshold</span> the boundary of the distance. If the distance is less than the threshold, it is assumed that the target data is used for the train. (float, default = 0.05/0.025/0.005/300)

* <span style="background-color:#D3D3D3">--n_target</span> The number of targets to infer. (int, default = 600)

* <span style="background-color:#D3D3D3">--attack_on</span> File name of the data to be attacked(csv).The data should be in the "/data" directory.  (text, default = "train_data.csv")

* <span style="background-color:#D3D3D3">--n_syn</span> The number of data FBA generates for attacks. This option is only for "build_FBA.py" (int, default = 500)

* <span style="background-color:#D3D3D3">--max_iterations</span> max_iterations of nelder_mead/l-bfgs algorithms. This option is only for "build_PBA.py" and "build_WGA.py".(text, default = 500)

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

* <span style="background-color:#D3D3D3">--epsilons</span> Epsilons to use (float, default = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20])

* <span style="background-color:#D3D3D3">--Models</span> Kind of model to train. 
There are two types of selectable models, "LDP" and "DP". (text , default = ['LDP', 'DP']')

* <span style="background-color:#D3D3D3">--fig_size</span> size of figures  (float, default = 5.0)

* <span style="background-color:#D3D3D3">--facecolor</span> Base color of figure. (text, default = '#eaeaf2')

* <span style="background-color:#D3D3D3">--color_DP</span> Color of DP chart. (text, default = "#fd625e")

* <span style="background-color:#D3D3D3">--color_LDP</span> Color of LDP chart. (text, default = "#01b8aa")

* <span style="background-color:#D3D3D3">--GPU</span> Color of LDP chart. (int, default = 0)


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

* <span style="background-color:#D3D3D3">--epsilons</span> Epsilons to use (float, default = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20])

* <span style="background-color:#D3D3D3">--Models</span> Kind of model to train. 
There are two types of selectable models, "LDP" and "DP". (text , default = ['LDP', 'DP'])

* <span style="background-color:#D3D3D3">--fig_size</span> size of figures  (float, default = 5.0)

* <span style="background-color:#D3D3D3">--facecolor</span> Base color of figure. (text, default = '#eaeaf2')

* <span style="background-color:#D3D3D3">--color_DP</span> Color of DP chart. (text, default = "#fd625e")

* <span style="background-color:#D3D3D3">--color_LDP</span> Color of LDP chart. (text, default = "#01b8aa")

* <span style="background-color:#D3D3D3">--GPU</span> Color of LDP chart. (int, default = 0)

* <span style="background-color:#D3D3D3">--utility_models</span> Utility metrics to display trade-off.(text, default = ['DWS','DWA','DWP','Corr'])

* <span style="background-color:#D3D3D3">--attack_models</span> Attack metrics to display trade-off (text,  default = ['FBA','PBA','WGA','WDA'])  

#### optional usage

```shell

python3 trade-off/trade-off.py  --fig_size --utility_models DWS DWA DWP --privacy_models FBA PBA WGA

```
