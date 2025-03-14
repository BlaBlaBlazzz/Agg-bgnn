### Aggregated Boosted Graph Neural Networks

The concepts are from ICLR 2021 paper: [Boost then Convolve: Gradient Boosting Meets Graph Neural Networks](https://openreview.net/pdf?id=ebS5NUfoMKL)
We extend the graph-based model to tabular data and compare it with state-of-the-art (SOTA) models.

This code contains implementation of the following models for graphs: 
* **CatBoost**
* **LightGBM**
* **XGBoost**
* **Random Forest**
* **ExcelFormer**
* **Trompt**
* **Tabtransformer**
* **FTTransformer**
* **TabNet**
* **Fully-Connected Neural Network** (FCNN)
* **GNN** (GAT, GCN, AGNN, APPNP)
* **FCNN-GNN** (GAT, GCN, AGNN, APPNP)
* **ResGNN** ({CatBoost, LightGBM, XGBoost} + {GAT, GCN, AGNN, APPNP})
* **BGNN** (end-to-end {CatBoost + {GAT, GCN, AGNN, APPNP}})
* **Agg-BGNN**

## Installation
To run the models you have to download the repo, install the requirements, and extract the datasets.

First, let's create a python environment:
```bash
mkdir envs
cd envs
python -m venv bgnn_env
source bgnn_env/bin/activate
cd ..
```
---
Second, let's download the code and install requirements
```bash
git clone https://github.com/nd7141/bgnn.git 
cd bgnn
unzip datasets.zip
make install
```
---
Next we need to install a proper version of [PyTorch](https://pytorch.org/) and [DGL](https://www.dgl.ai/), depending on the cuda version of your machine.
We strongly encourage to use GPU-supported versions of DGL (the speed up in training can be 100x).

First, determine your cuda version with `nvcc --version`. 
Then, check installation instructions for [pytorch](https://pytorch.org/get-started/locally/).
For example for cuda version 9.2, install it as follows:
```bash
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

If you don't have GPU, use the following: 
```bash
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
---
Similarly, you need to install [DGL library](https://docs.dgl.ai/en/0.4.x/install/). 
For example, cuda==9.2:

```bash
pip install dgl-cu92
```

For cpu version of DGL: 
```bash
pip install dgl
```

Tested versions of `torch` and `dgl` are:
* torch==1.7.1+cu92
* dgl_cu92==0.5.3

## Running
Starting point is file `scripts/run.py`:
```bash
python scripts/run.py dataset models 
    (optional) 
            --save_folder: str = None
            --task: str = 'regression',
            --repeat_exp: int = 1,
            --max_seeds: int = 5,
            --dataset_dir: str = None,
            --config_dir: str = None
```
Available options for dataset: 
* house (regression)
* county (regression)
* vk (regression)
* wiki (regression)
* avazu (regression)
* vk_class (classification)
* house_class (classification)
* dblp (classification)
* slap (classification)
* path/to/your/dataset
    
Available options for models are `catboost`, `lightgbm`, `gnn`, `resgnn`, `bgnn`, `all`.

Each model is specifed by its config. Check [`configs/`](https://github.com/nd7141/bgnn/tree/master/configs/model) folder to specify parameters of the model and run.

Upon completion, the results wil be saved in the specifed folder (default: `results/{dataset}/day_month/`).
This folder will contain `aggregated_results.json`, which will contain aggregated results for each model.
Each model will have 4 numbers in this order: `mean metric` (RMSE or accuracy), `std metric`, `mean runtime`, `std runtime`.
File `seed_results.json` will have results for each experiment and each seed. 
Additional folders will contain loss values during training. 

---

###Examples

The following script will launch all models on `House` dataset.  
```bash
python scripts/run.py house all
```

The following script will launch CatBoost and GNN models on `SLAP` classification dataset.  
```bash
python scripts/run.py slap catboost gnn --task classification
```

The following script will launch LightGBM model for 5 splits of data, repeating each experiment for 3 times.  
```bash
python scripts/run.py vk lightgbm --repeat_exp 3 --max_seeds 5
```

The following script will launch resgnn and bgnn models saving results to custom folder.  
```bash
python scripts/run.py county resgnn bgnn --save_folder ./county_resgnn_bgnn
```

### Running on your dataset
To run the code on your dataset, it's necessary to prepare the files in the right format. 

You can check examples in `datasets/` folder. 

There should be at least `X.csv` (node features), `y.csv` (target labels), `graph.graphml` (graph in graphml format).

Make sure to keep _these_ filenames for your dataset.

You can also have `cat_features.txt` specifying names of categorical columns.

You can also have `masks.json` specifying train/val/test splits. 

After that run the script as usual: 
```bash
python scripts/run.py path/to/your/dataset gnn catboost 
```

### Benchmark

We evaluate our model on multiple small datasets (48 classification tasks + 47 regression tasks) stated in 2023 paper: [ExcelFormer: A neural network surpassing GBDTs on tabular data](https://github.com/WhatAShot/ExcelFormer)

We benchmark models including classic machine learning models, tabular deep learning models, against graph-based models applying to tabular datasets.

The charts below summarize the average ranking and standard deviation of various models accross 48 small classification tasks and few-shot scenarios, where a lower rank indicates better performances. Each row corresponds to a model, while each column represents 0.6/0.2/0.2 proportion or few-shot scenarios across datasets.

| Model              | 0.6 /0.2/0.2 | few-shot 10 samples | few-shot 4 samples |
|--------------------|----------------------------------|-----------------------------------|----------------------------------|
| BGNN              | 10.542 ± 5.882                   | 9.522 ± 5.573                    | 9.062 ± 4.76                     |
| BGNN-PL           | 11.438 ± 5.772                   | 8.978 ± 4.553                    | 7.208 ± 3.842                    |
| resGNN           | 7.792 ± 5.592                    | 10.391 ± 6.198                   | 11.521 ± 5.765                   |
| resGNN-L         | 12.417 ± 5.764                   | 9.761 ± 5.309                    | 10.917 ± 6.277                   |
| resGNN-XGB       | 12.208 ± 5.653                   | 12.087 ± 5.826                   | 12.688 ± 5.288                   |
| GNN              | 13.375 ± 6.529                   | 10.913 ± 5.876                   | 9.938 ± 4.23                     |
| emb-GBDT         | 10.729 ± 5.775                   | 15.478 ± 5.146                   | 15.812 ± 4.009                   |
| Catboost         | 8.333 ± 5.98                     | 12.761 ± 5.486                   | 15.896 ± 5.244                   |
| LightGBM         | 10.354 ± 7.033                   | 16.543 ± 7.831                   | 14.0 ± 8.944                     |
| XGBoost          | 10.021 ± 6.262                   | 14.609 ± 5.331                   | 15.729 ± 5.779                   |
| Random Forest    | 7.917 ± 6.091                    | 10.978 ± 6.445                   | 10.771 ± 5.325                   |
| ExcelFormer-None | 12.083 ± 6.677                   | 10.174 ± 5.587                   | 11.208 ± 6.209                   |
| ExcelFormer-hidden | 13.562 ± 5.672                 | 10.696 ± 6.124                   | 10.812 ± 6.317                   |
| ExcelFormer-feat | 12.271 ± 6.69                    | 9.652 ± 6.023                    | 11.354 ± 6.183                   |
| Trompt          | 16.042 ± 4.907                    | 17.109 ± 4.663                   | 16.979 ± 4.624                   |
| TabNet          | 12.583 ± 7.001                    | 11.478 ± 5.73                    | 9.479 ± 6.804                    |
| TabTransformer  | 19.583 ± 4.262                    | 19.065 ± 3.007                   | 14.792 ± 6.556                   |
| FTTransformer   | 13.333 ± 5.751                    | 13.609 ± 6.718                   | 14.771 ± 7.329                   |
| aggBGNN         | **3.938 ± 2.942**                     | **6.261 ± 3.803**                    | 7.229 ± 3.816                    |
| aggBGNN-dnf     | 5.042 ± 4.41                      | 7.457 ± 3.698                    | 6.771 ± 3.816                    |
| aggBGNN-dg      | 4.896 ± 3.888                     | 6.891 ± 4.423                    | **6.75 ± 3.635**                     |
| aggBGNN-v2      | 4.438 ± 3.825                     | 6.457 ± 4.194                    | 7.875 ± 3.993                    |



The following charts also present the average ranking and standard deviation of various models across 47 small regression tasks.

| Model              | 0.6/0.2/0.2    | few-shot 10 samples |
|--------------------|----------------------|------------------------------------|
| BGNN               | 13.915 ± 5.332     | 14.085 ± 3.629                    |
| BGNN-PL            | 13.34 ± 4.125      | 13.553 ± 5.141                    |
| resGNN             | 9.787 ± 4.899      | 10.915 ± 4.457                    |
| resGNN-L           | 11.362 ± 5.754     | 10.596 ± 4.889                    |
| resGNN-XGB         | 11.277 ± 4.822     | 11.511 ± 5.034                    |
| GNN                | 11.809 ± 5.663     | 9.872 ± 4.739                     |
| emb-GBDT           | 7.638 ± 4.914      | 11.106 ± 3.737                    |
| Catboost           | **6.787 ± 4.496**      | 10.787 ± 3.939                    |
| LightGBM           | 9.447 ± 5.376      | 21.383 ± 1.095                    |
| XGBoost            | 16.0 ± 5.481       | 14.809 ± 6.103                    |
| Random Forest      | 8.277 ± 6.456         | 8.0 ± 4.695                       |
| ExcelFormer-None   | 7.617 ± 5.739         | 6.979 ± 5.991                     |
| ExcelFormer-hidden | 8.149 ± 5.254         | 5.745 ± 5.659                     |
| ExcelFormer-feat   | 7.467 ± 4.698         | **5.674 ± 5.437**                     |
| Trompt             | 7.702 ± 5.361         | 11.383 ± 5.647                    |
| TabNet             | 16.064 ± 6.326        | 16.894 ± 6.069                    |
| TabTransformer     | 18.362 ± 4.963        | 15.255 ± 6.247                    |
| FTTransformer      | 13.213 ± 6.104        | 12.234 ± 7.382                    |
| aggBGNN            | 9.66 ± 5.227          | 9.128 ± 4.698                     |
| aggBGNN-dnf        | 15.936 ± 5.411        | 10.404 ± 6.382                    |
| aggBGNN-dg         | 17.085 ± 4.889        | 9.426 ± 5.897                     |
| aggBGNN-v2         | 11.298 ± 5.86         | 12.83 ± 6.243                     |



## Citation
```
@inproceedings{
ivanov2021boost,
title={Boost then Convolve: Gradient Boosting Meets Graph Neural Networks},
author={Sergei Ivanov and Liudmila Prokhorenkova},
booktitle={International Conference on Learning Representations (ICLR)},
year={2021},
url={https://openreview.net/forum?id=ebS5NUfoMKL}
}
```
