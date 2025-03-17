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

| Model              | 0.6/0.2/0.2 | Few-shot 10 samples | Few-shot 4 samples |
|--------------------|--------------------|--------------------|--------------------|
| Catboost          | 8.104 ± 5.71       | 12.531 ± 5.268     | 15.188 ± 4.906     |
| Emb-GBDT          | 10.354 ± 5.436     | 14.816 ± 4.742     | 15.125 ± 3.779     |
| XGBoost           | 9.729 ± 5.978      | 13.755 ± 5.238     | 15.0 ± 5.508       |
| LightGBM          | 9.958 ± 6.668      | 14.857 ± 8.065     | 13.396 ± 8.497     |
| Random Forest     | 7.729 ± 5.834      | 10.714 ± 6.007     | 10.354 ± 5.004     |
| TabNet            | 12.125 ± 6.645     | 10.959 ± 5.631     | 9.104 ± 6.429      |
| TabTransformer    | 18.708 ± 4.032     | 17.939 ± 3.608     | 14.229 ± 6.21      |
| FT-Transformer    | 12.854 ± 5.407     | 13.061 ± 6.326     | 14.167 ± 6.966     |
| Trompt            | 15.354 ± 4.596     | 16.388 ± 4.358     | 16.25 ± 4.388      |
| ExcelFormer-None  | 11.583 ± 6.317     | 9.939 ± 5.226      | 10.708 ± 5.871     |
| ExcelFormer-HID   | 13.0 ± 5.383       | 10.102 ± 5.753     | 10.354 ± 5.991     |
| ExcelFormer-FEAT  | 11.792 ± 6.358     | 9.367 ± 5.865      | 10.917 ± 5.834     |
| GNN               | 10.188 ± 5.584     | 9.245 ± 5.305      | 8.792 ± 4.631      |
| BGNN              | 10.958 ± 5.508     | 8.796 ± 4.411      | 7.021 ± 3.681      |
| BGNN-PL           | 7.667 ± 5.459      | 9.98 ± 5.914       | 11.188 ± 5.538     |
| Res-GNN           | 11.958 ± 5.458     | 9.327 ± 5.166      | 10.479 ± 5.993     |
| Res-GNN-L         | 12.833 ± 6.203     | 10.347 ± 5.648     | 9.75 ± 4.087       |
| aggBGNN          | 3.833 ± 2.793      | 6.184 ± 3.712      | 7.021 ± 3.641      |
| aggBGNN-LI       | 4.917 ± 4.196      | 7.612 ± 3.785      | 6.521 ± 3.585      |
| aggBGNN-DG       | 4.75 ± 3.716       | 6.633 ± 4.06       | 6.521 ± 3.427      |
| aggBGNN-v2       | 4.333 ± 3.616      | 6.51 ± 4.199       | 7.646 ± 3.744      |



The following charts also present the average ranking and standard deviation of various models across 47 small regression tasks.

| Model              | 0.6/0.2/0.2 | Few-shot 10 samples |
|--------------------|--------------------|--------------------|
| Catboost          | 6.532 ± 4.195       | 10.34 ± 3.708      |
| Emb-GBDT          | 7.404 ± 4.6         | 10.66 ± 3.466      |
| XGBoost           | 15.234 ± 5.197      | 14.106 ± 5.783     |
| LightGBM          | 9.064 ± 5.079       | 20.383 ± 1.095     |
| Random Forest     | 7.957 ± 6.097       | 7.723 ± 4.427      |
| TabNet            | 15.404 ± 5.941      | 16.128 ± 5.709     |
| TabTransformer    | 17.447 ± 4.831      | 14.489 ± 5.941     |
| FT-Transformer    | 12.66 ± 5.738       | 11.723 ± 6.959     |
| Trompt            | 7.404 ± 4.977       | 10.809 ± 5.282     |
| ExcelFormer-None  | 7.255 ± 5.387       | 6.66 ± 5.666       |
| ExcelFormer-HID   | 7.787 ± 4.93        | 5.574 ± 5.356      |
| ExcelFormer-FEAT  | 7.532 ± 4.854       | 5.478 ± 5.128      |
| GNN               | 13.191 ± 5.085      | 13.383 ± 3.423     |
| BGNN              | 12.745 ± 3.898      | 13.0 ± 4.854       |
| BGNN-PL           | 9.362 ± 4.66        | 10.468 ± 4.333     |
| Res-GNN           | 10.894 ± 5.557      | 10.17 ± 4.594      |
| Res-GNN-L         | 11.234 ± 5.414      | 9.489 ± 4.634      |
| aggBGNN           | 9.362 ± 4.923       | 8.787 ± 4.369      |
| aggBGNN-LI        | 15.149 ± 5.196      | 9.957 ± 6.061      |
| aggBGNN-DG        | 16.277 ± 4.628      | 9.0 ± 5.572        |
| aggBGNN-v2        | 10.936 ± 5.495      | 12.255 ± 5.87      |





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
