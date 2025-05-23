import sys
import os
curPath = os.path.dirname(os.path.dirname(__file__))
sys.path.append(curPath)

from bgnn.models.GBDT import GBDTCatBoost, GBDTLGBM, GBDTXGBoost
from bgnn.models.RandomForest import RandomForest
from bgnn.models.MLP import MLP
from bgnn.models.GNN import GNN
from bgnn.models.BGNN import BGNN
from bgnn.models.BGNN_v2 import BGNN_v2
from bgnn.models.ExcelFormer import ExcelFormer
from bgnn.models.trompt import trompt
from bgnn.models.tabnet import tabnet
from bgnn.models.tabtransformer import tabtransformer
from bgnn.models.fttransformer import fttransformer
from bgnn.models.aggBGNN import aggBGNN
from bgnn.models.aggBGNN_dnf import aggBGNN_dnf
from bgnn.models.aggBGNN_dg import aggBGNN_dg
from bgnn.models.aggBGNN_v2 import aggBGNN_v2
from bgnn.models.ABGNN import ABGNN
from bgnn.scripts.utils import NpEncoder
from bgnn.models.Base import BaseModel

import os
import json
import time
import datetime
from pathlib import Path
from collections import defaultdict as ddict

import pandas as pd
import networkx as nx
import torch
import dgl
import random
import warnings
import numpy as np
import fire
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor 
from sklearn.metrics.pairwise import cosine_similarity

class RunModel:
    def __init__(self):
        super(RunModel, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def read_input(self, input_folder):
        self.X = pd.read_csv(f'{input_folder}/X.csv')
        self.y = pd.read_csv(f'{input_folder}/y.csv')

        # if os.path.exists(f'{input_folder}/graph.graphml'):
        #     networkx_graph = nx.read_graphml(f'{input_folder}/graph.graphml')
        #     networkx_graph = nx.relabel_nodes(networkx_graph, {str(i): i for i in range(len(networkx_graph))})
        #     self.networkx_graph = networkx_graph
        # else:
        #     self.networkx_graph = None

        categorical_columns = []
        if os.path.exists(f'{input_folder}/cat_features.txt'):
            with open(f'{input_folder}/cat_features.txt') as f:
                for line in f:
                    if line.strip():
                        categorical_columns.append(line.strip())

        self.cat_features = None
        if categorical_columns:
            columns = self.X.columns
            self.cat_features = np.where(columns.isin(categorical_columns))[0]

            for col in list(columns[self.cat_features]):
                self.X[col] = self.X[col].astype(str)


        if os.path.exists(f'{input_folder}/masks.json'):
            with open(f'{input_folder}/masks.json') as f:
                self.masks = json.load(f)
        else:
            print('Creating and saving train/val/test masks')
            idx = list(range(self.y.shape[0]))
            self.masks = dict()
            for i in range(self.max_seeds):
                random.shuffle(idx)
                r1, r2, r3 = idx[:int(.6*len(idx))], idx[int(.6*len(idx)):int(.8*len(idx))], idx[int(.8*len(idx)):]
                self.masks[str(i)] = {"train": r1, "val": r2, "test": r3}

            with open(f'{input_folder}/masks.json', 'w+') as f:
                json.dump(self.masks, f, cls=NpEncoder)


    def get_input(self, dataset_dir, dataset: str):
        if dataset == 'house':
            input_folder = dataset_dir / 'house'
        elif dataset == 'county':
            input_folder = dataset_dir / 'county'
        elif dataset == 'vk':
            input_folder = dataset_dir / 'vk'
        elif dataset == 'wiki':
            input_folder = dataset_dir / 'wiki'
        elif dataset == 'avazu':
            input_folder = dataset_dir / 'avazu'
        elif dataset == 'vk_class':
            input_folder = dataset_dir / 'vk_class'
        elif dataset == 'house_class':
            input_folder = dataset_dir / 'house_class'
        elif dataset == 'dblp':
            input_folder = dataset_dir / 'dblp'
        elif dataset == 'slap':
            input_folder = dataset_dir / 'slap'
        else:
            input_folder = dataset_dir / f'{dataset}'
            print(input_folder)

        if self.save_folder is None:
            self.save_folder = f'results_v2/{dataset}/{datetime.datetime.now().strftime("%d_%m")}'

        self.read_input(input_folder)
        print('Save to folder:', self.save_folder)


    def run_one_model(self, config_fn, model_name):
        self.config = OmegaConf.load(config_fn)
        grid = ParameterGrid(dict(self.config.hp))

        for ps in grid:
            param_string = ''.join([f'-{key}{ps[key]}' for key in ps])
            exp_name = f'{model_name}{param_string}'
            print(f'\nSeed {self.seed} RUNNING:{exp_name}')

            runs = []
            runs_custom = []
            times = []
            for _ in range(self.repeat_exp):
                start = time.time()
                model = self.define_model(model_name, ps)
                
                inputs = {'X': self.X, 'y': self.y, 'train_mask': self.train_mask,
                          'val_mask': self.val_mask, 'test_mask': self.test_mask, 'cat_features': self.cat_features}
                # graph
                if model_name in ['gnn', 'resgnn', 'resgnn_LI', 'bgnn', 'resgnnL', 'resgnnSVM', 'resgnnXG', 'bgnn_v2']:
                    inputs['networkx_graph'] = self.graph
                elif model_name in ['aggBGNN', 'aggBGNN_dnf', 'aggBGNN_dg', 'aggBGNN_v2', 'abgnn']:
                    inputs['graph'] = self.graph
                    inputs['graph_pred'] = self.graph_pred
                    inputs['graph_leaf'] = self.graph_leaf
                
                metrics = model.fit(num_epochs=self.config.num_epochs, patience=self.config.patience,
                           loss_fn=f"{self.seed_folder}/{exp_name}.txt",
                           metric_name='loss' if self.task == 'regression' else 'accuracy', **inputs)
                finish = time.time()

                best_loss = min(metrics['loss'], key=lambda x: x[1])
                best_custom = max(metrics['r2' if self.task == 'regression' else 'accuracy'], key=lambda x: x[1])
                runs.append(best_loss)
                runs_custom.append(best_custom)
                times.append(finish - start)
            self.store_results[exp_name] = (list(map(np.mean, zip(*runs))),
                                       list(map(np.mean, zip(*runs_custom))),
                                       np.mean(times),
                                       )
            # print("store result:", self.store_results)
    
    def define_model(self, model_name, ps):
        if model_name == 'catboost':
            return GBDTCatBoost(self.task, **ps)
        elif model_name == 'lightgbm':
            print(ps)
            return GBDTLGBM(self.task, **ps)
        elif model_name == 'xgboost':
            return GBDTXGBoost(self.task, **ps)
        elif model_name == 'mlp':
            return MLP(self.task, **ps)
        elif model_name == 'gnn':
            return GNN(self.task, **ps)
        elif model_name == 'emb-GBDT':
            model = GNN(self.task)
            x = model.pandas_to_torch(self.X.astype("float"))[0]
            node_features = model.init_node_features(x, False)
            # graph = model.networkx_to_torch(self.networkx_graph)
            # node embedding
            model.fit(self.X, self.y, self.train_mask, self.val_mask, self.test_mask,
                      cat_features=self.cat_features, networkx_graph=self.graph, 
                      num_epochs=1000, patience=100,
                      metric_name='loss' if self.task == 'regression' else 'accuracy')
            node_embedding = model.model(self.graph, node_features).detach().cpu().numpy()
            # print(node_embedding)
            return GBDTCatBoost(self.task, **ps, gnn_embedding=node_embedding)
        elif model_name == 'resgnn':
            gbdt = GBDTCatBoost(self.task)
            gbdt.fit(self.X, self.y, self.train_mask, self.val_mask, self.test_mask,
                     cat_features=self.cat_features,
                     num_epochs=1000, patience=100,
                     plot=False, verbose=False, loss_fn=None,
                     metric_name='loss' if self.task == 'regression' else 'accuracy')
            # print(gbdt.model.predict(self.X).shape)
            return GNN(task=self.task, gbdt_predictions=gbdt.model.predict(self.X), **ps)
        # resgnn with leaf index
        elif model_name == 'resgnn_LI':
            gbdt = GBDTCatBoost(self.task)
            gbdt.fit(self.X, self.y, self.train_mask, self.val_mask, self.test_mask,
                     cat_features=self.cat_features,
                     num_epochs=100, patience=100,
                     plot=False, verbose=False, loss_fn=None,
                     metric_name='loss' if self.task == 'regression' else 'accuracy')
            leaf_idx = gbdt.model.calc_leaf_indexes(self.X)
            # print("leaf_idx", leaf_idx)
            return GNN(task=self.task, gbdt_predictions=leaf_idx, **ps)
        elif model_name == 'resgnnL':
            gbdt = GBDTLGBM(self.task)
            gbdt.fit(self.X, self.y, self.train_mask, self.val_mask, self.test_mask,
                     cat_features = self.cat_features,
                     num_epochs = 300, patience = 100,
                     loss_fn=None, metric_name='accuracy')
            predictions = gbdt.model.predict(self.X)
            predictions = [np.argmax(line) for line in predictions]
            predictions = np.reshape(predictions, (len(predictions), 1))
            # print("prediction:", predictions.shape)
            return GNN(task=self.task, gbdt_predictions=predictions, **ps)
        elif model_name == 'resgnnXG':
            x = self.X.copy()
            if self.cat_features is not None:
                for col in list(self.X.columns[self.cat_features]):
                    x[col] = self.X[col].astype('category')
            gbdt = GBDTXGBoost(self.task)
            gbdt.fit(x, self.y, self.train_mask, self.val_mask, self.test_mask,
                     cat_features = self.cat_features,
                     num_epochs = 300, patience = 100,
                     loss_fn=None, metric_name='accuracy' if self.task=='classification' else 'loss')
            dX = xgb.DMatrix(x, enable_categorical=True)
            predictions = gbdt.model.predict(dX)
            predictions = np.reshape(predictions, (len(predictions), 1))
            return GNN(task=self.task, gbdt_predictions=predictions, **ps)

        elif model_name == 'bgnn':
            return BGNN(self.task, **ps)
        elif model_name == 'bgnn_v2':
            return BGNN_v2(self.task, **ps)
        elif model_name == 'ExcelFormer':
            return ExcelFormer(self.task, **ps)
        elif model_name == 'abgnn':
            return ABGNN(self.task, **ps)
        else:
            module = globals()[model_name]
            return module(self.task, **ps)

    def create_save_folder(self, seed):
        self.seed_folder = f'{self.save_folder}/{seed}'
        os.makedirs(self.seed_folder, exist_ok=True)

    def split_masks(self, seed):
        self.train_mask, self.val_mask, self.test_mask = self.masks[seed]['train'], \
                                                         self.masks[seed]['val'], self.masks[seed]['test']

    def save_results(self, seed):
        self.seed_results[seed] = self.store_results
        with open(f'{self.save_folder}/seed_results.json', 'w+') as f:
            json.dump(self.seed_results, f)

        self.aggregated = self.aggregate_results()
        save_path = f'{self.save_folder}/aggregated_results.json'
        # with open(f'{self.seed_folder}/aggregated_results.json', 'w+') as f:
        #     json.dump(self.aggregated, f)

        if os.path.exists(save_path):
            # update model results
            with open(save_path, 'r') as f:
                metrics = json.load(f)
            metrics.update(self.aggregated)

            with open(save_path, 'w+') as f:
                json.dump(metrics, f)
        else:
            with open(save_path, 'w+') as f:
                json.dump(self.aggregated, f)

    def get_model_name(self, exp_name: str, algos: list):
        # get name of the model (for gnn-like models (eg. gat))
        # print("exp name:", exp_name)
        # print("algos:", algos)
        if 'name' in exp_name:
            # print("1")
            model_name = '-' + [param[4:] for param in exp_name.split('-') if param.startswith('name')][0]
        elif 'mixup' in exp_name:
            model_name = '-Mixup' + [param[5:] for param in exp_name.split('-') if param.startswith('mixup')][0]
        else:
            model_name = ''

        # get a model used a MLP (eg. MLP-GNN)
        if 'gnn' in exp_name and 'mlpTrue' in exp_name:
            model_name += '-MLP'

        # algo corresponds to type of the model (eg. gnn, resgnn, bgnn)
        for algo in algos:
            if algo in exp_name.split("-"):
                return  algo + model_name
        return 'unknown'

    def aggregate_results(self):
        algos = ['catboost', 'lightgbm', 'mlp', 'gnn', 'resgnn', 'resgnn_LI', 'bgnn', 'bgnn_v2', 'resgnnL', 'resgnnSVM', 'resgnnXG',
                 'emb-GBDT', 'ExcelFormer', 'trompt', 'fttransformer', 'tabnet', 'tabtransformer', 
                 'aggBGNN', 'aggBGNN_dnf', 'aggBGNN_dg', 'aggBGNN_v2', 'abgnn', 'xgboost', 'lightgbm', 'RandomForest']
        model_best_score = ddict(list)
        model_best_time = ddict(list)

        results = self.seed_results
        for seed in results:
            model_results_for_seed = ddict(list)
            for name, output in results[seed].items():
                model_name = self.get_model_name(name, algos=algos)
                if self.task == 'regression': # rmse metric
                    val_metric, test_metric, time = output[0][1], output[0][2], output[2]
                else: # accuracy metric
                    val_metric, test_metric, time = output[1][1], output[1][2], output[2]
                model_results_for_seed[model_name].append((val_metric, test_metric, time))

            for model_name, model_results in model_results_for_seed.items():
                if self.task == 'regression':
                    best_result = min(model_results) # rmse
                else:
                    best_result = max(model_results) # accuracy
                model_best_score[model_name].append(best_result[1])
                model_best_time[model_name].append(best_result[2])

        aggregated = dict()
        for model, scores in model_best_score.items():
            # print(model)
            # print(scores)
            # print("model best time:", model_best_time[model])
            aggregated[model] = (np.nanmean(scores), np.nanstd(scores),
                                 np.nanmean(model_best_time[model]), np.nanstd(model_best_time[model]))
        return aggregated
    
    def feature_vector(self):
        if self.task == 'classification':
            catboost_loss_function = 'MultiClass'
            catboost_object = CatBoostClassifier
        else:
            catboost_loss_function = 'RMSE'
            catboost_object = CatBoostRegressor

        model = catboost_object(iterations=100,
                                   depth=6,
                                   learning_rate=0.1,
                                   loss_function=catboost_loss_function,
                                   random_seed=0,
                                   nan_mode='Min',
                                   allow_const_label=True)
        
        X_train = self.X.iloc[self.train_mask]
        y_train = self.y.iloc[self.train_mask]

        model.fit(X_train, y_train, verbose=False)
        # prediction
        if self.task == 'classification':
            prediction = model.predict_proba(self.X)
        else:
            prediction = model.predict(self.X)
        # leaf index
        leaf_index = model.calc_leaf_indexes(self.X)
        return prediction, leaf_index
    
    def construct_graph(self, nf):
        warnings.filterwarnings('ignore')   # ignore dgl warnings
        graph = dgl.DGLGraph()
        if isinstance(nf, np.ndarray):
            nf = pd.DataFrame(nf)
        nodes = list(nf.index)
        graph.add_nodes(len(nodes))

        simul = cosine_similarity(nf.values, nf.values)
        np.fill_diagonal(simul, 0)
        top5 = np.argpartition(simul, -5)[:, -5:]

        src_node = []
        dst_node = []
        for i in nodes:
            for j in top5[i]:
                src_node.append(i)
                dst_node.append(j)
        
        graph.add_edges(src_node, dst_node)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        graph = graph.to(self.device)

        return graph
    
    def get_graph(self, input_folder):
        if os.path.exists(f'{input_folder}/graph.graphml'):
            networkx_graph = nx.read_graphml(f'{input_folder}/graph.graphml')
            networkx_graph = nx.relabel_nodes(networkx_graph, {str(i): i for i in range(len(networkx_graph))})
            self.graph = dgl.from_networkx(networkx_graph)
            self.graph = dgl.remove_self_loop(self.graph)
            self.graph = dgl.add_self_loop(self.graph)
            self.graph = self.graph.to(self.device)
        elif os.path.exists(f'{input_folder}/cat_features.txt'):
            # normalize
            encoded_X = BaseModel().encode_cat_features(self.X, self.y, self.cat_features, self.train_mask, 
                                                        self.val_mask, self.test_mask)
            encoded_X = BaseModel().normalize_features(encoded_X, self.train_mask, self.val_mask, self.test_mask)
            self.graph = self.construct_graph(encoded_X)
        else:
            self.graph = self.construct_graph(self.X)
        
        pred_matrix, leaf_index = self.feature_vector()
        self.graph_pred = self.construct_graph(pred_matrix)
        self.graph_leaf = self.construct_graph(leaf_index)

        

    def run(self, dataset: str, *args,
            save_folder: str = None,
            task: str = 'regression',
            repeat_exp: int = 1,
            max_seeds: int = 5,
            dataset_dir: str = None,
            config_dir: str = None
            ):
        start2run = time.time()
        self.repeat_exp = repeat_exp
        self.max_seeds = max_seeds
        print(dataset, args, task, repeat_exp, max_seeds, dataset_dir, config_dir)

        dataset_dir = Path(dataset_dir) if dataset_dir else Path(__file__).parent.parent / 'datasets' / 'huggingFace_sd' / dataset
        # dataset_dir = Path(dataset_dir) if dataset_dir else Path(__file__).parent.parent / 'datasets'
        config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent / 'configs' / 'model'
        print(dataset_dir, config_dir)

        self.task = task
        self.save_folder = save_folder
        # self.save_folder = Path('./results/huggingFace_sd') / dataset / f'{dataset}_s10' / save_folder
        self.get_input(dataset_dir, dataset)

        self.seed_results = dict()
        for ix, seed in enumerate(self.masks):
            # print("ix:", ix, "seed:", seed)
            print(f'{dataset} Seed {seed}')
            self.seed = seed
            
            self.create_save_folder(seed)
            self.split_masks(seed)

            # construct graph
            self.get_graph(dataset_dir/dataset)
            

            self.store_results = dict()
            for arg in args:
                if arg == 'all':
                    # self.run_one_model(config_fn=config_dir / 'mlp.yaml', model_name="mlp")
                    self.run_one_model(config_fn=config_dir / 'bgnn.yaml', model_name="bgnn")
                    self.run_one_model(config_fn=config_dir / 'bgnn_v2.yaml', model_name="bgnn_v2")
                    self.run_one_model(config_fn=config_dir / 'resgnn.yaml', model_name="resgnn")
                    self.run_one_model(config_fn=config_dir / 'resgnn_LI.yaml', model_name="resgnn_LI")
                    # self.run_one_model(config_fn=config_dir / 'resgnnL.yaml', model_name="resgnnL")
                    # self.run_one_model(config_fn=config_dir / 'resgnnXG.yaml', model_name="resgnnXG")
                    self.run_one_model(config_fn=config_dir / 'emb-GBDT.yaml', model_name="emb-GBDT")
                    self.run_one_model(config_fn=config_dir / 'catboost.yaml', model_name="catboost")
                    # self.run_one_model(config_fn=config_dir / 'ExcelFormer.yaml', model_name='ExcelFormer')
                    # self.run_one_model(config_fn=config_dir / 'trompt.yaml', model_name='trompt')
                    # self.run_one_model(config_fn=config_dir / 'tabnet.yaml', model_name='tabnet')
                    # self.run_one_model(config_fn=config_dir / 'tabtransformer.yaml', model_name='tabtransformer')
                    # self.run_one_model(config_fn=config_dir / 'fttransformer.yaml', model_name='fttransformer')
                    self.run_one_model(config_fn=config_dir / 'gnn.yaml', model_name="gnn")
                    self.run_one_model(config_fn=config_dir / 'xgboost.yaml', model_name="xgboost")
                    self.run_one_model(config_fn=config_dir / 'lightgbm.yaml', model_name="lightgbm")
                    # self.run_one_model(config_fn=config_dir / 'RandomForest.yaml', model_name="RandomForest")
                    self.run_one_model(config_fn=config_dir / 'aggBGNN.yaml', model_name="aggBGNN")
                    self.run_one_model(config_fn=config_dir / 'aggBGNN_dnf.yaml', model_name="aggBGNN_dnf")
                    # self.run_one_model(config_fn=config_dir / 'aggBGNN_dg.yaml', model_name="aggBGNN_dg")
                    self.run_one_model(config_fn=config_dir / 'aggBGNN_dg.yaml', model_name="aggBGNN_v2")
                    break
                elif arg == 'catboost':
                    self.run_one_model(config_fn=config_dir / 'catboost.yaml', model_name="catboost")
                elif arg == 'xgboost':
                    self.run_one_model(config_fn=config_dir / 'xgboost.yaml', model_name="xgboost")
                elif arg == 'lightgbm':
                    self.run_one_model(config_fn=config_dir / 'lightgbm.yaml', model_name="lightgbm")
                elif arg == 'mlp':
                    self.run_one_model(config_fn=config_dir / 'mlp.yaml', model_name="mlp")
                elif arg == 'gnn':
                    self.run_one_model(config_fn=config_dir / 'gnn.yaml', model_name="gnn")
                elif arg == 'resgnn':
                    self.run_one_model(config_fn=config_dir / 'resgnn.yaml', model_name="resgnn")
                elif arg == 'resgnn_LI':
                    self.run_one_model(config_fn=config_dir / 'resgnn.yaml', model_name="resgnn_LI")
                elif arg == 'resgnnL':
                    self.run_one_model(config_fn=config_dir / 'resgnnL.yaml', model_name="resgnnL")
                elif arg == 'resgnnSVM':
                    self.run_one_model(config_fn=config_dir / 'resgnnSVM.yaml', model_name="resgnnSVM")
                elif arg == 'resgnnXG':
                    self.run_one_model(config_fn=config_dir / 'resgnnXG.yaml', model_name="resgnnXG")
                elif arg == 'bgnn':
                    self.run_one_model(config_fn=config_dir / 'bgnn.yaml', model_name="bgnn")
                elif arg == 'bgnn_v2':
                    self.run_one_model(config_fn=config_dir / 'bgnn_v2.yaml', model_name="bgnn_v2")
                elif arg == 'emb-GBDT':
                    self.run_one_model(config_fn=config_dir / 'emb-GBDT.yaml', model_name="emb-GBDT")
                elif arg == 'transformers':
                    self.run_one_model(config_fn=config_dir / 'ExcelFormer.yaml', model_name='ExcelFormer')
                    self.run_one_model(config_fn=config_dir / 'trompt.yaml', model_name='trompt')
                    self.run_one_model(config_fn=config_dir / 'tabnet.yaml', model_name='tabnet')
                    self.run_one_model(config_fn=config_dir / 'tabtransformer.yaml', model_name='tabtransformer')
                    self.run_one_model(config_fn=config_dir / 'fttransformer.yaml', model_name='fttransformer')
                elif arg == 'aggBGNN':
                    self.run_one_model(config_fn=config_dir / 'aggBGNN.yaml', model_name="aggBGNN")
                    self.run_one_model(config_fn=config_dir / 'aggBGNN_dnf.yaml', model_name="aggBGNN_dnf")
                    self.run_one_model(config_fn=config_dir / 'aggBGNN_dg.yaml', model_name="aggBGNN_dg")
                    # self.run_one_model(config_fn=config_dir / 'aggBGNN_dg.yaml', model_name="aggBGNN_v2")
                elif arg == 'aggBGNN_dnf':
                    self.run_one_model(config_fn=config_dir / 'aggBGNN_dnf.yaml', model_name="aggBGNN_dnf")
                elif arg == 'aggBGNN_dg':
                    self.run_one_model(config_fn=config_dir / 'aggBGNN_dg.yaml', model_name="aggBGNN_dg")
                elif arg == 'aggBGNN_v2':
                    self.run_one_model(config_fn=config_dir / 'aggBGNN_dg.yaml', model_name="aggBGNN_v2")
                elif arg == 'abgnn':
                    self.run_one_model(config_fn=config_dir / 'abgnn.yaml', model_name="abgnn")
                elif arg == 'remain':
                    self.run_one_model(config_fn=config_dir / 'gnn.yaml', model_name="gnn")
                    self.run_one_model(config_fn=config_dir / 'ExcelFormer.yaml', model_name='ExcelFormer')
                    self.run_one_model(config_fn=config_dir / 'xgboost.yaml', model_name="xgboost")
                    self.run_one_model(config_fn=config_dir / 'lightgbm.yaml', model_name="lightgbm")
                    self.run_one_model(config_fn=config_dir / 'RandomForest.yaml', model_name="RandomForest")
                else:
                    # try:
                    config_fn = config_dir / f'{arg}.yaml'
                    self.run_one_model(config_fn=config_fn, model_name=arg)
                    # except:
                    #     raise ValueError("Model not found.")

            
            self.save_results(seed)
            if ix+1 >= max_seeds:
                break

        print(f'Finished {dataset}: {time.time() - start2run} sec.')

if __name__ == '__main__':
    fire.Fire(RunModel().run) 

    # sd_path = Path(__file__).parent.parent / 'datasets' / 'huggingFace_sd'
    # datasets_ls = [dir for dir in os.listdir(sd_path) if os.path.isdir(os.path.join(sd_path, dir))]
    # datasets_ls = sorted(datasets_ls)
    # print(len(datasets_ls))

    # for dataset in datasets_ls:
    #     RunModel().run(dataset, "aggBGNN_v2",
    #                    save_folder="aggBGNN",
    #                    task="classification")