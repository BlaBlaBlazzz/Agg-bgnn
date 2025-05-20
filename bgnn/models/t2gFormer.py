import os
import math
import time
import json
import random
import argparse
import numpy as np
from pathlib import Path
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Optional, Union, cast, Dict, List, Tuple

from tqdm import tqdm
from collections import defaultdict as ddict
from .Base import BaseModel

from .t2gformer.bin import T2GFormer
from .t2gformer.lib import Transformations, build_dataset, prepare_tensors, DATA, make_optimizer



class t2gFormer(BaseModel):
    def __init__(self, task='classification', layers=1, d_ffn_factor=4/3, d_token=192, ffn_dropout=0.1, n_heads=8, residual_dropout=0.0,
                 attention_dropout=0.2, lr=0.001, col_lr=0.01):
        
        super(t2gFormer, self).__init__()
        self.task = task

        # model params
        self.n_layers = layers
        self.d_ffn_factor = d_ffn_factor
        self.d_token = d_token
        self.ffn_dropout = ffn_dropout
        self.n_heads = n_heads
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout

        # training params
        self.learning_rate = lr
        self.col_lr = col_lr
        
        self.prenormalization = True
        self.kv_compression = None
        self.kv_compression_sharing = None
        self.token_bias = True
        self.sym_weight = True
        self.sym_topology = False
        self.nsi = True
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    
    def __name__(self):
        return 'T2GFormer'

    # batch size settings for datasets in (Grinsztajn et al., 2022)
    def get_batch_size(self, n_features):
        if n_features <= 32:
            batch_size = 512
            val_batch_size = 8192
        elif n_features <= 100:
            batch_size = 128
            val_batch_size = 512
        elif n_features <= 1000:
            batch_size = 32
            val_batch_size = 64
        else:
            batch_size = 16
            val_batch_size = 16
        
        return batch_size, val_batch_size
    
    """Optimizers"""
    def needs_wd(self, name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

    # small learning rate for column embeddings 
    # for statble topology learning process
    def needs_small_lr(self, name):
        return any(x in name for x in ['.col_head', '.col_tail'])
    
    def model_evaluation(self, data_loader):
        metrics = {}
        total_counts = 0
        if self.task == "regression":
            metrics = {'loss':0, 'rmsle':0, 'mae':0, 'r2':0}
        elif self.task == 'classification':
            metrics = {'loss':0, 'accuracy':0}

        with torch.no_grad():
            # tensor frame of each batch
            for batch in data_loader:
                x_num, x_cat, y = (
                    (batch[0], None, batch[1])
                    if len(batch) == 2
                    else batch
                )

                pred = self.model(x_num, x_cat)
                total_counts += len(y)
                if self.task == 'regression':
                    metrics['loss'] += torch.sqrt(F.mse_loss(pred.view(-1), y.view(-1)) + 1e-8) * len(y)
                    metrics['rmsle'] += torch.sqrt(F.mse_loss(torch.log(pred + 1), torch.log(y + 1)).squeeze() + 1e-8) * len(y)
                    metrics['mae'] += F.l1_loss(pred, y) * len(y)
                    metrics['r2'] += torch.Tensor([(y == pred.max(1)[1]).sum().item() / y.shape[0]]) * len(y)
                elif self.task == 'classification':
                    metrics['loss'] += F.cross_entropy(pred, y)
                    pred_class = (torch.sigmoid(pred) > 0.5).long()
                    metrics['accuracy'] += torch.Tensor([(y == pred_class).sum().item()])
            
        for key in metrics.keys():
            metrics[key] /= total_counts
        
        return metrics
    
    def train_and_evaluate(self, train_loader, val_loader, test_loader, optimizer, metrics):
        loss = None
        loss = self.train_model(train_loader, optimizer)

        self.model.eval()
        train_results = self.model_evaluation(train_loader)
        val_results = self.model_evaluation(val_loader)
        test_results = self.model_evaluation(test_loader)
        for metric_name in train_results:
            metrics[metric_name].append((train_results[metric_name].detach().item(),
                                        val_results[metric_name].detach().item(),
                                        test_results[metric_name].detach().item()))
        
        return loss
        
    
    def train_model(self, train_loader, optimizer):
        self.model.train()
        loss_sum = 0
        total_counts = 0

        for iteration, batch in enumerate(train_loader):
            x_num, x_cat, y = (
                (batch[0], None, batch[1])
                if len(batch) == 2
                else batch
            )

            pred = self.model(x_num, x_cat)
            # print("pred:", pred)
            # print("y", y)
            if self.task == "regression":
                loss = torch.sqrt(F.mse_loss(pred.view(-1), y.view(-1)))
            elif self.task == "classification":
                loss = F.binary_cross_entropy_with_logits(pred, y)
            else:
                raise NotImplemented("Unknown task. Supported tasks: classification, regression.")
            

            optimizer.zero_grad()
            loss.backward()
            loss_sum += float(loss) * len(y)
            total_counts += len(y)
            optimizer.step()
        
        return loss_sum / total_counts

        
    

    def fit(self, X, y, train_mask, val_mask, test_mask, num_epochs,
            cat_features=None, patience=200, loss_fn="", metric_name="",
            normalization='quantile', logging_epochs=1):
        
        normalization = normalization if normalization != '__none__' else None
        transformation = Transformations(normalization=normalization)
        dataset = build_dataset(X, y, train_mask, val_mask, test_mask, cat_features, transformation)

        if dataset.X_num['train'].dtype == np.float64:
            dataset.X_num = {k: v.astype(np.float32) for k, v in dataset.X_num.items()}
        
        # out dim
        d_out = dataset.n_classes or 1
        X_num, X_cat, ys = prepare_tensors(dataset, device=self.device)

        # get batch_size
        n_features = dataset.n_features
        batch_size, val_batch_size = self.get_batch_size(n_features)

        num_workers = 0
        data_list = [X_num, ys] if X_cat is None else [X_num, X_cat, ys]
        train_dataset = TensorDataset(*(d['train'] for d in data_list))
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        val_dataset = TensorDataset(*(d['val'] for d in data_list))
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        test_dataset = TensorDataset(*(d['test'] for d in data_list))
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        n_num_features = dataset.n_num_features
        cardinalities = dataset.get_category_sizes('train')
        n_categories = len(cardinalities)
        cardinalities = None if n_categories == 0 else cardinalities

        kwargs = {
            'initialization': 'kaiming',
            'activation': 'reglu',
            'd_numerical': n_num_features,
            'categories': cardinalities,
            'd_out': d_out,
            'kv_compression': self.kv_compression,
            'kv_compression_sharing': self.kv_compression_sharing,
            'token_bias': self.token_bias,
            'sym_weight': self.sym_weight,
            'sym_topology': self.sym_topology,
            'nsi': self.nsi,

            'prenormalization': self.prenormalization,
            'n_layers': self.n_layers,
            'd_ffn_factor': self.d_ffn_factor,
            'd_token': self.d_token,
            'ffn_dropout': self.ffn_dropout,
            'n_heads': self.n_heads,
            'residual_dropout': self.residual_dropout,
            'attention_dropout': self.attention_dropout
            
        }

        # model initialization
        self.model = T2GFormer(**kwargs).to(self.device)

        for x in ['tokenizer', '.norm', '.bias']:
            assert any(x in a for a in (b[0] for b in self.model.named_parameters()))
        parameters_with_wd = [v for k, v in self.model.named_parameters() if self.needs_wd(k) and not self.needs_small_lr(k)]
        parameters_with_slr = [v for k, v in self.model.named_parameters() if self.needs_small_lr(k)]
        parameters_without_wd = [v for k, v in self.model.named_parameters() if not self.needs_wd(k)]

        optimizer = make_optimizer(
            "adamw",
            (
                [
                    {'params': parameters_with_wd},
                    {'params': parameters_with_slr, 'lr': self.col_lr, 'weight_decay': 0.0},
                    {'params': parameters_without_wd, 'weight_decay': 0.0},
                ]
            ),
            self.learning_rate,
            0.0
        )

        # n_epochs = 10000
        # report_frequency = len(ys['train']) // batch_size // 3
        # print(len(ys["train"]))
        # print("report frequency:", report_frequency)
        frozen_switch = True # whether to froze topology in later training phase

        if metric_name in ['r2', 'accuracy']:
            best_metric = [np.float32('-inf')] * 3  # for train/val/test
        else:
            best_metric = [np.float32('inf')] * 3  # for train/val/test
        metrics = ddict(list)
        best_val_epoch = 0
        epochs_since_last_best_metric = 0

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            start2epoch = time.time()
            
            loss = self.train_and_evaluate(train_loader, val_loader, test_loader,
                                            optimizer, metrics)
            self.log_epoch(pbar, metrics, epoch, loss, time.time()-start2epoch, logging_epochs,
                           metric_name=metric_name)
            
            best_metric, best_val_epoch, epochs_since_last_best_metric = \
                self.update_early_stopping(metrics, epoch, best_metric, best_val_epoch, epochs_since_last_best_metric, 
                                           metric_name, lower_better=(metric_name not in ['r2', 'accuracy']))
            if patience and epochs_since_last_best_metric > patience:
                if frozen_switch:
                    frozen_switch = False
                    _model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                    # fix topology in the following training phase
                    _model.froze_topology()
                    print(" >>> Froze FR-Graph topology")

                    epochs_since_last_best_metric = 0  # reset early stopping
                    continue
                break
        
        # if loss_fn:
        #     self.save_metrics(metrics, loss_fn)

        print('Best {} at iteration {}: {:.3f}/{:.3f}/{:.3f}'.format(metric_name, best_val_epoch, *best_metric))
        return metrics
            
        
    















