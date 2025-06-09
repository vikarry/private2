import keras_tuner as kt
from neural_nets.baseline import NN
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from copy import deepcopy
from neural_nets.source import train_infer_nn, train_infer_process
import statsmodels.api as sm
from dataset_engineering.source import compute_envs


def update_params(params, hp):
    for key, value in hp.values.items():
        params.__dict__[key] = value
    return params


class HyperModel(kt.HyperModel):
    def __init__(self, init_params, config):
        self.init_params = init_params
        self.config = config

    def build(self, hp):
        params = deepcopy(self.init_params)
        if 'hidden_dim' in self.config:
            params.hidden_dim = hp.Choice("hidden_dim", [4, 8])
        if 'n_hidden' in self.config:
            params.n_hidden = hp.Int("n_hidden", 1, 2)
        if 'dropout' in self.config:
            params.dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, sampling="log")
        if 'lr' in self.config:
            params.lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        if 'l1' in self.config:
            params.l1reg = hp.Float("l1", min_value=0.0001, max_value=0.001, sampling="log")

        neural_net = NN(params=params)
        return neural_net.init()

    def fit(self, hp, model, data: pd.DataFrame, features: list, target: str, days_ahead: int, cv, params,
            project_name: str, train_dates=None, val_dates=None, test_dates=None, verbose=None, **kwargs):
        if 'batch_size' in self.config:
            params.batch_size = hp.Choice("batch_size", [4, 8, 32, 64])
        if 'n_steps' in self.config:
            params.n_steps = hp.Choice("n_steps", [20, 40, 60, 80])
        if 'threshold' in self.config:
            params.threshold = hp.Choice("threshold", [0.80, 0.83, 0.86, 0.90, 0.93])
            print(f'threshold: {params.threshold}')

        no_missing_ts = True if 'no_missing_ts' in self.config else False
        scale_target = True if 'scale_y' in self.config else False
        add_transf = True if 'add_transf' in self.config else False
        fix_outliers = True if 'fix_outliers' in self.config else False
        nn_hp = model
        params = update_params(params, hp)

        nn, val_metric, preds_val, preds_test, results_val = train_infer_process(data, features, target, days_ahead,
                                                                                 train_dates=train_dates,
                                                                                 val_dates=val_dates,
                                                                                 test_dates=test_dates,
                                                                                 params=params,
                                                                                 project_name=project_name, cv=cv,
                                                                                 fix_outliers=fix_outliers,
                                                                                 no_missing_ts=no_missing_ts,
                                                                                 add_transf=add_transf,
                                                                                 scale_target=scale_target, nn_hp=nn_hp,
                                                                                 kwargs=kwargs,
                                                                                 verbose=verbose)

        writer = tf.summary.create_file_writer(f'{"/".join(nn.dirpath.split("/")[:-1])}')
        with writer.as_default():
            dt = datetime.datetime.now()
            seq = int(dt.strftime("%Y%m%d%H%M"))[8:]
            tf.summary.scalar(f'val_{params.metric.name}', val_metric, step=seq)
            # tf.summary.scalar('val_mean_acc', val_mean_acc, step=seq)

        nn.save_predictions(results_val, 'val')
        nn.save_predictions(preds_test, 'test')

        return {f'val_{params.metric.name}': val_metric}