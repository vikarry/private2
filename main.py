import pandas as pd
from dataset_engineering.cross_val import Set, WalkForwardFull, plot_tscv
import tensorflow as tf
from neural_nets.baseline import Params
import keras_tuner as kt
from finetuning.hypermodel import HyperModel
from finetuning.analyse import analyse_trials_opt
from criterion.metrics import Recall, F1Score
from dataset_engineering.source import compute_data
import random
import numpy as np

if __name__ == '__main__':

    data, dates, target, features = compute_data()

    days_ahead = 10
    n_steps = 60
    type_learn = 'classification'
    method_aggregate_target = 'return'
    config_hp = ['lr', 'dropout', 'l1', 'threshold', 'fix_outliers']  # , 'fix_outliers', 'scale_y', 'allow_missing_ts'

    if type_learn == 'regression':
        criterion = tf.keras.losses.MeanSquaredError
        metric = tf.keras.metrics.MeanSquaredError()
        metric.direction = 'min'
        output_dim = 1
    elif type_learn == 'classification':
        # Maximize recall in order to not miss the uncertainty time-steps
        criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metric = tf.keras.metrics.F1Score(average='weighted')
        metric.direction = 'max'
        output_dim = 2

    params = Params(input_dim=len(features), output_dim=output_dim, criterion=criterion, layer='LSTM',
                    activation='tanh', n_steps=n_steps, days_ahead=days_ahead,
                    n_hidden=1, hidden_dim=16, epochs=10, batch_size=16, patience_es=3, dropout=0.3,
                    method_aggregate_target=method_aggregate_target, type_learn=type_learn, metric=metric)
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    project_name = "_".join(config_hp)
    directory = 'finetune_results'

    cv = WalkForwardFull(data, start_test=pd.to_datetime('2019-12-31'), gap=days_ahead,
                         fix_start=False, val_size=3 * 20)
    project_name = f'flag_return_MOVEINDEX_ahead{days_ahead}_' + project_name

    max_trials = 40
    tuner = kt.BayesianOptimization(
        hypermodel=HyperModel(init_params=params, config=config_hp),
        objective=kt.Objective(f'val_f1_score', direction=metric.direction),
        max_trials=max_trials,
        directory=directory,
        overwrite=False,
        project_name=project_name
    )

    tuner.search(data=data, features=features, target=target, days_ahead=days_ahead, cv=cv, params=params,
                 project_name=project_name, train_dates=None, val_dates=None, test_dates=None, verbose=True)
    trials = analyse_trials_opt(max_trials, directory, project_name, metric='mae', sort_by='val_mean', ascending=True)