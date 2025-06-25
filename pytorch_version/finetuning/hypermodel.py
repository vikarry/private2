import optuna
from neural_nets.baseline import NN, Params
from neural_nets.model import Model
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from neural_nets.source import train_infer_nn, train_infer_process
from dataset_engineering.source import compute_envs
import os
import json


class HyperModel:
    """Hyperparameter optimization using Optuna instead of Keras Tuner"""

    def __init__(self, init_params, config):
        self.init_params = init_params
        self.config = config

    def objective(self, trial, data, features, target, days_ahead, cv, project_name,
                  train_dates=None, val_dates=None, test_dates=None, verbose=False):
        """Objective function for Optuna optimization"""

        # Create a copy of parameters
        params = deepcopy(self.init_params)

        # Sample hyperparameters
        if 'hidden_dim' in self.config:
            params.hidden_dim = trial.suggest_categorical('hidden_dim', [4, 8, 16, 32])

        if 'n_hidden' in self.config:
            params.n_hidden = trial.suggest_int('n_hidden', 1, 3)

        if 'dropout' in self.config:
            params.dropout = trial.suggest_float('dropout', 0.1, 0.5, log=True)

        if 'lr' in self.config:
            params.lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        if 'l1' in self.config:
            params.l1 = trial.suggest_float('l1', 0.0001, 0.001, log=True)

        if 'batch_size' in self.config:
            params.batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])

        if 'n_steps' in self.config:
            params.n_steps = trial.suggest_categorical('n_steps', [20, 40, 60, 80])

        if 'threshold' in self.config:
            params.threshold = trial.suggest_categorical('threshold', [0.80, 0.83, 0.86, 0.90, 0.93])

        # Additional flags
        no_missing_ts = 'no_missing_ts' in self.config
        scale_target = 'scale_y' in self.config
        add_transf = 'add_transf' in self.config
        fix_outliers = 'fix_outliers' in self.config

        # Train and evaluate
        try:
            _, val_metric, _, _, _ = train_infer_process(
                data, features, target, days_ahead,
                train_dates=train_dates,
                val_dates=val_dates,
                test_dates=test_dates,
                params=params,
                project_name=project_name,
                cv=cv,
                fix_outliers=fix_outliers,
                no_missing_ts=no_missing_ts,
                add_transf=add_transf,
                scale_target=scale_target,
                nn_hp=None,
                verbose=verbose
            )

            # Report intermediate value for pruning
            trial.report(val_metric, 0)

            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

            return val_metric

        except Exception as e:
            if verbose:
                print(f"Trial failed with error: {str(e)}")
            return float('inf') if params.metric_direction == 'min' else float('-inf')


def run_hyperparameter_search(hypermodel, data, features, target, days_ahead, cv, params,
                              project_name, max_trials=40, n_jobs=1):
    """Run hyperparameter optimization using Optuna"""

    # Create study
    direction = 'minimize' if params.metric_direction == 'min' else 'maximize'
    study = optuna.create_study(
        direction=direction,
        study_name=project_name,
        storage=f'sqlite:///optuna_{project_name}.db',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )

    # Optimize
    study.optimize(
        lambda trial: hypermodel.objective(
            trial, data, features, target, days_ahead, cv, project_name
        ),
        n_trials=max_trials,
        n_jobs=n_jobs
    )

    # Save results
    save_study_results(study, project_name)

    return study


def save_study_results(study, project_name):
    """Save Optuna study results in a format similar to Keras Tuner"""

    directory = f'finetune_results/{project_name}'
    os.makedirs(directory, exist_ok=True)

    # Save all trials
    for i, trial in enumerate(study.trials):
        trial_dir = f'{directory}/trial_{i:03d}'
        os.makedirs(trial_dir, exist_ok=True)

        trial_data = {
            'trial_id': i,
            'value': trial.value,
            'params': trial.params,
            'state': str(trial.state),
            'datetime_start': str(trial.datetime_start),
            'datetime_complete': str(trial.datetime_complete),
            'duration': str(trial.duration) if trial.duration else None,
            'metrics': {
                'val_metric': trial.value
            }
        }

        with open(f'{trial_dir}/trial.json', 'w') as f:
            json.dump(trial_data, f, indent=2)

    # Save best trial
    best_trial = study.best_trial
    best_params = {
        'best_trial_id': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params
    }

    with open(f'{directory}/best_trial.json', 'w') as f:
        json.dump(best_params, f, indent=2)


def analyze_optuna_trials(project_name, metric='mae', sort_by='value', ascending=True):
    """Analyze Optuna trials and return results as DataFrame"""

    directory = f'finetune_results/{project_name}'
    trials_data = []

    # Load all trial data
    trial_dirs = [d for d in os.listdir(directory) if d.startswith('trial_')]

    for trial_dir in sorted(trial_dirs):
        trial_path = f'{directory}/{trial_dir}/trial.json'

        if os.path.exists(trial_path):
            with open(trial_path, 'r') as f:
                trial_data = json.load(f)

            # Extract parameters and metrics
            trial_info = trial_data['params'].copy()
            trial_info[f'val_{metric}'] = trial_data['value']
            trial_info['trial_id'] = trial_data['trial_id']
            trial_info['state'] = trial_data['state']

            trials_data.append(trial_info)

    # Create DataFrame
    trials_df = pd.DataFrame(trials_data)

    # Sort by specified column
    if sort_by in trials_df.columns:
        trials_df = trials_df.sort_values(by=sort_by, ascending=ascending)

    return trials_df