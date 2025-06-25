import pandas as pd
from dataset_engineering.cross_val import Set, WalkForwardFull, plot_tscv
import torch
from neural_nets.baseline import Params
from finetuning.hypermodel import HyperModel, run_hyperparameter_search, analyze_optuna_trials
from dataset_engineering.source import compute_data
import random
import numpy as np
import os


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # Load data
    data, dates, target, features = compute_data()

    # Configuration
    days_ahead = 10
    n_steps = 60
    type_learn = 'classification'
    method_aggregate_target = 'return'
    config_hp = ['lr', 'dropout', 'l1', 'threshold', 'fix_outliers']

    # Set up parameters based on task type
    if type_learn == 'regression':
        criterion = 'mse'
        metric = 'mse'
        metric_direction = 'min'
        output_dim = 1
    elif type_learn == 'classification':
        criterion = 'bce'  # Binary Cross Entropy
        metric = 'f1_score'
        metric_direction = 'max'
        output_dim = 2

    # Initialize parameters
    params = Params(
        input_dim=len(features),
        output_dim=output_dim,
        criterion=criterion,
        layer='LSTM',
        activation='tanh',
        n_steps=n_steps,
        days_ahead=days_ahead,
        n_hidden=1,
        hidden_dim=16,
        epochs=10,
        batch_size=16,
        patience_es=3,
        dropout=0.3,
        method_aggregate_target=method_aggregate_target,
        type_learn=type_learn,
        metric=metric,
        metric_direction=metric_direction
    )

    # Set random seeds
    set_seed(params.seed)

    # Project naming
    project_name = "_".join(config_hp)
    directory = 'finetune_results'

    # Cross-validation setup
    cv = WalkForwardFull(
        data,
        start_test=pd.to_datetime('2019-12-31'),
        gap=days_ahead,
        fix_start=False,
        val_size=3 * 20
    )

    project_name = f'flag_return_MOVEINDEX_ahead{days_ahead}_' + project_name

    # Hyperparameter optimization
    max_trials = 40

    # Create hypermodel
    hypermodel = HyperModel(init_params=params, config=config_hp)

    # Run optimization
    print(f"Starting hyperparameter optimization with {max_trials} trials...")
    study = run_hyperparameter_search(
        hypermodel,
        data=data,
        features=features,
        target=target,
        days_ahead=days_ahead,
        cv=cv,
        params=params,
        project_name=project_name,
        max_trials=max_trials,
        n_jobs=1  # Set to -1 for parallel trials
    )

    # Analyze results
    print("\nBest trial:")
    print(f"Value: {study.best_value}")
    print(f"Params: {study.best_params}")

    # Get trials dataframe
    trials_df = analyze_optuna_trials(
        project_name,
        metric=metric,
        sort_by=f'val_{metric}',
        ascending=(metric_direction == 'min')
    )

    print("\nTop 5 trials:")
    print(trials_df.head())

    # Save results
    os.makedirs(f'{directory}/{project_name}', exist_ok=True)
    trials_df.to_csv(f'{directory}/{project_name}/trials_summary.csv', index=False)

    print(f"\nOptimization complete. Results saved to {directory}/{project_name}/")