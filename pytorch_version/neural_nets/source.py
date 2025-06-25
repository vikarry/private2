from neural_nets.baseline import NN, Params
from neural_nets.model import Model
from dataset_engineering.source import compute_envs
import pandas as pd
import numpy as np
import torch
import os
from matplotlib import pyplot as plt
import shutil


def train_infer_nn(data, features, target, days_ahead, train_dates, val_dates, test_dates, params, project_name,
                   fix_outliers: bool = False,
                   add_transf: bool = False, no_missing_ts: bool = False, scale_target: bool = False, nn_hp=None,
                   verbose: bool = False,
                   cv=None, i=None, plot=False, callbacks=None, count_step=None):
    """Train and infer with neural network"""

    # Create environments
    train_env, val_env, test_env = compute_envs(
        data, features=features, train_dates=train_dates, val_dates=val_dates,
        test_dates=test_dates, target=target, days_ahead=days_ahead,
        fix_outliers=fix_outliers, add_transf=add_transf,
        no_missing_timesteps=no_missing_ts, scale_target=scale_target,
        n_steps=params.n_steps,
        method_aggregate_target=params.method_aggregate_target,
        type_learn=params.type_learn, threshold=params.threshold
    )

    if cv is not None:
        params.fold = i

    # Create model
    model = Model(train_env, val_env, params, nn_hp=nn_hp, project_name=project_name, verbose=verbose)

    # Load previous weights if doing walk-forward validation
    if cv is not None and i > 1 and not cv.fix_start:
        try:
            model.load_weights(previous_fold=True)
        except:
            if verbose:
                print(f"Could not load weights from fold {params.fold - 1}")

    # Train model
    history = model.fit()

    # Validate
    if val_env is not None:
        if verbose:
            print("***********************Infer Validation Set***********************")
        preds_val, targets_val, val_metric = model.infer_predict(val_env, 'val', verbose=verbose, n_samples=50)

        if params.type_learn == 'regression':
            val_results = pd.concat([preds_val, targets_val], axis=1)
        else:
            val_results = pd.concat([preds_val.to_frame('pred'), targets_val], axis=1)
            val_results['raw_target'] = val_env.data[val_env.base_target].loc[preds_val.index]

        if plot:
            draw_plot(params, model, val_results, val_metric)
    else:
        preds_val, val_results, val_metric = None, None, None

    # Test predictions
    if verbose:
        print("***********************Pred Test Set***********************")
    preds_test = model.infer_predict(test_env, 'test', verbose=verbose, n_samples=1)

    return model, val_metric, preds_val, preds_test, val_results, callbacks, count_step


def train_infer_process(data, features, target, days_ahead, train_dates=None, val_dates=None, test_dates=None,
                        params=None, project_name=None, cv=None, fix_outliers: bool = False,
                        add_transf: bool = False, no_missing_ts: bool = False, scale_target: bool = False, nn_hp=None,
                        verbose: bool = False,
                        kwargs=None):
    """Process training and inference with optional cross-validation"""

    data = data.copy()

    if cv is None:
        # Single train-val-test split
        model, val_metric, preds_val, preds_test, results_val, callbacks, count_step = train_infer_nn(
            data, features, target, days_ahead, train_dates, val_dates, test_dates, params, project_name,
            fix_outliers, add_transf, no_missing_ts, scale_target, nn_hp, verbose=verbose
        )
    else:
        # Walk-forward cross-validation
        metrics_val = {f'val_{params.metric}': []}
        preds_val, preds_test, results_val = [], [], []

        for i, walk in cv.get_walks(False):
            train_dates, val_dates, test_dates = walk.train, walk.val, walk.test

            model, val_metric_f, preds_val_f, preds_test_f, results_val_f, callbacks, count_step = train_infer_nn(
                data.copy(), features, target, days_ahead,
                train_dates, val_dates, test_dates, params, project_name,
                fix_outliers, add_transf, no_missing_ts, scale_target,
                nn_hp, verbose=True, cv=cv, i=i
            )

            metrics_val[f'val_{params.metric}'].append(val_metric_f)
            preds_val.append(preds_val_f)
            preds_test.append(preds_test_f)
            results_val.append(results_val_f)

        # Average metrics across folds
        val_metric = np.mean(metrics_val[f'val_{params.metric}'])
        preds_val = pd.concat(preds_val)
        preds_test = pd.concat(preds_test)
        results_val = pd.concat(results_val)

    # Plot final results
    draw_plot(params, model, results_val, val_metric)

    # Clean up temporary files
    if os.path.exists('train_scalers'):
        shutil.rmtree('train_scalers')

    return model, val_metric, preds_val, preds_test, results_val


def draw_plot(params, model, val_results, metric):
    """Draw plots for results"""

    filename = f'plots/infer/predictions/{model.project_name}/{model.name}.jpg'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if params.type_learn == 'regression':
        # Regression plot
        plt.figure(figsize=(20, 10))
        plt.plot(val_results.index, val_results.iloc[:, 0], label='Predictions', alpha=0.8)
        plt.plot(val_results.index, val_results['target'], label='Target', alpha=0.8)
        plt.title(f"Target vs Forecasts - {params.metric}={round(metric, 3)}")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

    else:
        # Classification plot
        from sklearn.metrics import recall_score, f1_score
        recall = recall_score(val_results.target.values, val_results.pred.values)
        f1s = f1_score(val_results.target.values, val_results.pred.values)
        list_dates = list(val_results.index)

        fig, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=True)

        # First subplot - predictions
        ax1 = axes[0]
        ax1.plot(val_results['raw_target'], label='Raw Target', color='blue')
        idx_flag_pred = val_results[val_results['pred'] == 1].index

        for xc in idx_flag_pred:
            ax1.axvline(x=xc, color="steelblue", linestyle="--", alpha=0.5)

        ax1.set_title(f'Forecasting Uncertainty (high pct change) in the Next {params.days_ahead}d - '
                      f'Recall={recall:.3f}, F1={f1s:.3f}')

        # Add uncertainty periods
        idx_uncertainty = []
        for idx_flag in idx_flag_pred:
            start = list_dates.index(idx_flag) + 1
            end = start + params.days_ahead
            for i in range(start, end):
                try:
                    idx_uncertainty.append(list_dates[i])
                except IndexError:
                    pass

        idx_uncertainty = list(set(idx_uncertainty))
        idx_uncertainty = [x for x in idx_uncertainty if x not in idx_flag_pred]

        for xc in idx_uncertainty:
            ax1.axvline(x=xc, color="mediumseagreen", linestyle="--", alpha=0.5)

        # Second subplot - ground truth
        ax2 = axes[1]
        ax2.plot(val_results['raw_target'], label='Raw Target', color='blue')
        idx_flag_target = val_results[val_results['target'] == 1].index

        for xc in idx_flag_target:
            ax2.axvline(x=xc, color="orchid", linestyle="--", alpha=0.5)

        ax2.set_title("Ground Truth")

        # Add uncertainty periods for ground truth
        idx_uncertainty = []
        for idx_flag in idx_flag_target:
            start = list_dates.index(idx_flag) + 1
            end = start + params.days_ahead
            for i in range(start, end):
                try:
                    idx_uncertainty.append(list_dates[i])
                except IndexError:
                    pass

        idx_uncertainty = list(set(idx_uncertainty))
        idx_uncertainty = [x for x in idx_uncertainty if x not in idx_flag_target]

        for xc in idx_uncertainty:
            ax2.axvline(x=xc, color="mediumseagreen", linestyle="--", alpha=0.5)

        plt.tight_layout()

    plt.savefig(filename)
    plt.close()