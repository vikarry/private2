import pandas as pd
import json
import os
from neural_nets.baseline import Params
import torch

pd.set_option('display.max_columns', None)


def analyse_trials_opt(max_trials: int, directory: str, project_name: str, sort_by: str = 'val_mean',
                       ascending: bool = True, metric: str = 'mae'):
    """
    Analyzing the tuning, finding the hyper-parameters leading to the maximal performance on validation set
    """
    list_trials = []
    len_max = len(str(max_trials))

    for num_trial in range(max_trials):
        len_diff = len_max - len(str(num_trial))
        if len_diff != 0:
            str_num_trial = '0' * len_diff + str(num_trial)
        else:
            str_num_trial = str(num_trial)

        try:
            with open(f'{directory}/{project_name}/trial_{str_num_trial}/trial.json') as f:
                trial = json.load(f)
                found = True
        except FileNotFoundError:
            found = False

        if found and trial['state'] == 'TrialState.COMPLETE':
            try:
                # Extract hyperparameters
                trial_hp = pd.DataFrame([trial['params']])
                trial_hp[f'val_{metric}'] = trial['value']
                trial_hp['trial_id'] = trial['trial_id']
                list_trials.append(trial_hp)
            except:
                pass

    if list_trials:
        trials = pd.concat(list_trials, ignore_index=True)
        trials = trials.sort_values(by=sort_by, ascending=ascending)

        print(100 * '-')
        print(f'Trials configuration optimizing the {sort_by} of the cross validation: ')
        print(100 * '-')
        print(trials.to_string())

        return trials
    else:
        print("No completed trials found.")
        return pd.DataFrame()


def select_hp(trials: pd.DataFrame, config_hp: list, params: Params, num: int = 0):
    """Select hyperparameters from trial results"""
    if len(trials) == 0:
        return params

    trial = trials.iloc[num]

    for hp, value in trial.to_dict().items():
        if hp in config_hp and hp in params.__dict__:
            # Convert to appropriate type
            if isinstance(getattr(params, hp), int):
                value = int(value)
            elif isinstance(getattr(params, hp), float):
                value = float(value)

            setattr(params, hp, value)

    return params


def get_predictions(params, config_hp, path_os, n_model=1, max_trials=20, type_hp='optuna', metric='mae',
                    sort_by='val_mae', ascending=True, n_features_permodel=20):
    """Get predictions from trained models"""

    features_names = os.listdir(path_os) if os.path.exists(path_os) else []
    features_names = [f for f in features_names if len(f.split('_')) == n_features_permodel]

    test_predictions = []
    val_performances = []

    for features_name in features_names:
        project_name = f'{type_hp}/{features_name}/{params.criterion}__{params.layer}__' + "__".join(config_hp)
        directory = "finetune_results"

        try:
            trials = analyse_trials_opt(max_trials, directory, project_name, metric=metric, sort_by=sort_by,
                                        ascending=ascending)

            if len(trials) > 0:
                for i in range(min(n_model, len(trials))):
                    params_loc = select_hp(trials, config_hp, num=i, params=Params(**params.__dict__))
                    params_dict = {key: value for key, value in sorted(params_loc.__dict__.items())}

                    keys = project_name.split("__")
                    name = "__".join([str(k) + '_' + str(round(v, 5)) for k, v in params_dict.items() if k in keys])

                    filename = f'outputs/{project_name}/test/{name}.csv'

                    if os.path.exists(filename):
                        pred = pd.read_csv(filename).set_index('Date')
                        pred.columns = [features_name]
                        test_predictions.append(pred)

                        perf = trials.iloc[i][[sort_by]].to_frame().reset_index(drop=True)
                        perf['features'] = features_name
                        perf['params'] = [{k: v for k, v in params_dict.items() if k in keys}]
                        val_performances.append(perf)
        except Exception as e:
            print(f"Error processing {features_name}: {str(e)}")

    if val_performances and test_predictions:
        return pd.concat(val_performances), pd.concat(test_predictions, axis=1)
    else:
        return pd.DataFrame(), pd.DataFrame()


def select_hp_idxfeat(val_perf: pd.DataFrame, config_hp: list, params: Params, num: int = 0):
    """Select hyperparameters and feature indices"""
    if len(val_perf) == 0:
        return params, []

    val_perf_row = val_perf.iloc[num]

    # Update parameters
    for hp, value in val_perf_row.params.items():
        if hp in config_hp and hasattr(params, hp):
            if isinstance(getattr(params, hp), int):
                value = int(value)
            elif isinstance(getattr(params, hp), float):
                value = float(value)

            setattr(params, hp, value)

    # Extract feature indices
    idx_features = val_perf_row.features.split('_')
    idx_features = [int(idx) for idx in idx_features]

    return params, idx_features


def load_pytorch_model(checkpoint_path, params):
    """Load a PyTorch model from checkpoint"""
    from neural_nets.baseline import NN

    # Create model
    model = NN(params)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def compare_models(directory='finetune_results', project_names=None):
    """Compare performance across different model configurations"""

    if project_names is None:
        # Find all project directories
        project_names = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    comparison_data = []

    for project in project_names:
        best_trial_path = f'{directory}/{project}/best_trial.json'

        if os.path.exists(best_trial_path):
            with open(best_trial_path, 'r') as f:
                best_trial = json.load(f)

            comparison_data.append({
                'project': project,
                'best_value': best_trial['best_value'],
                'best_params': best_trial['best_params']
            })

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('best_value', ascending=True)

        print("\nModel Comparison:")
        print(comparison_df.to_string())

        return comparison_df
    else:
        print("No completed projects found for comparison.")
        return pd.DataFrame()