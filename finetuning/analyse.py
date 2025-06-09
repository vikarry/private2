import pandas as pd
import json
import os
from neural_nets.baseline import Params

pd.set_option('display.max_columns', None)


def analyse_trials_opt(max_trials: int, directory: str, project_name: str, sort_by: str = 'val_mean',
                       ascending: str = 'True', metric: str = 'mae'):
    """
    Analysing the tuning, finding the hyper-parameters leading to the maximal performance on validation set
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
            with open(directory + '/' + project_name + '/trial_' + str_num_trial + '/trial.json') as f:
                trial = json.load(f)
                found = True
        except FileNotFoundError:
            found = False

        if found:
            try:
                trial_hp = pd.DataFrame(trial['hyperparameters']['values'].values(),
                                        index=trial['hyperparameters']['values'].keys()).T
                trial_hp[f'val_{metric}'] = trial['metrics'][f'metrics'][f'val_{metric}']['observations'][0]['value'][0]
                list_trials.append(trial_hp)
            except:
                pass

    trials = pd.concat(list_trials, ignore_index=True)
    trials = trials.sort_values(by=sort_by, ascending=ascending)
    print(100 * '-')
    print(f'Trials configuration maximizing the {sort_by} of the cross validation: ')
    print(100 * '-')
    print(trials.to_string())
    return trials


def select_hp(trials: pd.DataFrame, config_hp: list, params, num: int = 0):
    trial = trials.iloc[num]
    for hp, value in trial.to_dict().items():
        if hp in config_hp:
            value = int(value) if value - round(value) == 0 else value
            params.__dict__[hp] = value
    return params


def get_predictions(params, config_hp, path_os, n_model=1, max_trials=20, type_hp='bayesian', metric='mae',
                    sort_by='val_mean', ascending=True, n_features_permodel=20):
    features_names = os.listdir(path_os)
    features_names = [f for f in features_names if len(f.split('_')) == n_features_permodel]
    test_predictions = []
    val_performances = []

    for features_name in features_names:
        project_name = f'{type_hp}/{features_name}/{params.criterion.name}__{params.layer}__' + "__".join(config_hp)
        directory = "my_finetuning"

        try:
            trials = analyse_trials_opt(max_trials, directory, project_name, metric=metric, sort_by=sort_by,
                                        ascending=ascending)
            for i in range(n_model):
                params_loc = select_hp(trials, config_hp, num=i, params=Params(**params.__dict__))
                params_loc = {key: value for key, value in sorted(params_loc.__dict__.items())}
                keys = project_name.split("__")
                name = "__".join([str(k) + '_' + str(round(v, 5)) for k, v in params_loc.items() if k in keys])
                filename = f'outputs/{project_name}/test/{name}.csv'
                pred = pd.read_csv(filename).set_index('Date')
                pred.columns = [features_name]
                test_predictions.append(pred)

                perf = trials.iloc[n_model][sort_by].to_frame().reset_index(drop=True)
                perf['features'] = features_name
                perf['params'] = 0
                perf['params'][0] = {k: v for k, v in params_loc.items() if k in keys}
                val_performances.append(perf)
        except:
            pass

    return pd.concat(val_performances), pd.concat(test_predictions, axis=1)


def select_hp_idxfeat(val_perf: pd.DataFrame, config_hp: list, params, num: int = 0):
    val_perf = val_perf.iloc[num]
    for hp, value in val_perf.params.items():
        if hp in config_hp:
            value = int(value) if value - round(value) == 0 else value
            params.__dict__[hp] = value

    idx_features = val_perf.features.split('_')
    idx_features = [int(idx) for idx in idx_features]
    return params, idx_features