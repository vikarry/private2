import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.mode.chained_assignment = None

plt.style.use('ggplot')
mpl.rcParams['savefig.dpi'] = 1000
mpl.rcParams['font.family'] = 'serif'


class Env:
    def __init__(self,
                 data: pd.DataFrame,
                 features: list = None,
                 start: int = None,
                 end: int = None,
                 target: str = 'spot_id_delta',
                 days_ahead: int = 5,
                 scaler_train_X: Union[MinMaxScaler, StandardScaler] = None,
                 knn_imputer: KNNImputer = KNNImputer(),
                 fix_outliers: bool = False,
                 scaler_outliers: StandardScaler = StandardScaler(),
                 add_transf: bool = False,
                 no_missing_timesteps: bool = False,
                 scale_target: bool = True,
                 scaler_train_y: Union[MinMaxScaler, StandardScaler] = None,
                 scaler_outliers_y: StandardScaler = StandardScaler(),
                 n_steps: int = 5,
                 type_scaler: str = 'normalizer',
                 method_aggregate_target: str = 'returns',
                 type_learn: str = 'regression',
                 oneh_encoder: OneHotEncoder = OneHotEncoder(sparse_output=False, dtype=int,
                                                             categories=[np.array([0, 1])]),
                 flag_threshold: float = None,
                 threshold: float = 0.95,
                 verbose: bool = False,
                 print: bool = False
                 ):

        self.data = data.copy()
        self.start = start
        self.end = end
        self.features = features
        self.days_ahead = days_ahead
        self.knn_imputer = knn_imputer
        self.fix_outliers = fix_outliers
        self.scaler_outliers = scaler_outliers
        self.add_transf = add_transf
        self.no_missing_timesteps = no_missing_timesteps
        self.scale_target = scale_target
        self.scaler_outliers_y = scaler_outliers_y
        self.n_steps = n_steps
        self.type_scaler = type_scaler
        self.method_aggregate_target = method_aggregate_target
        self.type_learn = type_learn
        self.oneh_encoder = oneh_encoder
        self.flag_threshold = flag_threshold
        self.threshold = threshold
        self.verbose = verbose
        self.print = print
        self.type_env = 'train' if scaler_train_X is None else 'infer'
        self.main(target, scaler_train_X, scaler_train_y)

    def compute_target(self, target):
        if target in self.data.columns:
            self.base_target = target
            self.target = f'{self.method_aggregate_target}_{target}_ahead_{self.days_ahead}'
            if self.method_aggregate_target == 'returns':
                self.data[self.target] = np.nan
                for i in range(len(self.data)):
                    try:
                        self.data[self.target].iloc[i] = self.data[target].iloc[i + self.days_ahead + 1] / \
                                                         self.data[target].iloc[i]
                    except IndexError:
                        pass
            elif self.method_aggregate_target == 'ma':
                self.data[self.target] = np.nan
                for i in range(len(self.data)):
                    try:
                        self.data[self.target].iloc[i] = self.data[target].iloc[i + 1: i + self.days_ahead + 1].mean()
                    except IndexError:
                        pass
            self.data.dropna(subset=[self.target], inplace=True)
        else:
            print("The target is not in the DataFrame, considered as test set")

    def index_data(self):
        self.data.index = pd.to_datetime(self.data.index)
        start_proxy = self.data.loc[self.start:].index[0]
        start_index = list(self.data.index).index(start_proxy)
        if 1 < self.n_steps < start_index:
            self.data = self.data.iloc[start_index - self.n_steps + 1:].loc[: self.end]
        else:
            self.data = self.data.loc[self.start: self.end]
        self.dates = self.data.index
        if self.verbose:
            print('*' * 50)
            print(f'Data starting date = {self.data.index[0]}')
            print(f'Data ending date = {self.data.index[-1]}')
        self.X = self.data[self.features].copy()
        if self.target in self.data.columns:
            if self.type_learn == 'classification':
                if self.flag_threshold is None:
                    self.flag_threshold = self.data[self.target].quantile(self.threshold)
                idx_flag = self.data[self.target][self.data[self.target] >= self.flag_threshold].index
                self.data[f'flag_{self.target}'] = 0
                self.data.loc[idx_flag, f'flag_{self.target}'] = 1
                self.target = f'flag_{self.target}'
                try:
                    c0, c1 = np.bincount(self.data[self.target])
                    w0 = (1 / c0) * (len(self.data)) / 2
                    w1 = (1 / c1) * (len(self.data)) / 2
                    self.cw = {0: w0, 1: w1}
                    self.pct_flag = self.data[f'{self.target}'].value_counts(normalize=True)[1] * 100
                except ValueError:
                    self.cw = None
                    self.pct_flag = None
            if self.print:
                ax = self.data[self.base_target].plot(legend=True, figsize=(20, 6), cmap='tab10',
                                                      title=f'Labeling uncertainty using {self.days_ahead}d pct change above'
                                                            f' {round(self.flag_threshold, 3) * 100}% \\n Pct of time-steps being'
                                                            f' uncertain = {round(self.pct_flag, 3)}%')
                for xc in idx_flag:
                    ax.axvline(x=xc, color="red", linestyle="--", alpha=0.05)
                plt.savefig(f'plots/{self.type_env}/y_with_uncertainty_labels.jpg')
                plt.close()
            self.y = self.data[self.target].copy()

        else:
            self.y = None


def info_data(self):
    if self.verbose: print('*' * 50)
    self.outliers = (np.abs((self.X - self.X.mean()) / self.X.std()) > 3)
    if self.y is not None: self.outliers_y = (np.abs((self.y - self.y.mean()) / self.y.std()) > 3)
    if self.verbose:
        print(f'Features number of outliers : \\n {self.outliers.sum()}')
        if self.y is not None: print(f'Target number of outliers : \\n {self.outliers_y.sum()}')
    self.missing_values = self.X.isnull().sum()
    if self.verbose:
        print('*' * 50)
        print(f'Missing Values: \\n {self.missing_values}')
        print('*' * 50)
    ts_with_delta = pd.concat([pd.Series(self.data.index.to_list()),
                               pd.Series(pd.to_datetime(self.data.index.to_list())).diff()], axis=1)
    self.missing_ts = ts_with_delta[1].value_counts().iloc[1:]
    corr = self.data.corr()[[self.base_target, self.target]].iloc[:-2]
    if self.verbose:
        print(f'Missing TimeSteps : \\n {self.missing_ts}')
        print('*' * 50)
        print(f'Correlation : \\n {corr}')
        print('*' * 50)
    if self.print:
        corr.plot.bar(figsize=(18, 14))
        plt.savefig(f'plots/{self.type_env}/raw_correlation.jpg')
        plt.close()
        self.X.plot(figsize=(20, 10), cmap='tab20')
        plt.savefig(f'plots/{self.type_env}/raw_X.jpg')
        plt.close()
        self.y.to_frame(self.target).plot(figsize=(20, 10), cmap='tab20')
        plt.savefig(f'plots/{self.type_env}/raw_y.jpg')
        plt.close()


@staticmethod
def init_scaler(scaler_train, type_scaler):
    if scaler_train is None:
        if type_scaler == "normalizer":
            return MinMaxScaler((-1, 1))
        else:
            return StandardScaler()
    else:
        return scaler_train


def use_scaler(self, scaler_train, scaler, data):
    if scaler_train is None:
        if self.verbose: print(f'Fit_transform {type(scaler)}')
        return scaler.fit_transform(data)
    else:
        if self.verbose: print(f'Transform {type(scaler)}')
        return scaler.transform(data)


def remove_outliers(self, scaler_train_X):
    cols = self.outliers.sum().index.to_list()
    zscore = pd.DataFrame(self.use_scaler(scaler_train_X, self.scaler_outliers, self.X),
                          index=self.X.index, columns=self.X.columns)

    for col in cols:
        if (zscore[col].abs() > 3).sum() > 0:
            if self.verbose: print(f'Removing outliers for {col}')
            serie_zscore = zscore[col]
            serie_zscore_bool = np.abs(serie_zscore) > 3
            serie_w_outliers = self.X[col][~serie_zscore_bool]
            if scaler_train_X is None:
                try:
                    max_no_outliers = load_object(f'train_scalers/max_no_outliers_{col}.pkl')
                    min_no_outliers = load_object(f'train_scalers/min_no_outliers_{col}.pkl')
                    max_no_outliers_new = np.max(serie_w_outliers)
                    min_no_outliers_new = np.min(serie_w_outliers)
                    if max_no_outliers_new > max_no_outliers:
                        dump_object(max_no_outliers_new, f'train_scalers/max_no_outliers_{col}.pkl')
                        max_no_outliers = max_no_outliers_new
                    if min_no_outliers_new < min_no_outliers:
                        dump_object(min_no_outliers_new, f'train_scalers/min_no_outliers_{col}.pkl')
                        min_no_outliers = min_no_outliers_new
                except FileNotFoundError:
                    max_no_outliers = np.max(serie_w_outliers)
                    min_no_outliers = np.min(serie_w_outliers)
                    dump_object(max_no_outliers, f'train_scalers/max_no_outliers_{col}.pkl')
                    dump_object(min_no_outliers, f'train_scalers/min_no_outliers_{col}.pkl')
            else:
                max_no_outliers = load_object(f'train_scalers/max_no_outliers_{col}.pkl')
                min_no_outliers = load_object(f'train_scalers/min_no_outliers_{col}.pkl')
            self.X.loc[:, col] = np.where(serie_zscore > 3, max_no_outliers, self.X[col])
            self.X.loc[:, col] = np.where(serie_zscore < -3, min_no_outliers, self.X[col])
    else:
        if scaler_train_X is None:
            try:
                max_no_outliers = load_object(f'train_scalers/max_no_outliers_{col}.pkl')
                min_no_outliers = load_object(f'train_scalers/min_no_outliers_{col}.pkl')
                max_no_outliers_new = np.max(self.X[col])
                min_no_outliers_new = np.min(self.X[col])
                if max_no_outliers_new > max_no_outliers:
                    dump_object(max_no_outliers_new, f'train_scalers/max_no_outliers_{col}.pkl')
                if min_no_outliers_new < min_no_outliers:
                    dump_object(min_no_outliers_new, f'train_scalers/min_no_outliers_{col}.pkl')
            except FileNotFoundError:
                max_no_outliers = np.max(self.X[col])
                min_no_outliers = np.min(self.X[col])
                dump_object(max_no_outliers, f'train_scalers/max_no_outliers_{col}.pkl')
                dump_object(min_no_outliers, f'train_scalers/min_no_outliers_{col}.pkl')


def remove_outliers_y(self, scaler_train_X):
    zscore = pd.DataFrame(self.use_scaler(scaler_train_X, self.scaler_outliers_y, self.y.to_frame()),
                          index=self.y.index, columns=[self.target]).T.squeeze()

    serie_zscore_bool = np.abs(zscore) > 3
    serie_w_outliers = self.y[~serie_zscore_bool]
    self.y = np.where(zscore > 3, np.max(serie_w_outliers), self.y)
    self.y = np.where(zscore < -3, np.min(serie_w_outliers), self.y)
    self.y = pd.Series(self.y, index=self.X.index)
    self.y.name = self.target


def form_data3d(self):
    X_array, y_array = self.X.values, self.y.values if self.y is not None else np.random.randn(len(self.X))
    batch_size = len(self.X) - self.n_steps + 1
    X_3d = np.zeros((batch_size, self.n_steps, len(self.features)))
    y_2d = y_array[self.n_steps - 1:].reshape(-1, 1)  # .ravel()
    for start_idx in range(batch_size):
        X_3d[start_idx] = X_array[start_idx: start_idx + self.n_steps]
    return X_3d, y_2d if self.y is not None else None


def main(self, target: str, scaler_train_X, scaler_train_y):
    train = True if scaler_train_X is None else False

    if train:
        try:
            self.scaler_X = load_object('train_scalers/scaler_X.pkl')
            self.knn_imputer = load_object('train_scalers/knn_imputer.pkl')
            self.scaler_outliers = load_object('train_scalers/scaler_outliers.pkl')
            self.scaler_outliers_y = load_object('train_scalers/scaler_outliers_y.pkl')
            self.scaler_y = load_object('train_scalers/scaler_y.pkl')
            self.oneh_encoder = load_object('train_scalers/oneh_encoder.pkl')
            self.flag_threshold = load_object('train_scalers/flag_threshold.pkl')
        except FileNotFoundError:
            pass

    self.compute_target(target)
    self.index_data()
    self.info_data()

    if self.y is not None: self.raw_y = self.y.copy()

    if self.fix_outliers and self.outliers.sum().sum() > 1:
        self.remove_outliers(scaler_train_X=scaler_train_X)
        if self.y is not None and self.type_learn == 'regression':
            self.remove_outliers_y(scaler_train_X=scaler_train_X)
        if self.print:
            self.X.plot(figsize=(20, 10), cmap='tab20')
            plt.savefig(f'plots/{self.type_env}/X_no_outliers.jpg')
            plt.close()
            self.y.to_frame(self.target).plot(figsize=(20, 10), cmap='tab20')
            plt.savefig(f'plots/{self.type_env}/y_no_outliers.jpg')
            plt.close()

    if self.missing_values.sum() > 0:
        self.perform_inputation(scaler_train_X=scaler_train_X)

    if self.add_transf:
        self.X = self.X ** 0.5

    self.scaler_X = self.init_scaler(scaler_train_X, self.type_scaler)
    self.X = pd.DataFrame(self.use_scaler(scaler_train_X, self.scaler_X, self.X),
                          index=self.X.index, columns=self.X.columns)
    if self.print:
        self.X.plot(figsize=(20, 10), cmap='tab20')
        plt.savefig(f'plots/{self.type_env}/X_scale.jpg')
        plt.close()

    if self.scale_target:
        self.scaler_y = self.init_scaler(scaler_train_y, self.type_scaler)
        if self.y is not None:
            self.y = pd.DataFrame(self.use_scaler(scaler_train_y, self.scaler_y, self.y.to_frame()),
                                  index=self.y.index, columns=[self.target]).T.squeeze()
        if self.print:
            self.y.to_frame(self.target).plot(figsize=(20, 10), cmap='tab20')
            plt.savefig(f'plots/{self.type_env}/y_scale_no_outliers.jpg')
            plt.close()
    else:
        self.scaler_y = None

    if self.no_missing_timesteps and len(self.missing_ts) >= 1:
        self.fill_missing_ts()

    if self.n_steps > 1:
        self.X, self.y = self.form_data3d()
    else:
        self.X = self.X.values.reshape(self.X.shape[0], 1, self.X.shape[1])
        self.y = self.y.values.reshape(-1, 1) if self.y is not None else None

    if self.type_learn == 'classification':
        self.y_encod = self.use_scaler(scaler_train_X, self.oneh_encoder, self.y)

    if self.verbose: print("*************************Processed*************************")

    if train:
        dump_object(self.scaler_X, 'train_scalers/scaler_X.pkl')
        dump_object(self.knn_imputer, 'train_scalers/knn_imputer.pkl')
        dump_object(self.scaler_outliers, 'train_scalers/scaler_outliers.pkl')
        dump_object(self.scaler_outliers_y, 'train_scalers/scaler_outliers_y.pkl')
        dump_object(self.scaler_y, 'train_scalers/scaler_y.pkl')
        dump_object(self.oneh_encoder, 'train_scalers/oneh_encoder.pkl')
        dump_object(self.flag_threshold, 'train_scalers/flag_threshold.pkl')


def dump_object(obj, file_path):
    os.makedirs('/'.join(file_path.split('/')[:-1]), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def load_object(file_path):
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    return obj