import numpy as np
import pandas as pd
from .baseline import Params, NN
from dataset_engineering.env import Env
from keras import callbacks
import os
import tensorflow as tf
from criterion.metrics import BinaryAccuracy, Recall, F1Score


class Model(NN):
    def __init__(self,
                 self,
                 train_env: Env = None,
                 val_env: Env = None,
                 params: Params = None,
                 project_name: str = None,
                 nn_hp=None,
                 verbose: bool = False,
                 vis_board: bool = False
                 ):
        super().__init__(params=params)
        self.train_env = train_env
        self.val_env = val_env
        self.project_name = project_name
        self.nn_hp = nn_hp
        self.verbose = verbose
        self.vis_board = vis_board
        self.set_name()
        self.set_dir()
        self.set_model()
        self.set_monitor()

    def set_name(self):
        params = {key: value for key, value in sorted(self.params.__dict__.items())}
        keys = self.project_name.split("_")
        self.name = "_".join([str(k) + '_' + str(round(v, 5)) for k, v in params.items() if k in keys])

    def set_dir(self):
        x = "logs" if self.nn_hp is None else "hp_logs"
        self.dirpath = f'{x}/{self.project_name}/{self.name}'
        if self.params.fold is not None: self.dirpath += f"/train{self.params.fold}"

    def set_model(self):
        if self.nn_hp is None:
            self.model = self.init()
        else:
            self.model = self.nn_hp
            self.nn_hp = True

    def compile(self):
        self.model.compile(optimizer=self.set_optimizer(), loss=self.params.criterion, metrics=[self.params.metric, BinaryAccuracy(), Recall()])

    def set_monitor(self):
        if self.val_env is not None:
            self.monitor = f'val_{self.params.metric.name}'
        else:
            self.monitor = f'{self.params.metric.name}'

    def checkpoint(self):
        return callbacks.ModelCheckpoint(monitor=self.monitor, mode=self.params.metric.direction,
                                        save_weights_only=True,
                                        filepath=f'{self.dirpath}_checkpoint.weights.h5',
                                        save_best_only=True)

    def earlystopping(self):
        return callbacks.EarlyStopping(monitor=self.monitor, mode=self.params.metric.direction,
                                      patience=self.params.patience_es, restore_best_weights=True)

    def scheduler(self):
        return callbacks.ReduceLROnPlateau(monitor=self.monitor, mode=self.params.metric.direction,
                                          factor=0.2, patience=int(self.params.patience_es * 0.6))

    def tensorboard(self):
        return callbacks.TensorBoard(self.dirpath)

    def find_data(self, env: Env):
        if self.params.type_learn == 'regression':
            return env.X, env.y
        else:
            return env.X, env.y_encod

    def init_data(self):
        X_train, y_train = self.find_data(self.train_env)
        if self.val_env is not None:
            X_val, y_val = self.find_data(self.val_env)
        else:
            X_val, y_val = None, None
        return X_train, y_train, X_val, y_val

    def fit(self, tune_callbacks=None):
        X_train, y_train, X_val, y_val = self.init_data()
        validation_data = None if self.val_env is None else (X_val, y_val)
        callbacks = [self.earlystopping(), self.scheduler(), self.checkpoint()]
        if self.vis_board: callbacks.append(self.tensorboard())
        if tune_callbacks is not None: callbacks.extend(tune_callbacks)
        if self.params.type_learn == 'classification':
            class_weight = self.train_env.cw
        else:
            class_weight = None
        return self.model.fit(X_train, y_train, batch_size=self.params.batch_size,
                             epochs=self.params.epochs, validation_data=validation_data,
                             callbacks=callbacks, shuffle=self.params.shuffle,
                             validation_batch_size=self.params.batch_size,
                             class_weight=class_weight, verbose=self.verbose)

    def load_weights(self, previous_fold: bool = False):
        if not previous_fold:
            self.model.load_weights(f'{self.dirpath}_checkpoint.weights.h5')
        else:
            dir = self.dirpath.replace(f'train{str(self.params.fold)}', f'train{str(self.params.fold-1)}')
            if self.verbose: print(f"directory weights: {dir}")
            self.model.load_weights(f'{dir}_checkpoint.weights.h5')

    def evaluate(self, env):
        X, y = self.find_data(self.params.layer, env)
        return self.model.evaluate(X, y, batch_size=self.params.batch_size, return_dict=True)

    def predict(self, X: np.array):
        return self.model.predict(X, batch_size=1, verbose=False)

    def process_predict(self, X, env):
        outputs = self.predict(X)
        if self.params.layer == "DNN":
            outputs = tf.squeeze(outputs, axis=2)
        if env.scale_target:
            outputs = env.scaler_y.inverse_transform(outputs)
        if self.params.type_learn == 'classification':
            probs = tf.nn.sigmoid(outputs).numpy()
            outputs = np.argmax(probs, axis=1)
        else:
            outputs = tf.squeeze(outputs).numpy()
        return pd.Series(outputs, index=env.dates[-len(outputs):])

    def infer_predict(self, env: Env, mode: str = None, save: bool = False, verbose: bool = True, n_samples=50):
        X, y = env.X, env.y
        predictions = pd.concat([list(map(lambda i: self.process_predict(X, env), range(n_samples)))], axis=1)
        if y is not None:
            if env.scale_target:
                y = self.reshape_raw_y(env, y)
            if self.params.type_learn == 'classification':
                self.params.metric._from_logits = False
                predictions = pd.Series([list(map(lambda i: predictions.iloc[i].value_counts().index[0], range(len(predictions))))], index=predictions.index)
                n_samples = 1
                predictions_expanded = self.expand_uncertainty(predictions)
                y_expanded = self.expand_uncertainty(pd.Series(y.squeeze())).values
            if n_samples == 1:
                if self.params.type_learn == 'classification':
                    metric = self.params.metric(y_expanded, predictions_expanded.values).numpy()
                else:
                    metric = self.params.metric(y, predictions.values).numpy()
            else:
                metrics = list(map(lambda i: self.params.metric(y, np.expand_dims(predictions[i].values, axis=1)).numpy(), range(n_samples)))
                metric = np.mean(metrics)
            if verbose: print(f'{mode}_{self.params.metric.name}=', round(metric, 6))
        else:
            mode = 'test'
        if save:
            self.save_predictions(predictions, mode)
        if self.params.type_learn == 'classification': self.params.metric._from_logits = True
        if mode == 'val':
            return predictions, pd.Series(y.squeeze(), index=predictions.index).to_frame('target'), metric
        else:
            return predictions

    @staticmethod
    def reshape_raw_y(env, y):
        raw_y = env.raw_y.values.reshape(-1, 1)
        if len(raw_y) > len(y):
            raw_y = raw_y[env.n_steps-1:]
        return raw_y

    def save_predictions(self, outputs, mode):
        filename = f'outputs/{self.project_name}/{mode}/{self.name}.csv'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        outputs.to_csv(filename)

    def expand_uncertainty(self, series):
        series = series.copy()
        list_dates = list(series.index)
        idx_flag_pred = series[series == 1].index
        idx_uncertainty = []
        for idx_flag in idx_flag_pred:
            start = list_dates.index(idx_flag) + 1
            end = start + self.params.days_ahead
            for i in range(start, end):
                try:
                    idx_uncertainty.append(list_dates[i])
                except IndexError:
                    pass
        idx_uncertainty = list(set(idx_uncertainty))
        idx_uncertainty = [x for x in idx_uncertainty if x not in idx_flag_pred]
        series.loc[idx_uncertainty] = 1
        return series