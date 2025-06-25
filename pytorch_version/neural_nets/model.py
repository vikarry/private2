import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .baseline import Params, NN
from dataset_engineering.env import Env
import os
from typing import Optional, Tuple
from sklearn.metrics import f1_score, accuracy_score, recall_score
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stopping callback for PyTorch"""

    def __init__(self, patience=5, mode='min', delta=0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_score):
        score = -val_score if self.mode == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
            return True
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class Model:
    """PyTorch model wrapper for training and inference"""

    def __init__(self,
                 train_env: Env = None,
                 val_env: Env = None,
                 params: Params = None,
                 project_name: str = None,
                 nn_hp=None,
                 verbose: bool = False,
                 device: str = None):

        self.train_env = train_env
        self.val_env = val_env
        self.params = params
        self.project_name = project_name
        self.nn_hp = nn_hp
        self.verbose = verbose

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

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
        if self.params.fold is not None:
            self.dirpath += f"/train{self.params.fold}"

    def set_model(self):
        if self.nn_hp is None:
            self.model = NN(self.params).to(self.device)
        else:
            self.model = self.nn_hp.to(self.device)

        self.optimizer = self.model.get_optimizer()
        self.criterion = self.model.get_criterion()

    def set_monitor(self):
        if self.val_env is not None:
            self.monitor = f'val_{self.params.metric}'
        else:
            self.monitor = f'{self.params.metric}'

    def find_data(self, env: Env):
        if self.params.type_learn == 'regression':
            return env.X, env.y
        else:
            return env.X, env.y_encod

    def create_dataloader(self, X, y, shuffle=False):
        """Create PyTorch DataLoader from numpy arrays"""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y) if y.ndim > 1 else torch.FloatTensor(y.reshape(-1, 1))

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=shuffle)

        return dataloader

    def init_data(self):
        X_train, y_train = self.find_data(self.train_env)
        train_loader = self.create_dataloader(X_train, y_train, shuffle=self.params.shuffle)

        if self.val_env is not None:
            X_val, y_val = self.find_data(self.val_env)
            val_loader = self.create_dataloader(X_val, y_val, shuffle=False)
        else:
            val_loader = None

        return train_loader, val_loader

    def calculate_metric(self, y_true, y_pred):
        """Calculate the specified metric"""
        if self.params.type_learn == 'regression':
            if self.params.metric == 'mse':
                return np.mean((y_true - y_pred) ** 2)
            elif self.params.metric == 'mae':
                return np.mean(np.abs(y_true - y_pred))
        else:
            # Classification metrics
            y_pred_class = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
            y_true_class = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true

            if self.params.metric == 'f1_score':
                return f1_score(y_true_class, y_pred_class, average='weighted')
            elif self.params.metric == 'accuracy':
                return accuracy_score(y_true_class, y_pred_class)
            elif self.params.metric == 'recall':
                return recall_score(y_true_class, y_pred_class, average='weighted')

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)

            # Calculate loss
            loss = self.criterion(outputs, batch_y)

            # Add L1 regularization if specified
            if self.params.l1 > 0:
                loss = loss + self.model.l1_regularization()

            # Backward pass
            loss.backward()

            # Gradient clipping if specified
            if self.params.gradient_clip is not None:
                nn.utils.clip_grad_value_(self.model.parameters(), self.params.gradient_clip)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        all_outputs = np.vstack(all_outputs)
        all_targets = np.vstack(all_targets)

        metric_value = self.calculate_metric(all_targets, all_outputs)

        return total_loss / len(val_loader), metric_value

    def fit(self):
        """Train the model"""
        train_loader, val_loader = self.init_data()

        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=self.params.patience_es,
            mode=self.params.metric_direction
        )

        # Best model tracking
        best_val_metric = float('inf') if self.params.metric_direction == 'min' else float('-inf')
        best_model_state = None

        # Training loop
        for epoch in range(self.params.epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_loss, val_metric = self.validate(val_loader)

                if self.verbose:
                    print(f'Epoch {epoch + 1}/{self.params.epochs} - '
                          f'Train Loss: {train_loss:.4f} - '
                          f'Val Loss: {val_loss:.4f} - '
                          f'Val {self.params.metric}: {val_metric:.4f}')

                # Check if best model
                if early_stopping(val_metric):
                    best_val_metric = val_metric
                    best_model_state = self.model.state_dict().copy()

                if early_stopping.early_stop:
                    if self.verbose:
                        print(f'Early stopping at epoch {epoch + 1}')
                    break
            else:
                if self.verbose and epoch % 10 == 0:
                    print(f'Epoch {epoch + 1}/{self.params.epochs} - Train Loss: {train_loss:.4f}')

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Save checkpoint
        self.save_checkpoint()

        return {'best_val_metric': best_val_metric}

    def save_checkpoint(self):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(self.dirpath), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'params': self.params
        }
        torch.save(checkpoint, f'{self.dirpath}_checkpoint.pth')

    def load_weights(self, previous_fold: bool = False):
        """Load model weights"""
        if not previous_fold:
            checkpoint_path = f'{self.dirpath}_checkpoint.pth'
        else:
            dir_path = self.dirpath.replace(f'train{str(self.params.fold)}', f'train{str(self.params.fold - 1)}')
            checkpoint_path = f'{dir_path}_checkpoint.pth'
            if self.verbose:
                print(f"Loading weights from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def predict(self, X: np.array, n_samples: int = 1):
        """Make predictions with optional Monte Carlo sampling"""
        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        if n_samples == 1:
            with torch.no_grad():
                outputs = self.model(X_tensor)
                return outputs.cpu().numpy()
        else:
            # Monte Carlo predictions
            predictions = []
            for _ in range(n_samples):
                with torch.no_grad():
                    outputs = self.model(X_tensor)
                    predictions.append(outputs.cpu().numpy())
            return np.array(predictions)

    def process_predict(self, X, env, n_samples=1):
        """Process predictions and return as pandas Series"""
        outputs = self.predict(X, n_samples)

        if n_samples > 1:
            # Average Monte Carlo samples
            outputs = np.mean(outputs, axis=0)

        if self.params.layer == "DNN":
            outputs = np.squeeze(outputs, axis=1)

        if env.scale_target and self.params.type_learn == 'regression':
            outputs = env.scaler_y.inverse_transform(outputs.reshape(-1, 1)).squeeze()

        if self.params.type_learn == 'classification':
            # Apply sigmoid for probabilities
            probs = torch.sigmoid(torch.tensor(outputs)).numpy()
            outputs = np.argmax(probs, axis=1)
        else:
            outputs = outputs.squeeze()

        return pd.Series(outputs, index=env.dates[-len(outputs):])

    def infer_predict(self, env: Env, mode: str = None, save: bool = False,
                      verbose: bool = True, n_samples: int = 50):
        """Inference and prediction"""
        X, y = env.X, env.y

        if self.params.type_learn == 'classification':
            n_samples = 1  # For classification, we don't need MC sampling

        predictions = self.process_predict(X, env, n_samples)

        if y is not None:
            if env.scale_target and self.params.type_learn == 'regression':
                y = self.reshape_raw_y(env, y)

            if self.params.type_learn == 'classification':
                predictions_expanded = self.expand_uncertainty(predictions)
                y_expanded = self.expand_uncertainty(pd.Series(y.squeeze()))
                metric = self.calculate_metric(y_expanded.values, predictions_expanded.values)
            else:
                metric = self.calculate_metric(y, predictions.values.reshape(-1, 1))

            if verbose:
                print(f'{mode}_{self.params.metric}=', round(metric, 6))
        else:
            mode = 'test'
            metric = None

        if save:
            self.save_predictions(predictions, mode)

        if mode == 'val':
            return predictions, pd.Series(y.squeeze(), index=predictions.index).to_frame('target'), metric
        else:
            return predictions

    @staticmethod
    def reshape_raw_y(env, y):
        raw_y = env.raw_y.values.reshape(-1, 1)
        if len(raw_y) > len(y):
            raw_y = raw_y[env.n_steps - 1:]
        return raw_y

    def save_predictions(self, outputs, mode):
        filename = f'outputs/{self.project_name}/{mode}/{self.name}.csv'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        outputs.to_csv(filename)

    def expand_uncertainty(self, series):
        """Expand uncertainty predictions to cover future periods"""
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