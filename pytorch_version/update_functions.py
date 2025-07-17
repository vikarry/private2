def create_volatility_adjusted_flags(self, target, window=30, num_std=2):
    """Flag when movement exceeds N standard deviations of recent volatility"""
    # Calculate returns
    returns = self.data[target].pct_change()

    # Calculate rolling volatility
    rolling_std = returns.rolling(window=window).std()

    # Calculate future returns
    future_returns = self.data[target].shift(-self.days_ahead).pct_change(self.days_ahead)

    # Flag when future return exceeds N standard deviations
    threshold = num_std * rolling_std
    self.data['flag_unusual_movement'] = (np.abs(future_returns) > threshold).astype(int)


from dataclasses import dataclass
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from dataset_engineering.source import compute_envs

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 800


@dataclass
class Set:
    idx: int
    start: int
    end: int


@dataclass
class Walk:
    train: Set
    val: Set
    test: Set


class WalkForwardFull:
    def __init__(self,
                 data: pd.DataFrame,
                 start_test=pd.to_datetime('2024-07-01'),
                 val_size: int = 20,  # 20 * 3,
                 test_size: int = 20,
                 gap: int = 0,
                 fix_start: bool = False):
        self.val_size = val_size
        self.test_size = test_size
        self.gap = gap + 1
        self.dates = pd.to_datetime(data.index)
        self.train_size = list(self.dates).index(start_test) - (self.val_size + self.gap + 1)
        self.fix_start = fix_start
        self.count = 0

    def split(self):
        i, start_train, breakk = 0, 0, False
        while True:
            if not self.fix_start:
                if i == 0:
                    end_train = start_train + self.train_size
                elif i >= 1:
                    end_train += self.val_size + 1
            else:
                pass

            start_val = end_train + self.gap
            stop_val = start_val + self.val_size
            start_test = stop_val + self.gap
            stop_test = start_test + self.test_size

            if stop_test >= len(self.dates) - self.gap:
                stop_test = len(self.dates) - self.gap - 1
                breakk = True

            if not self.fix_start:
                if i == 1:
                    start_train = self.train_size + 1
                elif i > 1:
                    start_train += self.val_size + 1

            yield (start_train, end_train), (start_val, stop_val), (start_test, stop_test)
            i += 1
            self.count = i
            if breakk:
                break

    def get_walks(self, verbose: bool = True):
        idx = 0
        date = pd.DataFrame(index=self.dates)
        for train_index, valid_index, test_index in self.split():
            idx += 1
            if not self.fix_start:
                start_train = self.dates[train_index[0]] if idx == 1 else self.dates[train_index[0]]
            else:
                start_train = self.dates[train_index[0]]

            end_train = self.dates[train_index[-1]]
            start_valid = self.dates[valid_index[0]]
            end_valid = self.dates[valid_index[-1]]
            start_test = self.dates[test_index[0]]
            end_test = self.dates[test_index[-1]]

            walk = Walk(train=Set(idx=idx, start=start_train, end=end_train),
                        val=Set(idx=idx, start=start_valid, end=end_valid),
                        test=Set(idx=idx, start=start_test, end=end_test))

            if verbose:
                print('*' * 20, f'{idx}th walking forward', '*' * 20)
                print(f'Training: {walk.train.start} to {walk.train.end}, len={len(date.loc[start_train:end_train])}')
                print(f'Validation: {walk.val.start} to {walk.val.end}, len={len(date.loc[start_valid:end_valid])}')
                print(f'Test: {walk.test.start} to {walk.test.end}, len={len(date.loc[start_test:end_test])}')

            yield idx, walk


class YearlyValidation:
    """
    Method 1: For every year, train on one year of data and the next 3 months as validation data
    """

    def __init__(self,
                 data: pd.DataFrame,
                 start_year: int = None,
                 end_year: int = None,
                 val_months: int = 3):
        self.dates = pd.to_datetime(data.index)
        self.val_months = val_months
        self.count = 0

        # Determine year range
        if start_year is None:
            start_year = self.dates.min().year
        if end_year is None:
            end_year = self.dates.max().year

        self.start_year = start_year
        self.end_year = end_year

    def split(self):
        """Generate train/validation splits for each year"""
        for year in range(self.start_year, self.end_year):
            # Training period: full year
            train_start = pd.to_datetime(f'{year}-01-01')
            train_end = pd.to_datetime(f'{year}-12-31')

            # Validation period: next 3 months
            val_start = pd.to_datetime(f'{year + 1}-01-01')
            val_end = val_start + pd.DateOffset(months=self.val_months) - pd.Timedelta(days=1)

            # Convert to indices
            try:
                train_start_idx = self.dates.searchsorted(train_start)
                train_end_idx = self.dates.searchsorted(train_end, side='right') - 1
                val_start_idx = self.dates.searchsorted(val_start)
                val_end_idx = self.dates.searchsorted(val_end, side='right') - 1

                # Check if we have enough data
                if (train_end_idx < len(self.dates) and
                        val_end_idx < len(self.dates) and
                        train_start_idx < train_end_idx and
                        val_start_idx < val_end_idx):
                    yield (train_start_idx, train_end_idx), (val_start_idx, val_end_idx), (val_start_idx, val_end_idx)

            except (IndexError, ValueError):
                # Skip if dates are not available in the dataset
                continue

    def get_walks(self, verbose: bool = True):
        """Get walk objects for each year"""
        idx = 0
        date = pd.DataFrame(index=self.dates)

        for train_index, valid_index, test_index in self.split():
            idx += 1

            start_train = self.dates[train_index[0]]
            end_train = self.dates[train_index[1]]
            start_valid = self.dates[valid_index[0]]
            end_valid = self.dates[valid_index[1]]
            start_test = self.dates[test_index[0]]
            end_test = self.dates[test_index[1]]

            walk = Walk(train=Set(idx=idx, start=start_train, end=end_train),
                        val=Set(idx=idx, start=start_valid, end=end_valid),
                        test=Set(idx=idx, start=start_test, end=end_test))

            if verbose:
                print('*' * 20, f'{idx}th yearly validation', '*' * 20)
                print(f'Training: {walk.train.start} to {walk.train.end}, len={len(date.loc[start_train:end_train])}')
                print(f'Validation: {walk.val.start} to {walk.val.end}, len={len(date.loc[start_valid:end_valid])}')
                print(f'Test: {walk.test.start} to {walk.test.end}, len={len(date.loc[start_test:end_test])}')

            yield idx, walk
            self.count = idx


class ExpandingWindowValidation:
    """
    Method 2: First fold is 1 year of training data, second fold is 2 years of training data, etc.
    Validation data is always the next 3 months after training period.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 start_year: int = None,
                 max_folds: int = None,
                 val_months: int = 3):
        self.dates = pd.to_datetime(data.index)
        self.val_months = val_months
        self.count = 0

        # Determine starting year
        if start_year is None:
            start_year = self.dates.min().year
        self.start_year = start_year

        # Determine maximum possible folds
        end_year = self.dates.max().year
        max_possible_folds = end_year - start_year

        if max_folds is None:
            max_folds = max_possible_folds

        self.max_folds = min(max_folds, max_possible_folds)

    def split(self):
        """Generate expanding window splits"""
        for fold in range(1, self.max_folds + 1):
            # Training period: fold number of years starting from start_year
            train_start = pd.to_datetime(f'{self.start_year}-01-01')
            train_end = pd.to_datetime(f'{self.start_year + fold - 1}-12-31')

            # Validation period: next 3 months after training
            val_start = pd.to_datetime(f'{self.start_year + fold}-01-01')
            val_end = val_start + pd.DateOffset(months=self.val_months) - pd.Timedelta(days=1)

            # Convert to indices
            try:
                train_start_idx = self.dates.searchsorted(train_start)
                train_end_idx = self.dates.searchsorted(train_end, side='right') - 1
                val_start_idx = self.dates.searchsorted(val_start)
                val_end_idx = self.dates.searchsorted(val_end, side='right') - 1

                # Check if we have enough data
                if (train_end_idx < len(self.dates) and
                        val_end_idx < len(self.dates) and
                        train_start_idx < train_end_idx and
                        val_start_idx < val_end_idx):
                    yield (train_start_idx, train_end_idx), (val_start_idx, val_end_idx), (val_start_idx, val_end_idx)

            except (IndexError, ValueError):
                # Skip if dates are not available in the dataset
                continue

    def get_walks(self, verbose: bool = True):
        """Get walk objects for each expanding window"""
        idx = 0
        date = pd.DataFrame(index=self.dates)

        for train_index, valid_index, test_index in self.split():
            idx += 1

            start_train = self.dates[train_index[0]]
            end_train = self.dates[train_index[1]]
            start_valid = self.dates[valid_index[0]]
            end_valid = self.dates[valid_index[1]]
            start_test = self.dates[test_index[0]]
            end_test = self.dates[test_index[1]]

            walk = Walk(train=Set(idx=idx, start=start_train, end=end_train),
                        val=Set(idx=idx, start=start_valid, end=end_valid),
                        test=Set(idx=idx, start=start_test, end=end_test))

            if verbose:
                print('*' * 20, f'{idx}th expanding window', '*' * 20)
                print(f'Training: {walk.train.start} to {walk.train.end}, len={len(date.loc[start_train:end_train])}')
                print(f'Validation: {walk.val.start} to {walk.val.end}, len={len(date.loc[start_valid:end_valid])}')
                print(f'Test: {walk.test.start} to {walk.test.end}, len={len(date.loc[start_test:end_test])}')

            yield idx, walk
            self.count = idx


def plot_tscv(tscv, data, target, params, features):
    for i, walk in tscv.get_walks(False):
        pass

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.tick_params(labelleft=False, left=False, labelbottom=False)
    gs = fig.add_gridspec(i, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True, )

    # Determine CV type for title
    cv_type = type(tscv).__name__
    if cv_type == 'WalkForwardFull':
        title_suffix = f'Walk Forward CV using days_ahead={params.days_ahead} and gap={tscv.gap}'
    elif cv_type == 'YearlyValidation':
        title_suffix = f'Yearly CV using {tscv.val_months} months validation'
    elif cv_type == 'ExpandingWindowValidation':
        title_suffix = f'Expanding Window CV using {tscv.val_months} months validation'
    else:
        title_suffix = f'Time Series CV using days_ahead={params.days_ahead}'

    fig.suptitle(f'Time Series Cross Validation - {title_suffix}')

    for i, walk in tscv.get_walks(verbose=False):
        cv_data = pd.DataFrame(columns=['train', 'val', 'test'], index=pd.to_datetime(data.index))
        train_dates, val_dates, test_dates = walk.train, walk.val, walk.test

        train_env, val_env, test_env = compute_envs(data, features, train_dates, val_dates, test_dates,
                                                    target, params.days_ahead, scale_target=False,
                                                    n_steps=params.n_steps,
                                                    method_aggregate_target=params.method_aggregate_target)

        train_ts = pd.Series(train_env.y.squeeze(), index=train_env.dates[params.n_steps - 1:])
        val_ts = pd.Series(val_env.y.squeeze(), index=val_env.dates[params.n_steps - 1:])
        test_ts = pd.Series(test_env.y.squeeze(), index=test_env.dates[params.n_steps - 1:])

        cv_data['train'].loc[train_ts.index] = train_ts
        cv_data['val'].loc[val_ts.index] = val_ts
        cv_data['test'].loc[test_ts.index] = test_ts

        print('*' * 20, f'{i}th fold', '*' * 20)
        print(f'Training: {train_ts.index[0]} to {train_ts.index[-1]}, len={len(train_ts)}')
        print(f'Validation: {val_ts.index[0]} to {val_ts.index[-1]}, len={len(val_ts)}')
        print(f'Test: {test_ts.index[0]} to {test_ts.index[-1]}, len={len(test_ts)}')

        axs[i - 1].plot(cv_data)
        axs[i - 1].set_ylabel(f'Fold {i}')

    ax.legend([Patch(color='red'), Patch(color='blue'), Patch(color='purple')],
              ['Train', 'Val', 'Test'], loc=(1.02, .8))

    for ax in axs:
        ax.label_outer()

    plt.savefig(f'plots/tscv_{cv_type.lower()}.jpg')

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
    import os
    from typing import List, Dict, Any

    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    def plot_training_history(history, model_name: str, save_path: str = "plots/training"):
        """
        Plot training and validation loss, precision, and recall over epochs

        Args:
            history: Keras training history object
            model_name: Name of the model for plot titles
            save_path: Directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)

        # Get metrics from history
        epochs = range(1, len(history.history['loss']) + 1)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {model_name}', fontsize=16)

        # Plot 1: Loss
        axes[0, 0].plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history.history:
            axes[0, 0].plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Precision
        if 'precision' in history.history:
            axes[0, 1].plot(epochs, history.history['precision'], 'b-', label='Training Precision', linewidth=2)
            if 'val_precision' in history.history:
                axes[0, 1].plot(epochs, history.history['val_precision'], 'r-', label='Validation Precision',
                                linewidth=2)
            axes[0, 1].set_title('Model Precision')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Recall
        if 'recall' in history.history:
            axes[1, 0].plot(epochs, history.history['recall'], 'b-', label='Training Recall', linewidth=2)
            if 'val_recall' in history.history:
                axes[1, 0].plot(epochs, history.history['val_recall'], 'r-', label='Validation Recall', linewidth=2)
            axes[1, 0].set_title('Model Recall')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: F1 Score (if available)
        if 'f1_score' in history.history:
            axes[1, 1].plot(epochs, history.history['f1_score'], 'b-', label='Training F1', linewidth=2)
            if 'val_f1_score' in history.history:
                axes[1, 1].plot(epochs, history.history['val_f1_score'], 'r-', label='Validation F1', linewidth=2)
            axes[1, 1].set_title('Model F1 Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # If no F1, plot learning rate or other metric
            if 'lr' in history.history:
                axes[1, 1].plot(epochs, history.history['lr'], 'g-', label='Learning Rate', linewidth=2)
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].set_yscale('log')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_cross_validation_performance(cv_results: Dict[str, List[float]],
                                          cv_type: str = "Cross-Validation",
                                          save_path: str = "plots/cv_performance"):
        """
        Plot cross-validation performance across folds

        Args:
            cv_results: Dictionary with metric names as keys and lists of fold results as values
            cv_type: Type of cross-validation used
            save_path: Directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(cv_results)
        df['fold'] = range(1, len(df) + 1)

        # Number of metrics to plot
        metrics = [col for col in df.columns if col != 'fold']
        n_metrics = len(metrics)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{cv_type} Performance Across Folds', fontsize=16)
        axes = axes.flatten()

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for i, metric in enumerate(metrics[:4]):  # Plot up to 4 metrics
            if i < len(axes):
                ax = axes[i]

                # Box plot for distribution
                ax.boxplot(df[metric], positions=[1], widths=0.6, patch_artist=True,
                           boxprops=dict(facecolor=colors[i % len(colors)], alpha=0.7))

                # Line plot for trend across folds
                ax2 = ax.twinx()
                ax2.plot(df['fold'], df[metric], 'o-', color=colors[i % len(colors)],
                         linewidth=2, markersize=8, alpha=0.8)
                ax2.set_ylabel(f'{metric.title()} Value', color=colors[i % len(colors)])
                ax2.tick_params(axis='y', labelcolor=colors[i % len(colors)])

                ax.set_title(f'{metric.title()} Distribution')
                ax.set_xlabel('Fold')
                ax.set_ylabel('Distribution')
                ax.grid(True, alpha=0.3)

                # Add mean line
                mean_val = df[metric].mean()
                ax2.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7,
                            label=f'Mean: {mean_val:.4f}')
                ax2.legend()

        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{save_path}/{cv_type.lower()}_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create summary statistics table
        summary_stats = df[metrics].describe()
        print(f"\n{cv_type} Performance Summary:")
        print("=" * 50)
        print(summary_stats)

        return summary_stats

    def plot_confusion_matrix_heatmap(y_true, y_pred, class_names: List[str] = None,
                                      model_name: str = "Model", save_path: str = "plots/confusion"):
        """
        Plot confusion matrix as heatmap

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            model_name: Name of the model
            save_path: Directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Confusion Matrix - {model_name}', fontsize=16)

        # Plot 1: Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=class_names, yticklabels=class_names)
        axes[0].set_title('Raw Counts')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        # Plot 2: Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                    xticklabels=class_names, yticklabels=class_names)
        axes[1].set_title('Normalized')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig(f'{save_path}/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Print classification report
        print(f"\nClassification Report - {model_name}:")
        print("=" * 50)
        print(classification_report(y_true, y_pred, target_names=class_names))

    def plot_precision_recall_curve(y_true, y_pred_proba, model_name: str = "Model",
                                    save_path: str = "plots/pr_curve"):
        """
        Plot Precision-Recall curve

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities for positive class
            model_name: Name of the model
            save_path: Directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)

        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.axhline(y=y_true.mean(), color='red', linestyle='--', alpha=0.7,
                    label=f'Baseline (Random) = {y_true.mean():.4f}')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.tight_layout()
        plt.savefig(f'{save_path}/{model_name}_precision_recall.png', dpi=300, bbox_inches='tight')
        plt.close()

        return pr_auc

    def plot_roc_curve(y_true, y_pred_proba, model_name: str = "Model",
                       save_path: str = "plots/roc_curve"):
        """
        Plot ROC curve

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities for positive class
            model_name: Name of the model
            save_path: Directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random Classifier')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.tight_layout()
        plt.savefig(f'{save_path}/{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

        return roc_auc

    def plot_threshold_analysis(y_true, y_pred_proba, model_name: str = "Model",
                                save_path: str = "plots/threshold_analysis"):
        """
        Plot how precision, recall, and F1 change with threshold

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities for positive class
            model_name: Name of the model
            save_path: Directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)

        # Compute curves
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

        # Calculate F1 scores
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero

        # Find optimal threshold (max F1)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, precision[:-1], 'b-', linewidth=2, label='Precision')
        plt.plot(thresholds, recall[:-1], 'r-', linewidth=2, label='Recall')
        plt.plot(thresholds, f1_scores[:-1], 'g-', linewidth=2, label='F1 Score')

        # Mark optimal threshold
        plt.axvline(x=optimal_threshold, color='black', linestyle='--', alpha=0.7,
                    label=f'Optimal Threshold = {optimal_threshold:.4f}')

        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'Threshold Analysis - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.tight_layout()
        plt.savefig(f'{save_path}/{model_name}_threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        return optimal_threshold

    def plot_prediction_distribution(y_true, y_pred_proba, model_name: str = "Model",
                                     save_path: str = "plots/prediction_dist"):
        """
        Plot distribution of predicted probabilities for each class

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities for positive class
            model_name: Name of the model
            save_path: Directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)

        # Create DataFrame
        df = pd.DataFrame({
            'probability': y_pred_proba,
            'true_class': y_true
        })

        # Plot
        plt.figure(figsize=(12, 6))

        # Subplot 1: Histogram
        plt.subplot(1, 2, 1)
        plt.hist(df[df['true_class'] == 0]['probability'], bins=30, alpha=0.7,
                 label='Class 0', color='red', density=True)
        plt.hist(df[df['true_class'] == 1]['probability'], bins=30, alpha=0.7,
                 label='Class 1', color='blue', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Distribution of Predicted Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Box plot
        plt.subplot(1, 2, 2)
        df.boxplot(column='probability', by='true_class', ax=plt.gca())
        plt.title('Probability Distribution by True Class')
        plt.suptitle('')  # Remove default title
        plt.xlabel('True Class')
        plt.ylabel('Predicted Probability')

        plt.suptitle(f'Prediction Distribution - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{save_path}/{model_name}_prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_performance_dashboard(y_true, y_pred, y_pred_proba, model_name: str = "Model",
                                     class_names: List[str] = None,
                                     save_path: str = "plots/dashboard"):
        """
        Create a comprehensive performance dashboard

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities for positive class
            model_name: Name of the model
            class_names: Names of classes
            save_path: Directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)

        # Create comprehensive dashboard
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'Performance Dashboard - {model_name}', fontsize=20)

        # 1. Confusion Matrix
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # 2. ROC Curve
        plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Precision-Recall Curve
        plt.subplot(2, 3, 3)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, linewidth=2, label=f'PR (AUC = {pr_auc:.4f})')
        plt.axhline(y=y_true.mean(), color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. Prediction Distribution
        plt.subplot(2, 3, 4)
        df = pd.DataFrame({'probability': y_pred_proba, 'true_class': y_true})
        plt.hist(df[df['true_class'] == 0]['probability'], bins=20, alpha=0.7,
                 label='Class 0', color='red', density=True)
        plt.hist(df[df['true_class'] == 1]['probability'], bins=20, alpha=0.7,
                 label='Class 1', color='blue', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Prediction Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Threshold Analysis
        plt.subplot(2, 3, 5)
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)

        plt.plot(thresholds, precision[:-1], 'b-', linewidth=2, label='Precision')
        plt.plot(thresholds, recall[:-1], 'r-', linewidth=2, label='Recall')
        plt.plot(thresholds, f1_scores[:-1], 'g-', linewidth=2, label='F1 Score')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 6. Metrics Summary
        plt.subplot(2, 3, 6)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1 Score': f1_score(y_true, y_pred),
            'ROC AUC': roc_auc,
            'PR AUC': pr_auc
        }

        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        bars = plt.bar(metric_names, metric_values,
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        plt.title('Performance Metrics Summary')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.ylim([0, 1])

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{save_path}/{model_name}_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

        return metrics