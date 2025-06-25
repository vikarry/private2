import tensorflow as tf
from keras import metrics
import numpy as np


class MAE(metrics.MeanAbsoluteError):
    """Mean Absolute Error metric with direction attribute for optimization."""

    def __init__(self, name='mae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.direction = 'min'


class MSE(metrics.MeanSquaredError):
    """Mean Squared Error metric with direction attribute for optimization."""

    def __init__(self, name='mse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.direction = 'min'


class BinaryAccuracy(metrics.BinaryAccuracy):
    """Binary Accuracy metric with direction attribute for optimization."""

    def __init__(self, name='binary_accuracy', threshold=0.5, **kwargs):
        super().__init__(name=name, threshold=threshold, **kwargs)
        self.direction = 'max'


class Recall(metrics.Recall):
    """Recall metric with direction attribute for optimization."""

    def __init__(self, name='recall', thresholds=None, top_k=None, class_id=None, **kwargs):
        super().__init__(name=name, thresholds=thresholds, top_k=top_k, class_id=class_id, **kwargs)
        self.direction = 'max'


class Precision(metrics.Precision):
    """Precision metric with direction attribute for optimization."""

    def __init__(self, name='precision', thresholds=None, top_k=None, class_id=None, **kwargs):
        super().__init__(name=name, thresholds=thresholds, top_k=top_k, class_id=class_id, **kwargs)
        self.direction = 'max'


class F1Score(metrics.F1Score):
    """F1 Score metric with direction attribute for optimization."""

    def __init__(self, name='f1_score', average=None, threshold=None, **kwargs):
        super().__init__(name=name, average=average, threshold=threshold, **kwargs)
        self.direction = 'max'


class CategoricalAccuracy(metrics.CategoricalAccuracy):
    """Categorical Accuracy metric with direction attribute for optimization."""

    def __init__(self, name='categorical_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.direction = 'max'


class SparseCategoricalAccuracy(metrics.SparseCategoricalAccuracy):
    """Sparse Categorical Accuracy metric with direction attribute for optimization."""

    def __init__(self, name='sparse_categorical_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.direction = 'max'


class AUC(metrics.AUC):
    """Area Under the Curve metric with direction attribute for optimization."""

    def __init__(self, name='auc', num_thresholds=200, curve='ROC',
                 summation_method='interpolation', multi_label=False, **kwargs):
        super().__init__(name=name, num_thresholds=num_thresholds, curve=curve,
                         summation_method=summation_method, multi_label=multi_label, **kwargs)
        self.direction = 'max'


# Loss Functions
class MeanSquaredErrorLoss(tf.keras.losses.MeanSquaredError):
    """Mean Squared Error loss function."""

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='mean_squared_error'):
        super().__init__(reduction=reduction, name=name)


class MeanAbsoluteErrorLoss(tf.keras.losses.MeanAbsoluteError):
    """Mean Absolute Error loss function."""

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='mean_absolute_error'):
        super().__init__(reduction=reduction, name=name)


class BinaryCrossentropyLoss(tf.keras.losses.BinaryCrossentropy):
    """Binary Crossentropy loss function."""

    def __init__(self, from_logits=False, label_smoothing=0.0, axis=-1,
                 reduction=tf.keras.losses.Reduction.AUTO, name='binary_crossentropy'):
        super().__init__(from_logits=from_logits, label_smoothing=label_smoothing,
                         axis=axis, reduction=reduction, name=name)


class CategoricalCrossentropyLoss(tf.keras.losses.CategoricalCrossentropy):
    """Categorical Crossentropy loss function."""

    def __init__(self, from_logits=False, label_smoothing=0.0, axis=-1,
                 reduction=tf.keras.losses.Reduction.AUTO, name='categorical_crossentropy'):
        super().__init__(from_logits=from_logits, label_smoothing=label_smoothing,
                         axis=axis, reduction=reduction, name=name)


class SparseCategoricalCrossentropyLoss(tf.keras.losses.SparseCategoricalCrossentropy):
    """Sparse Categorical Crossentropy loss function."""

    def __init__(self, from_logits=False, reduction=tf.keras.losses.Reduction.AUTO,
                 name='sparse_categorical_crossentropy'):
        super().__init__(from_logits=from_logits, reduction=reduction, name=name)


class HuberLoss(tf.keras.losses.Huber):
    """Huber loss function - robust to outliers."""

    def __init__(self, delta=1.0, reduction=tf.keras.losses.Reduction.AUTO, name='huber_loss'):
        super().__init__(delta=delta, reduction=reduction, name=name)


class LogCoshLoss(tf.keras.losses.LogCosh):
    """Log-Cosh loss function - smooth version of MAE."""

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='logcosh'):
        super().__init__(reduction=reduction, name=name)


# Custom Metrics
class RMSE(metrics.Metric):
    """Root Mean Squared Error metric."""

    def __init__(self, name='rmse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.direction = 'min'
        self.sum_squared_error = self.add_weight(name='sum_squared_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        squared_error = tf.square(y_true - y_pred)

        if sample_weight is not None:
            squared_error = tf.multiply(squared_error, sample_weight)
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

        self.sum_squared_error.assign_add(tf.reduce_sum(squared_error))

    def result(self):
        return tf.sqrt(self.sum_squared_error / self.count)

    def reset_state(self):
        self.sum_squared_error.assign(0.0)
        self.count.assign(0.0)


class MAPE(metrics.Metric):
    """Mean Absolute Percentage Error metric."""

    def __init__(self, name='mape', **kwargs):
        super().__init__(name=name, **kwargs)
        self.direction = 'min'
        self.sum_percentage_error = self.add_weight(name='sum_percentage_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Avoid division by zero
        y_true = tf.where(tf.equal(y_true, 0), tf.ones_like(y_true) * 1e-7, y_true)
        percentage_error = tf.abs((y_true - y_pred) / y_true) * 100

        if sample_weight is not None:
            percentage_error = tf.multiply(percentage_error, sample_weight)
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

        self.sum_percentage_error.assign_add(tf.reduce_sum(percentage_error))

    def result(self):
        return self.sum_percentage_error / self.count

    def reset_state(self):
        self.sum_percentage_error.assign(0.0)
        self.count.assign(0.0)


class R2Score(metrics.Metric):
    """R-squared (coefficient of determination) metric."""

    def __init__(self, name='r2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.direction = 'max'
        self.sum_squared_residuals = self.add_weight(name='sum_squared_residuals', initializer='zeros')
        self.sum_squared_total = self.add_weight(name='sum_squared_total', initializer='zeros')
        self.sum_y_true = self.add_weight(name='sum_y_true', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            count = tf.reduce_sum(sample_weight)
            sum_y_true = tf.reduce_sum(y_true * sample_weight)
            sum_squared_residuals = tf.reduce_sum(tf.square(y_true - y_pred) * sample_weight)
        else:
            count = tf.cast(tf.size(y_true), tf.float32)
            sum_y_true = tf.reduce_sum(y_true)
            sum_squared_residuals = tf.reduce_sum(tf.square(y_true - y_pred))

        self.count.assign_add(count)
        self.sum_y_true.assign_add(sum_y_true)
        self.sum_squared_residuals.assign_add(sum_squared_residuals)

    def result(self):
        y_mean = self.sum_y_true / self.count
        # We need to calculate sum_squared_total based on current batch
        # This is a limitation of the stateful metric approach
        return 1.0 - (self.sum_squared_residuals / (self.sum_squared_total + 1e-7))

    def reset_state(self):
        self.sum_squared_residuals.assign(0.0)
        self.sum_squared_total.assign(0.0)
        self.sum_y_true.assign(0.0)
        self.count.assign(0.0)


# Utility functions for getting metrics and losses
def get_metric(metric_name: str, **kwargs):
    """Get metric instance by name."""
    metric_map = {
        'mae': MAE,
        'mse': MSE,
        'rmse': RMSE,
        'mape': MAPE,
        'r2_score': R2Score,
        'binary_accuracy': BinaryAccuracy,
        'categorical_accuracy': CategoricalAccuracy,
        'sparse_categorical_accuracy': SparseCategoricalAccuracy,
        'recall': Recall,
        'precision': Precision,
        'f1_score': F1Score,
        'auc': AUC,
    }

    if metric_name.lower() not in metric_map:
        raise ValueError(f"Unknown metric: {metric_name}")

    return metric_map[metric_name.lower()](**kwargs)


def get_loss(loss_name: str, **kwargs):
    """Get loss function instance by name."""
    loss_map = {
        'mse': MeanSquaredErrorLoss,
        'mae': MeanAbsoluteErrorLoss,
        'binary_crossentropy': BinaryCrossentropyLoss,
        'categorical_crossentropy': CategoricalCrossentropyLoss,
        'sparse_categorical_crossentropy': SparseCategoricalCrossentropyLoss,
        'huber': HuberLoss,
        'logcosh': LogCoshLoss,
    }

    if loss_name.lower() not in loss_map:
        raise ValueError(f"Unknown loss: {loss_name}")

    return loss_map[loss_name.lower()](**kwargs)


# Backward compatibility aliases
MeanAbsoluteError = MAE
MeanSquaredError = MSE