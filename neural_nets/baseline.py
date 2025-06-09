from dataclasses import dataclass
from functools import partial
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
from keras.src.layers.layer import Layer
from criterion.metrics import MAE, MSE
from keras.src import backend


@dataclass
class Params:
    seed: int = 42

    layer: str = "LSTM"  # "DNN" #"LSTM"
    activation: str = "tanh"  # "relu" "tanh"
    kernel_initializer: str = "glorot_uniform"  # "he_normal" "glorot_uniform"
    input_dim: int = None
    hidden_dim: int = 32
    n_hidden: int = 1
    dropout: float = 0.
    output_dim: int = 1

    lr: float = 0.0001
    l1: float = 0.
    opt: str = 'adam'
    criterion: MSE() = MSE()
    gradient_clip: float = None

    batch_size: int = 32
    n_steps: int = 100
    days_ahead: int = 100
    epochs: int = 100
    shuffle: bool = False
    patience_es: int = 5

    metric: MSE = MSE()
    fold: int = None

    attention_heads: int = 0
    mask: bool = True

    method_aggregate_target: str = 'return'
    type_learn: str = 'regression'

    threshold: float = 0.9


class MCDropout(Layer):

    def __init__(self, rate, mask=None, noise_shape=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= rate <= 1:
            raise ValueError(
                f"Invalid value received for argument "
                "`rate`. Expected a float value between 0 and 1. "
                f"Received: rate={rate}"
            )
        self.rate = rate
        self.mask = mask
        self.seed = seed
        self.noise_shape = noise_shape
        if rate > 0:
            self.seed_generator = backend.random.SeedGenerator(seed)
        self.supports_masking = True
        self.built = True

    def call(self, inputs, training=True):
        if self.mask:
            return backend.random.dropout(
                inputs,
                self.rate,
                noise_shape=self.noise_shape,
                seed=self.seed_generator,
            )
        return inputs * (1 - self.rate)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        config = {
            "rate": self.rate,
            "seed": self.seed,
            "noise_shape": self.noise_shape,
        }
        return {**base_config, **config}


class NN:
    """
    This class allows to compile a Neural Network
    """

    def __init__(self,
                 self,
                 params: Params
                 ):
        self.params = params

    def set_optimizer(self):
        if self.params.opt == 'adam':
            optimizer = optimizers.Adam(learning_rate=self.params.lr, clipvalue=self.params.gradient_clip)
        elif self.params.opt == 'sgd':
            optimizer = optimizers.SGD(learning_rate=self.params.lr, clipvalue=self.params.gradient_clip)
        else:
            raise ValueError('Only Adam and SGD optimizers have been implemented')
        return optimizer

    def set_dense(self):
        return partial(layers.Dense, activation=self.params.activation,
                       kernel_initializer=self.params.kernel_initializer,
                       kernel_regularizer=regularizers.l1(self.params.l1))

    def set_lstm(self):
        return partial(layers.LSTM, activation=self.params.activation,
                       kernel_initializer=self.params.kernel_initializer,
                       kernel_regularizer=regularizers.l1(self.params.l1))

    def set_gru(self):
        return partial(layers.GRU, activation=self.params.activation,
                       kernel_initializer=self.params.kernel_initializer,
                       kernel_regularizer=regularizers.l1(self.params.l1))

    def set_attention(self):
        return partial(layers.MultiHeadAttention, num_heads=self.params.attention_heads,
                       key_dim=self.params.hidden_dim,
                       kernel_initializer=self.params.kernel_initializer,
                       kernel_regularizer=regularizers.l1(self.params.l1))

    def set_rnn(self):
        if self.params.layer == 'LSTM':
            return self.set_lstm()
        elif self.params.layer == 'GRU':
            return self.set_gru()

    def set_dropout(self):
        return MCDropout(self.params.dropout, self.params.mask)

    def init(self):
        model = models.Sequential()

        if self.params.layer == 'DNN':
            model.add(layers.Input(shape=[1, self.params.input_dim]))
            model.add(self.set_dense()(self.params.hidden_dim))
            if self.params.attention_heads > 0: model.add(self.set_attention()(self.params.hidden_dim))
            model.add(self.set_dropout())
            for _ in range(self.params.n_hidden):
                model.add(self.set_dense()(self.params.hidden_dim))
                if self.params.attention_heads > 0: model.add(self.set_attention()(self.params.hidden_dim))
                model.add(self.set_dropout())

        elif self.params.layer in ['LSTM', 'GRU']:
            model.add(layers.Input(shape=[self.params.n_steps, self.params.input_dim]))
            model.add(self.set_rnn()(self.params.hidden_dim, return_sequences=True))
            if self.params.attention_heads > 0: model.add(self.set_attention()(self.params.hidden_dim))
            model.add(self.set_dropout())
            if self.params.n_hidden >= 2:
                for _ in range(1, self.params.n_hidden):
                    model.add(self.set_rnn()(self.params.hidden_dim, return_sequences=True))
                    if self.params.attention_heads > 0: model.add(self.set_attention()(self.params.hidden_dim))
                    model.add(self.set_dropout())
            model.add(self.set_rnn()(self.params.hidden_dim))

        model.add(layers.Dense(self.params.output_dim))
        # model.summary()
        return model