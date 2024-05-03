import gc
import platform
from typing import Union, Tuple

from sklearn.ensemble import VotingRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from tensorflow.keras import optimizers, activations, callbacks, regularizers, layers
from tensorflow.keras.models import Sequential

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if platform.platform().startswith('macOS') and platform.machine() == 'arm64':
    # Disable GPU acceleration on M1 Macs due to extremely slow performance
    # Seems like a bug in Apple's newest tf implementation
    # https://developer.apple.com/forums/thread/686098
    tf.config.set_visible_devices([], 'GPU')

    # Also disable meta-optimization, since TF can not get the CPU freq of M1
    _tf_exp_opts = tf.config.optimizer.get_experimental_options()
    _tf_exp_opts['disable_meta_optimizer'] = True
    tf.config.optimizer.set_experimental_options(_tf_exp_opts)
    del _tf_exp_opts


def get_activation(name: Union[str, None]):
    if name is None or name.lower() == 'none':
        return None
    try:
        return activations.deserialize(name)
    except ValueError:
        name_lower = name.lower()
        if name_lower in ['leakyrelu', 'leaky_relu', ]:
            return activations.LeakyReLU()
        raise ValueError()


def build_mlp_model(meta,
                    hidden_layer_sizes: Tuple[int] = (20,),
                    nonlin: Union[str, None] = 'leaky_relu',
                    nonlin_last: Union[str, None] = 'relu',
                    lr: float = 0.001,
                    loss_fn: str = 'mean_squared_error',
                    optimizer: str = 'Adam',
                    kernel: str = 'normal',
                    dropout_p: float = 0.0,
                    l1_coeff: float = 0.0,
                    l2_coeff: float = 0.0,
                    kernel_l1_coeff: float = None,
                    kernel_l2_coeff: float = None,
                    bias_l1_coeff: float = None,
                    bias_l2_coeff: float = None,
                    input_noise: float = 0.0,
                    **kwargs):
    model = Sequential()
    model.add(layers.Input(shape=meta['n_features_in_'], name=f'l0_input', ))
    if input_noise > 0.0:
        model.add(layers.GaussianNoise(input_noise, name=f'l0_noise', ))

    kernel_l1_coeff = l1_coeff if kernel_l1_coeff is None else kernel_l1_coeff
    kernel_l2_coeff = l2_coeff if kernel_l2_coeff is None else kernel_l2_coeff
    bias_l1_coeff = l1_coeff if bias_l1_coeff is None else bias_l1_coeff
    bias_l2_coeff = l2_coeff if bias_l2_coeff is None else bias_l2_coeff

    i = 1
    for dim in hidden_layer_sizes:
        model.add(layers.Dense(dim,
                               name=f'l{i}_hidden',
                               kernel_initializer=kernel,
                               activation=get_activation(nonlin),
                               kernel_regularizer=regularizers.L1L2(l1=kernel_l1_coeff, l2=kernel_l2_coeff),
                               bias_regularizer=regularizers.L1L2(l1=bias_l1_coeff, l2=bias_l2_coeff)
                               ))
        if dropout_p > 0.0:
            model.add(layers.Dropout(dropout_p, name=f'l{i}_dropout'))
        i += 1
    model.add(layers.Dense(meta['n_outputs_'],
                           name=f'l{i}_output',
                           kernel_initializer=kernel,
                           activation=get_activation(nonlin_last),
                           kernel_regularizer=regularizers.L1L2(l1=kernel_l1_coeff, l2=kernel_l2_coeff),
                           bias_regularizer=regularizers.L1L2(l1=bias_l1_coeff, l2=bias_l2_coeff)
                           ))

    opt = optimizers.get(optimizer)
    opt.learning_rate = lr

    model.compile(loss=loss_fn, optimizer=opt)
    return model


def build_lr_mlp_model(meta, n_max_nodes=20, n_layers=5, **kwargs):
    hidden_layer_sizes = []

    for i in range(0, n_layers):
        hidden_layer_sizes.append(n_max_nodes - int((n_max_nodes - meta['n_outputs_']) / n_layers * i))

    return build_mlp_model(meta, hidden_layer_sizes=tuple(hidden_layer_sizes), **kwargs)

def build_smlp_model(meta, n_top_nodes=10, n_bot_nodes=5, n_layers=5, linearity=1.0, **kwargs):
    if n_layers > 1:
        base = (n_top_nodes / n_bot_nodes) ** (1 / (n_layers - 1))
        slope = (n_top_nodes - n_bot_nodes) / (n_layers - 1)
        hidden_layer_sizes = []
        
        for i in range(0, n_layers):
            lr_n = n_bot_nodes + (slope * i)
            pw_n = n_bot_nodes * (base ** i)
            hidden_layer_sizes.append(round(linearity * lr_n + (1 - linearity) * pw_n))
    else:
        hidden_layer_sizes = [n_top_nodes, ]

    return build_mlp_model(meta, hidden_layer_sizes=tuple(reversed(hidden_layer_sizes)), **kwargs)

def plot_mlp_model(model, node_size=0.8, wspace=0.3, hspace=0.5,
                   input_cols=None, cmap='bwr', vmax=None, float_fmt='.2f'):
    if isinstance(model, KerasRegressor):
        model = model.model_
    assert isinstance(model, Sequential), f'Unknown model format: {model}'

    auto_vmax = False if vmax is not None else True

    weights = [v.numpy() for i, v in enumerate(model.trainable_variables) if not i % 2]
    bias = [v.numpy() for i, v in enumerate(model.trainable_variables) if i % 2]

    assert len(weights) == len(bias), ''
    n_layers = len(weights)
    n_inputs = weights[0].shape[0]
    n_nodes_per_layer = [w.shape[1] for w in weights]

    fig_w = sum(n_nodes_per_layer) * node_size * (1.0 + wspace)
    fig_h = sum([max(n_inputs, max(n_nodes_per_layer)), 1]) * node_size
    fig, axs = plt.subplots(2, n_layers,
                            gridspec_kw=dict(width_ratios=n_nodes_per_layer,
                                             height_ratios=[n_inputs, 1],
                                             hspace=hspace,
                                             wspace=wspace,
                                             ),
                            figsize=(fig_w, fig_h)
                            )

    for i in range(2 * n_layers):
        ax = axs[i % 2][i // 2]
        layer_no = i // 2
        if i % 2:
            ax.set_title(f'Layer {layer_no + 1}\nBias')
            ylabel = ''
            xlabel = f'⇩\nLayer {layer_no + 1} Output'
            v = bias[layer_no].reshape(1, -1)
            if auto_vmax:
                vmax = np.abs(v).max()
        else:
            ax.set_title(f'Layer {layer_no + 1}\nWeights')
            ylabel = ('Input\n' if layer_no == 0 else f'Layer {layer_no} Output\n') + r'$\times$'
            xlabel = ''
            v = weights[layer_no]
            if auto_vmax:
                vmax = np.abs(v).max()
        sns.heatmap(v, square=True, annot=True, fmt=float_fmt,
                    cmap=cmap, vmax=vmax, vmin=-vmax, cbar=True if auto_vmax else False,
                    ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if i == 0 and input_cols is not None:
            ax.set_yticks(np.arange(len(input_cols)) + 0.5)
            ax.set_yticklabels(input_cols, rotation=45)

    if not auto_vmax:
        fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-vmax, vmax=vmax),
                                           cmap=cmap),
                     ax=axs)

    return fig


class PlotMLPCallback(callbacks.Callback):
    def __init__(self, interval=1, file_prefix='MLP', dpi=None, ext='png', **kwargs):
        super(PlotMLPCallback, self).__init__()
        self.interval = interval
        self.file_prefix = file_prefix
        self.dpi = dpi
        self.ext = ext
        self.plot_kwargs = kwargs

    def save_plot(self, epoch):
        fig = plot_mlp_model(self.model, **self.plot_kwargs)
        fig.suptitle(f'epoch #{epoch:08d}')
        fig.savefig(f'{self.file_prefix}.{epoch:08d}.{self.ext}', dpi=self.dpi)
        plt.close(fig)
        del fig
        gc.collect()

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            self.save_plot(epoch)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            self.save_plot(epoch + 1)


class MLPNetwork(BaseEstimator, RegressorMixin):
    def __init__(self, name=None, input_norm=False, output_scale=1.0, dtype=tf.float32):
        self.name = name
        self.input_norm = input_norm
        self.output_scale = output_scale
        self.dtype = dtype

    def predict(self, X):
        check_is_fitted(self)
        X = tf.constant(X, dtype=self.dtype)
        if self.input_norm:
            X /= tf.math.reduce_sum(X, axis=1, keepdims=True)
        for w, b, f in self.layers_:
            X = f(X @ w + b)
        return X.numpy() * self.output_scale
    
    def fit(self, X, y):
        raise NotImplementedError('This class is designed for prediction with minimum overhead and does not implement fitting logic, please use keras instead.')
    
    def _initialize(self):
        self.layers_ = []

    @classmethod
    def from_keras(cls, model, **kwargs):
        net = cls(**kwargs)
        net._initialize()
        for klayer in model.layers:
            if isinstance(klayer, tf.keras.layers.Dense):
                net.layers_.append((tf.cast(klayer.kernel.value(), net.dtype), 
                                    tf.cast(klayer.bias.value(), net.dtype), 
                                    klayer.activation))
            elif isinstance(klayer, tf.keras.layers.Dropout):
                pass
            elif isinstance(klayer, tf.keras.layers.GaussianNoise):
                pass
            else:
                print(f'Ignoring unknown layer type: {klayer}')
        return net
    
    @classmethod
    def from_scikeras(cls, model, **kwargs):
        return cls.from_keras(model.model_, **kwargs)


class MOVotingRegressor(VotingRegressor):
    ''' Multi-output VotingRegressor'''
    def predict(self, X):
        check_is_fitted(self)

        pred = self._predict(X)
        avg_pred = np.average(self._predict(X), axis=-1, weights=self._weights_not_none)

        if len(pred.shape) > 0:
            avg_pred = avg_pred.T

        return avg_pred
    
    def predict_variance(self, X):
        check_is_fitted(self)

        pred = self._predict(X)
        var_pred = np.var(self._predict(X), axis=-1)

        if len(pred.shape) > 0:
            var_pred = var_pred.T

        return var_pred
