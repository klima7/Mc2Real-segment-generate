import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import keras
from keras import Model, Sequential, initializers
from keras.layers import Layer, Conv2D, LeakyReLU, Dropout


class SPADE(Layer):
    def __init__(self, filters: int, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.conv = Conv2D(128, 3, padding="same", activation="relu")
        self.conv_gamma = Conv2D(filters, 3, padding="same")
        self.conv_beta = Conv2D(filters, 3, padding="same")

    def build(self, input_shape):
        self.resize_shape = input_shape[1:3]

    def call(self, input_tensor, raw_mask):
        mask = tf.image.resize(raw_mask, self.resize_shape, method="nearest")
        x = self.conv(mask)
        gamma = self.conv_gamma(x)
        beta = self.conv_beta(x)
        mean, var = tf.nn.moments(input_tensor, axes=(0, 1, 2), keepdims=True)
        std = tf.sqrt(var + self.epsilon)
        normalized = (input_tensor - mean) / std
        output = gamma * normalized + beta
        return output

    def get_config(self):
        return {
            "epsilon": self.epsilon,
            "conv": self.conv,
            "conv_gamma": self.conv_gamma,
            "conv_beta": self.conv_beta
        }


class ResBlock(Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.spade_1 = SPADE(input_filter)
        self.spade_2 = SPADE(self.filters)
        self.conv_1 = Conv2D(self.filters, 3, padding="same")
        self.conv_2 = Conv2D(self.filters, 3, padding="same")
        self.leaky_relu = LeakyReLU(0.2)
        self.learned_skip = False

        if self.filters != input_filter:
            self.learned_skip = True
            self.spade_3 = SPADE(input_filter)
            self.conv_3 = Conv2D(self.filters, 3, padding="same")

    def call(self, input_tensor, mask):
        x = self.spade_1(input_tensor, mask)
        x = self.conv_1(self.leaky_relu(x))
        x = self.spade_2(x, mask)
        x = self.conv_2(self.leaky_relu(x))
        skip = (
            self.conv_3(self.leaky_relu(self.spade_3(input_tensor, mask)))
            if self.learned_skip
            else input_tensor
        )
        output = skip + x
        return output

    def get_config(self):
        return {"filters": self.filters}
    
    
class Downsample(Layer):
    def __init__(self,
                 channels: int,
                 kernels: int,
                 strides: int = 2,
                 apply_norm=True,
                 apply_activation=True,
                 apply_dropout=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.apply_norm = apply_norm
        self.apply_activation = apply_activation
        self.apply_dropout = apply_dropout

    def build(self, input_shape):
        self.block = Sequential([
            Conv2D(
                self.channels,
                self.kernels,
                strides=self.strides,
                padding="same",
                use_bias=False,
                kernel_initializer=initializers.GlorotNormal(),
            )])
        if self.apply_norm:
            self.block.add(tfa.layers.InstanceNormalization())
        if self.apply_activation:
            self.block.add(LeakyReLU(0.2))
        if self.apply_dropout:
            self.block.add(Dropout(0.5))

    def call(self, inputs):
        return self.block(inputs)

    def get_config(self):
        return {
            "channels": self.channels,
            "kernels": self.kernels,
            "strides": self.strides,
            "apply_norm": self.apply_norm,
            "apply_activation": self.apply_activation,
            "apply_dropout": self.apply_dropout,
        }


class GaussianSampler(Layer):
    def __init__(self, latent_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    def call(self, inputs):
        means, variance = inputs
        epsilon = tf.random.normal(
            shape=(tf.shape(means)[0], self.latent_dim), mean=0.0, stddev=1.0
        )
        samples = means + tf.exp(0.5 * variance) * epsilon
        return samples

    def get_config(self):
        return {"latent_dim": self.latent_dim}


class Predictor():
    def __init__(self, model_g_path: str, model_e_path: str = None) -> None:
        custom_objects = {
            'ResBlock': ResBlock,
            'Downsample': Downsample,
        }
        if model_e_path is not None:
            self.encoder: Model = keras.models.load_model(model_e_path, custom_objects=custom_objects)
            self.sampler = GaussianSampler(256)
        self.gen: Model = keras.models.load_model(
            model_g_path, custom_objects=custom_objects)

    def __call__(self, im: np.ndarray) -> np.ndarray:
        if len(im.shape) == 3:
            im = im[np.newaxis]
        z = tf.random.normal((im.shape[0], 256))
        tmp = self.gen.predict_on_batch([z, im])
        x = np.array((tmp + 1) * 127.5, np.uint8)
        return x

    def predict_reference(self, im: np.ndarray, reference_im: np.ndarray) -> np.ndarray:
        if len(im.shape) == 3:
            im = im[np.newaxis]
            reference_im = reference_im[np.newaxis]
        mean, variance = self.encoder(reference_im)
        z = self.sampler([mean, variance])
        x = np.array((self.gen.predict_on_batch([z, im]) + 1) * 127.5, np.uint8)
        return x
