import tensorflow as tf
from tensorflow import image
import keras
from keras import layers


class DoubleConvLayer(layers.Layer):
    
    def __init__(self, n_filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")
        self.conv2 = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")
    
    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        return out
    
    
class DownsampleLayer(layers.Layer):
    
    def __init__(self, n_filters, **kwargs):
        super().__init__(**kwargs)
        self.conv = DoubleConvLayer(n_filters)
        self.pool = layers.MaxPool2D(2)
        self.dropout = layers.Dropout(0.3)
    
    def call(self, inputs):
        large = self.conv(inputs)
        small = self.pool(large)
        small = self.dropout(small)
        return large, small
    
    
class UpsampleLayer(layers.Layer):
    
    def __init__(self, n_filters, **kwargs):
        super().__init__(**kwargs)
        self.conv_trans = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")
        self.dropout = layers.Dropout(0.3)
        self.conv = DoubleConvLayer(n_filters)
    
    def call(self, inputs, conv_features):
        out = self.conv_trans(inputs)
        out  = layers.concatenate([out, conv_features])
        out = self.dropout(out)
        out = self.conv(out)
        return out
    
    
class UnetModel(keras.Model):
    
    INPUT_SIZE = 128
    
    def __init__(self, name='unet', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.down1 = DownsampleLayer(64)
        self.down2 = DownsampleLayer(128)
        self.down3 = DownsampleLayer(256)
        self.down4 = DownsampleLayer(512)
        
        self.middle_conv = DoubleConvLayer(1024)
        
        self.up1 = UpsampleLayer(512)
        self.up2 = UpsampleLayer(256)
        self.up3 = UpsampleLayer(128)
        self.up4 = UpsampleLayer(64)
        
        self.last_conv = layers.Conv2D(11, 1, padding="same", activation = "softmax")
        
        self.build(input_shape=(None, self.INPUT_SIZE, self.INPUT_SIZE, 3))
        
    def call(self, inputs):
        f1, p1 = self.down1(inputs)
        f2, p2 = self.down2(p1)
        f3, p3 = self.down3(p2)
        f4, p4 = self.down4(p3)
        
        bottleneck = self.middle_conv(p4)
        
        u1 = self.up1(bottleneck, f4)
        u2 = self.up2(u1, f3)
        u3 = self.up3(u2, f2)
        u4 = self.up4(u3, f1)
        
        outputs = self.last_conv(u4)
        return outputs

    def segment(self, inputs):
        _, height, width, _ = inputs.shape
        
        inputs = image.resize(
            inputs,
            (self.INPUT_SIZE, self.INPUT_SIZE),
            method=image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        
        preds = self.call(inputs)
        labels = tf.argmax(preds, axis=-1)[..., tf.newaxis]
        
        labels = image.resize(
            labels,
            (height, width),
            method=image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        
        return labels
