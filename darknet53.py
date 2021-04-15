import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.regularizers import l2


class DarknetConvBlock(Layer):
    def __init__(self, out_channels, kernel_size, strides):
        super().__init__()
        self.conv2d = Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            use_bias=False,  # BatchNormalization --> no bias
            kernel_regularizer=l2(5e-4),
        )
        self.batchnorm = BatchNormalization()
        self.leakyrelu = LeakyReLU(alpha=0.1)
        
    def call(self, x):
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x
    
    
class DarknetResidualBlock(Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = DarknetConvBlock(out_channels=in_channels//2, kernel_size=1, strides=1)
        self.conv3x3 = DarknetConvBlock(out_channels=in_channels, kernel_size=3, strides=1)
        self.add = Add()
        
    def call(self, x):
        x_input = x
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        x = self.add([x_input, x])
        return x
    
    
class DarknetResidualBlockRepeat(Layer):
    def __init__(self, in_channels, n_repeats):
        super().__init__()
        self.resblock_repeated = Sequential()
        for _ in range(n_repeats):
            self.resblock_repeated.add(DarknetResidualBlock(in_channels=in_channels))
            
    def call(self, x):
        x = self.resblock_repeated(x)
        return x
    
    
class Darknet53(Model):
    def __init__(self, include_top=False):
        super().__init__()
        self.conv1 = DarknetConvBlock(out_channels=32, kernel_size=3, strides=1)
        self.conv1_1 = DarknetConvBlock(out_channels=64, kernel_size=3, strides=2)
        self.resblock_n1 = DarknetResidualBlockRepeat(in_channels=64, n_repeats=1)
        
        self.conv2 = DarknetConvBlock(out_channels=128, kernel_size=3, strides=2)
        self.resblock_n2 = DarknetResidualBlockRepeat(in_channels=128, n_repeats=2)
        
        self.conv3 = DarknetConvBlock(out_channels=256, kernel_size=3, strides=2)
        self.resblock_n8 = DarknetResidualBlockRepeat(in_channels=256, n_repeats=8)
        
        self.conv4 = DarknetConvBlock(out_channels=512, kernel_size=3, strides=2)
        self.resblock_n8_2 = DarknetResidualBlockRepeat(in_channels=512, n_repeats=8)
        
        self.conv5 = DarknetConvBlock(out_channels=1024, kernel_size=3, strides=2)
        self.resblock_n4 = DarknetResidualBlockRepeat(in_channels=1024, n_repeats=4)
        
        self.include_top = include_top
        if self.include_top:
            self.gap = GlobalAveragePooling2D()
            self.dense = Dense(1000, activation='softmax')
        
    def call(self, x):
        x = self.conv1(x)
        x = self.conv1_1(x)
        x = self.resblock_n1(x)
        
        x = self.conv2(x)
        x = self.resblock_n2(x)
        
        x = self.conv3(x)
        x = x256 = self.resblock_n8(x)
        
        x = self.conv4(x)
        x = x512 = self.resblock_n8_2(x)
        
        x = self.conv5(x)
        x1024 = self.resblock_n4(x)
        
        if self.include_top:
            x = self.gap(x1024)
            x = self.dense(x)
            return x
        else:
            return x1024, x512, x256
