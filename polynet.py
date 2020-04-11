import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

INPUT_SHAPE = (11,11,40)
FILTER = 128
POLICY_OUT = 81
NUM_RES_NET_BLOCK = 3


class PolyNet:

    def __init__(self):
        def res_net_block(input_data, filters, conv_size):
            x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, input_data])
            x = layers.Activation('relu')(x)
            return x

        self.inputs = tf.keras.Input(shape=INPUT_SHAPE)
        x = layers.Conv2D(FILTER, 3, padding='same')(self.inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        for i in range(NUM_RES_NET_BLOCK):
            x = res_net_block(x, FILTER, 3)

        p = layers.Conv2D(2, 1, padding='same')(x)
        p = layers.BatchNormalization()(p)
        p = layers.Activation('relu')(p)
        p = layers.Conv2D(POLICY_OUT, 1, padding='same', name='policy')(p)

        v = layers.Conv2D(1, 1, padding='same')(x)
        v = layers.BatchNormalization()(v)
        v = layers.Activation('relu')(v)
        v = layers.Flatten()(v)
        v = layers.Dense(FILTER, activation='relu')(v)
        v = layers.Dense(1, activation='tanh', name='value')(v)

        self.res_net_model = tf.keras.Model(self.inputs, [v,p])


    def inference(self, x):
        if len(x.shape) == 3:
            x = x[np.newaxis, :, :, :]
        # follow channel-first convention for input and output
        x = np.einsum('bcij->bijc', x) 
        v, p = self.res_net_model(x)
        return np.squeeze(v.numpy()), np.einsum('bijc->bcij',p.numpy())

    def get_weights(self):
        return self.res_net_model.get_weights()


if __name__ == "__main__":
    input_test = np.random.rand(1,40,11,11)
    polynet = PolyNet()
    v, p = polynet.inference(input_test)
    assert p.shape[-1] == 11
    assert p.shape[-2] == 11
    print(polynet.get_weights())


