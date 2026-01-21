import tensorflow as tf

class SharedEncoderKeras(tf.keras.Model):
    def __init__(self, output_size=32, name='SharedEncoderKeras', leaky_relu_alpha=0.01, **kwargs):
        super(SharedEncoderKeras, self).__init__(name=name, **kwargs)
        self._output_size = output_size

        # REMOVE the Flatten layer from __init__
        # self._flat_layer = tf.keras.layers.Flatten()

        self._dense_layers_list = [
            tf.keras.layers.Dense(128, name=f'{name}_dense_1'),
            tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha),
            tf.keras.layers.Dense(128, name=f'{name}_dense_2'),
            tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha),
            tf.keras.layers.Dense(self._output_size, activation=None, name=f'{name}_output')
        ]

    def call(self, observation, training=False):
        x = observation

        # REMOVE the call to the Flatten layer
        # x = self._flat_layer(observation, training=training)

        for layer in self._dense_layers_list:
            x = layer(x, training=training)
        return x

    def get_config(self):
        config = super(SharedEncoderKeras, self).get_config()
        config.update({'output_size': self._output_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TaskSpecificRacingEncoderKeras(tf.keras.Model):
    def __init__(self, output_size=32, name='TaskSpecificRacingEncoderKeras', leaky_relu_alpha=0.01, **kwargs):
        super(TaskSpecificRacingEncoderKeras, self).__init__(name=name, **kwargs)
        self._output_size = output_size
        self._dense_layers_list = [
            tf.keras.layers.Dense(128, name=f'{name}_dense_1'),
            tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha),
            tf.keras.layers.Dense(128, name=f'{name}_dense_2'),
            tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha),
            tf.keras.layers.Dense(self._output_size, activation=None, name=f'{name}_output') # Linear output
        ]

    def call(self, observation, training=False):
        # The input 'observation' is now the starting point.
        x = observation

        # REMOVE the call to the Flatten layer
        # x = self._flat_layer(observation, training=training)

        for layer in self._dense_layers_list:
            x = layer(x, training=training)
        return x

    def get_config(self):
        config = super(TaskSpecificRacingEncoderKeras, self).get_config()
        config.update({'output_size': self._output_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



class TaskSpecificStabilizationEncoderKeras(tf.keras.Model):
    """
    A simple encoder for the stabilization task's specific observations.
    Input: [acc_x, acc_y, acc_z, target_z, ...zeros]
    """
    def __init__(self, output_size=32, name='TaskSpecificStabilizationEncoderKeras', leaky_relu_alpha=0.01, **kwargs):
        super(TaskSpecificStabilizationEncoderKeras, self).__init__(name=name, **kwargs)
        self._output_size = output_size
        self._dense_layers_list = [
            tf.keras.layers.Dense(128, name=f'{name}_dense_1'),
            tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha),
            tf.keras.layers.Dense(128, name=f'{name}_dense_2'),
            tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha),
            tf.keras.layers.Dense(self._output_size, activation=None, name=f'{name}_output') # Linear output
        ]

    def call(self, observation, training=False):
        # REMOVE all slicing and conditional logic from here.
        # The network now assumes it receives the correct 4-element input.
        x = observation
        for layer in self._dense_layers_list:
            x = layer(x, training=training)
        return x

    def get_config(self):
        # BUG FIX: Corrected super() call to use its own class name
        config = super(TaskSpecificStabilizationEncoderKeras, self).get_config()
        config.update({'output_size': self._output_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)