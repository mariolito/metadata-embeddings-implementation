from tensorflow.keras import layers
import tensorflow as tf


class ReductionSumLayer(layers.Layer):
    def __init__(self):
        super(ReductionSumLayer, self).__init__()

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)


def set_input(model, config):
    model.add(
        layers.InputLayer(
            input_shape=config.get("input_shape", None),
            batch_size=config.get("batch_size", None),
            dtype=config.get("dtype", None),
            input_tensor=config.get("input_tensor", None),
            sparse=config.get("sparse", None),
            name=config.get("name", None),
            ragged=config.get("ragged", None)
        )
    )
    return model


def _add_activation(model, activation_name, activation_config=None):
    if activation_name == "leaky_relu":
        model.add(
            layers.LeakyReLU(alpha=activation_config.get("alpha", None))
        )
    else:
        model.add(
            layers.Activation(activation_name)
        )
    return model


def add_layer(model, config, prev_name):
    if (prev_name.startswith("cnn") | prev_name.startswith("maxpool")) and config['name'] == "dense":
        model.add(layers.Flatten())
    if config['name'] == "cnn2d":
        model.add(
            layers.Conv2D(
                filters=config.get("filters"),
                kernel_size=config.get("kernel_size"),
                strides=config.get("strides", (1, 1)),
                padding=config.get("padding", 'valid'),
                data_format=config.get("data_format", 'channels_last'),
                dilation_rate=config.get("dilation_rate", (1, 1)),
                groups=config.get("groups", 1),
                use_bias=config.get("use_bias", True),
                kernel_initializer=config.get("kernel_initializer", 'glorot_uniform'),
                bias_initializer=config.get("bias_initializer", 'zeros'),
                kernel_regularizer=config.get("kernel_regularizer", None),
                bias_regularizer=config.get("bias_regularizer", None),
                activity_regularizer=config.get("activity_regularizer", None),
                kernel_constraint=config.get("kernel_constraint", None),
                bias_constraint=config.get("bias_constraint", None)
            )
        )
    elif config['name'] == "cnn2dT":
        model.add(
            layers.Conv2DTranspose(
                filters=config.get("filters"),
                kernel_size=config.get("kernel_size"),
                strides=config.get("strides", (1, 1)),
                padding=config.get("padding", 'valid'),
                data_format=config.get("data_format", 'channels_last'),
                dilation_rate=config.get("dilation_rate", (1, 1)),
                use_bias=config.get("use_bias", True),
                kernel_initializer=config.get("kernel_initializer", 'glorot_uniform'),
                bias_initializer=config.get("bias_initializer", 'zeros'),
                kernel_regularizer=config.get("kernel_regularizer", None),
                bias_regularizer=config.get("bias_regularizer", None),
                activity_regularizer=config.get("activity_regularizer", None),
                kernel_constraint=config.get("kernel_constraint", None),
                bias_constraint=config.get("bias_constraint", None)
            )
        )
    elif config['name'] == "cnn1d":
        model.add(
            layers.Conv1D(
                filters=config.get("filters"),
                kernel_size=config.get("kernel_size"),
                strides=config.get("strides", 1),
                padding=config.get("padding", 'valid'),
                data_format=config.get("data_format", 'channels_last'),
                dilation_rate=config.get("dilation_rate", 1),
                groups=config.get("groups", 1),
                use_bias=config.get("use_bias", True),
                kernel_initializer=config.get("kernel_initializer", 'glorot_uniform'),
                bias_initializer=config.get("bias_initializer", 'zeros'),
                kernel_regularizer=config.get("kernel_regularizer", None),
                bias_regularizer=config.get("bias_regularizer", None),
                activity_regularizer=config.get("activity_regularizer", None),
                kernel_constraint=config.get("kernel_constraint", None),
                bias_constraint=config.get("bias_constraint", None)
            )
        )
    elif config['name'] == "cnn1dT":
        model.add(
            layers.Conv1DTranspose(
                filters=config.get("filters"),
                kernel_size=config.get("kernel_size"),
                strides=config.get("strides", 1),
                padding=config.get("padding", 'valid'),
                data_format=config.get("data_format", 'channels_last'),
                dilation_rate=config.get("dilation_rate", 1),
                groups=config.get("groups", 1),
                use_bias=config.get("use_bias", True),
                kernel_initializer=config.get("kernel_initializer", 'glorot_uniform'),
                bias_initializer=config.get("bias_initializer", 'zeros'),
                kernel_regularizer=config.get("kernel_regularizer", None),
                bias_regularizer=config.get("bias_regularizer", None),
                activity_regularizer=config.get("activity_regularizer", None),
                kernel_constraint=config.get("kernel_constraint", None),
                bias_constraint=config.get("bias_constraint", None)
            )
        )
    elif config['name'] == 'maxpool1D':
        model.add(
            layers.MaxPool1D(
                pool_size=config.get("pool_size"),
                strides=config.get("strides"),
                padding=config.get("padding", 'valid'),
            )
        )
    elif config['name'] == 'lstm_layers':
        model.add(
            layers.RNN(
                layers.LSTMCell(
                    units=config.get("units"),
                    recurrent_activation=config.get("recurrent_activation", 'sigmoid'),
                    use_bias=config.get("use_bias", True),
                    kernel_initializer=config.get("kernel_initializer", 'glorot_uniform'),
                    recurrent_initializer=config.get("recurrent_initializer", 'orthogonal'),
                    bias_initializer=config.get("bias_initializer", 'zeros'),
                    unit_forget_bias=config.get("unit_forget_bias", True),
                    kernel_regularizer=config.get("kernel_regularizer", None),
                    recurrent_regularizer=config.get("recurrent_regularizer", None),
                    bias_regularizer=config.get("bias_regularizer", None),
                    kernel_constraint=config.get("kernel_constraint", None),
                    recurrent_constraint=config.get("recurrent_constraint", None),
                    bias_constraint=config.get("bias_constraint", None),
                    dropout=config.get("dropout", 0.),
                    recurrent_dropout=config.get("recurrent_dropout", 0.),
                    implementation=config.get("implementation", 2)
                )
            )
        )
    elif config['name'] == 'lstm':
        model.add(
            layers.LSTMCell(
                units=config.get("units"),
                recurrent_activation=config.get("recurrent_activation", 'sigmoid'),
                use_bias=config.get("use_bias", True),
                kernel_initializer=config.get("kernel_initializer", 'glorot_uniform'),
                recurrent_initializer=config.get("recurrent_initializer", 'orthogonal'),
                bias_initializer=config.get("bias_initializer", 'zeros'),
                unit_forget_bias=config.get("unit_forget_bias", True),
                kernel_regularizer=config.get("kernel_regularizer", None),
                recurrent_regularizer=config.get("recurrent_regularizer", None),
                bias_regularizer=config.get("bias_regularizer", None),
                kernel_constraint=config.get("kernel_constraint", None),
                recurrent_constraint=config.get("recurrent_constraint", None),
                bias_constraint=config.get("bias_constraint", None),
                dropout=config.get("dropout", 0.),
                recurrent_dropout=config.get("recurrent_dropout", 0.),
                implementation=config.get("implementation", 2)
            )
        )
    elif config['name'] == "dense":
        model.add(
            layers.Dense(
                units=config.get("units"),
                use_bias=config.get("use_bias", True),
                kernel_initializer=config.get("kernel_initializer", 'glorot_uniform'),
                bias_initializer=config.get("bias_initializer", 'zeros'),
                kernel_regularizer=config.get("kernel_regularizer", None),
                bias_regularizer=config.get("bias_regularizer", None),
                activity_regularizer=config.get("activity_regularizer", None),
                kernel_constraint=config.get("kernel_constraint", None),
                bias_constraint=config.get("bias_constraint", None)
                # ,name=str(config.get("layer_name", None))
            )
        )
    elif config['name'] == "embedding":
        model.add(
            layers.Embedding(
                input_dim=config.get("input_dim"),
                output_dim=config.get("output_dim"),
                input_length=config.get("input_length", None),
                embeddings_initializer=config.get("embeddings_initializer", "uniform")
                # ,name=str(config.get("layer_name", None))
            )
        )
    if config.get('activation', False):
        model = _add_activation(
            model=model,
            activation_name=config['activation'],
            activation_config=config.get('activation_config', None)
        )
    if config.get('batch_normalization', False):
        model.add(layers.BatchNormalization())
    if config.get('dropout', False):
        model.add(layers.Dropout(config['dropout']))
    if config.get("reshape", None):
        model.add(layers.Reshape(config["reshape"]))
    if config.get("flatten", None):
        model.add(layers.Flatten())
    if config.get("reduction_sum", None):
        model.add(ReductionSumLayer())
    return model
