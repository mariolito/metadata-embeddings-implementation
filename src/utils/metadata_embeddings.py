import tensorflow as tf
from tensorflow.keras import layers, Model
import math
import numpy as np
from tqdm import trange
import warnings
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s  - %(message)s', level=logging.INFO)
warnings.filterwarnings('ignore')


class ReductionSumLayer(layers.Layer):
    def __init__(self):
        super(ReductionSumLayer, self).__init__()

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)


class MetaDataEmbeddings(object):

    def __init__(self, config):
        self.num_epochs = config['num_epochs']
        self.verbose = config['verbose']
        self.learning_rate = config['learning_rate']
        self.beta1 = config['beta1']
        self.mini_batch_size = config['mini_batch_size']
        self.user_layers = config['user_layers']
        self.item_layers = config['item_layers']
        self.config = config
        self.activation = config['activation']

    def build_model(self):
        user_network_input = layers.Input(shape=(self.user_input_shape,), name="user_network")
        user_Embedding = layers.Embedding(
                input_dim=self.user_layers[0].get("input_dim"),
                output_dim=self.user_layers[0].get("output_dim"),
                input_length=self.user_layers[0].get("input_length"),
                embeddings_regularizer=tf.keras.regularizers.l2(self.item_layers[0].get("l2", 1e-6)),
                name='User_Embedding'
        )(user_network_input)
        user_Bias = layers.Embedding(
                input_dim=self.user_layers[0].get("input_dim"),
                output_dim=1,
                input_length=self.user_layers[0].get("input_length"),
                embeddings_initializer='zeros',
                name='user_Bias'
        )(user_network_input)
        user_factors = ReductionSumLayer()(user_Embedding)
        user_biases = ReductionSumLayer()(user_Bias)


        item_network_input = layers.Input(shape=(self.item_input_shape,), name="item_network")
        item_Embedding = layers.Embedding(
                input_dim=self.item_layers[0].get("input_dim"),
                output_dim=self.item_layers[0].get("output_dim"),
                input_length=self.item_layers[0].get("input_length", None),
                embeddings_regularizer=tf.keras.regularizers.l2(self.item_layers[0].get("l2", 1e-6)),
                name='Item_Embedding'
        )(item_network_input)
        item_Bias = layers.Embedding(
                input_dim=self.item_layers[0].get("input_dim"),
                output_dim=1,
                input_length=self.item_layers[0].get("input_length", None),
                embeddings_initializer='zeros',
                name='item_Bias'
        )(item_network_input)

        item_factors = ReductionSumLayer()(item_Embedding)
        item_biases = ReductionSumLayer()(item_Bias)

        dot_network = layers.Dot(axes=1, normalize=True)([user_factors, item_factors])
        dot_network = layers.Add()([dot_network, user_biases, item_biases])
        activation = layers.Activation(self.activation)(dot_network)

        model = Model(inputs=[user_network_input, item_network_input], outputs=activation, name='Model')

        user_factors_model = Model(inputs=[user_network_input], outputs=user_factors, name='user_factors')
        user_biases_model = Model(inputs=[user_network_input], outputs=user_biases, name='user_biases')

        item_factors_model = Model(inputs=[item_network_input], outputs=item_factors, name='item_factors')
        item_biases_model = Model(inputs=[item_network_input], outputs=item_biases, name='item_biases')
        return model, user_biases_model, user_factors_model, item_biases_model, item_factors_model

    def random_mini_batches(self, users, items, Y):
        mini_batches = []
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(self.m))
        shuffled_items = items[permutation, :]
        shuffled_users = users[permutation, :]
        shuffled_Y = Y[permutation, :]


        num_complete_minibatches = math.floor(self.m / self.mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_items = shuffled_items[
                           k * self.mini_batch_size: k * self.mini_batch_size +
                                                     self.mini_batch_size, :]
            mini_batch_users = shuffled_users[
                           k * self.mini_batch_size: k * self.mini_batch_size +
                                                     self.mini_batch_size, :]
            mini_batch_Y = shuffled_Y[
                           k * self.mini_batch_size: k * self.mini_batch_size +
                                                     self.mini_batch_size, :]
            mini_batch = (mini_batch_users, mini_batch_items, mini_batch_Y)
            mini_batches.append(mini_batch)
        # Handling the end case (last mini-batch < mini_batch_size)
        if self.m % self.mini_batch_size != 0:
            mini_batch_items = shuffled_items[
                           num_complete_minibatches * self.mini_batch_size:
                           self.m, :]
            mini_batch_users = shuffled_users[
                           num_complete_minibatches * self.mini_batch_size:
                           self.m, :]
            mini_batch_Y = shuffled_Y[
                           num_complete_minibatches * self.mini_batch_size:
                           self.m, :]
            mini_batch = (mini_batch_users, mini_batch_items, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches

    def get_item_representations(self):
        return self.item_biases_model(self.item_features).numpy(), self.item_factors_model(self.item_features).numpy()

    def get_user_representations(self):
        return self.user_biases_model(self.user_features).numpy(), self.user_factors_model(self.user_features).numpy()

    def _initiate(self, users, items, Y, user_features, item_features):
        self.users = users
        self.items = items
        (_, self.user_input_shape) = users.shape
        (_, self.item_input_shape) = items.shape
        (self.m, _) = Y.shape
        if len(item_features) > 0:
            self.item_features = item_features
        else:
            self.item_features = np.unique(items, axis=0)
        if len(user_features) > 0:
            self.user_features = user_features
        else:
            self.user_features = np.unique(users, axis=0)
        self.model, self.user_biases_model, self.user_factors_model, self.item_biases_model, self.item_factors_model = self.build_model()
        print(self.model.summary())

    def train(self, users, items, Y, user_features=[], item_features=[]):

        self._initiate(users, items, Y, user_features, item_features)

        loss_function = tf.keras.losses.BinaryCrossentropy()

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1)

        loss_l = []

        t = trange(self.num_epochs, desc="Epoch back-propagation", leave=True)
        for epoch in t:
            minibatches = self.random_mini_batches(users, items, Y)
            for minibatch in minibatches:
                (mini_batch_users, mini_batch_items, mini_batch_Y) = minibatch
                with tf.GradientTape() as tape:

                    activations = self.model([mini_batch_users, mini_batch_items])

                    loss = loss_function(mini_batch_Y, activations)

                grads = tape.gradient(loss, self.model.trainable_variables)

                optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables))

                msg = "Loss: {}".format(str(round(loss.numpy(), 4)))
                t.set_description(msg)
                t.refresh()
            if epoch % 10 == 0:
                loss_l.append(loss.numpy())
        all_activations = self.model([users, items])
        print(
            "Train is done. Train activations quantiles. 50%: {:.2f} 90%: {:.2f}".format(
            np.quantile(all_activations, 0.5), np.quantile(all_activations, 0.9))
        )
