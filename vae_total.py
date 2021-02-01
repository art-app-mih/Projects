# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# fix random seed for reproducibility
np.random.seed(7)


dataframe_train = pd.read_csv('train.csv')
dataframe_test = pd.read_csv('test.csv')

def load_data(dataframe):
    # get wine Label
    df_base = dataframe.iloc[:, 1:]
    x = df_base.values.reshape(-1, df_base.shape[1]).astype('float32')
    # stadardize values   
    return x

train_data= load_data(dataframe_train)
test_data = load_data(dataframe_test)

train_data = train_data.reshape((len(train_data), np.prod(train_data.shape[1:])))
test_data = test_data.reshape((len(test_data), np.prod(test_data.shape[1:])))

"""#Variational Autoencoder"""

original_dim = len(train_data[0])
intermediate_dim = 1600
latent_dim = 50

inputs = tensorflow.keras.Input(shape=(original_dim,))
h = layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(h)
z_log_sigma = layers.Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_sigma])

# Create encoder
encoder = tensorflow.keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# Create decoder
latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = layers.Dense(original_dim, activation='sigmoid')(x)
decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = tf.keras.Model(inputs, outputs, name='vae_mlp')

reconstruction_loss = tensorflow.keras.losses.mse(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = reconstruction_loss + kl_loss
vae.add_loss(vae_loss)
vae.compile(optimizer='Adam')

vae.fit(train_data, train_data,
        epochs=25,
        batch_size=64,
        validation_data=(test_data, test_data))

x_train = np.array(encoder.predict(train_data, batch_size=64))
x_test = np.array(encoder.predict(test_data))
