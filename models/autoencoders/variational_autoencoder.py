import os
import time
import shap
import sys
import random
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
import keras
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../stats')))
import data_preprocessor
from keras import ops
import matplotlib.pyplot as plt
import pickle
import keras_tuner as kt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.')))
from stats import data_preprocessor
import helper_methods

# Setting seeds for reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)



class VariationalAutoencoderPipeline:
    def __init__(self, data_path):
        """
        Initialize the DataPreparation class with the path to the dataset.

        Parameters:
            data_path (str): Path to the dataset
        """
        self.data_processor = data_preprocessor.DataProcessor(data_path)
        self.scaler = MinMaxScaler()

    def load_and_prepare_data(self):
        """
        load and prepare data for training and testing
        """
        data = self.data_processor.prepare_data_for_project(add_target=True)

        X = data.drop(columns=['Y_Target'])
        y = data['Y_Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        self.feature_names = X.columns.tolist()

        X_train_normalized = self.scaler.fit_transform(X_train)
        X_test_normalized = self.scaler.transform(X_test)

        X_train_reshaped, X_test_reshaped, X_train_good_reshaped, X_train_bad_reshaped, X_test_good_reshaped, X_test_bad_reshaped = self.data_processor.reshape_and_categorize_data_for_autoencoder(X_train_normalized, X_test_normalized, y_train, y_test)

        return  X_train_reshaped, X_test_reshaped, X_train_good_reshaped, X_train_bad_reshaped, X_test_good_reshaped, X_test_bad_reshaped, y_train, y_test
    
 

@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    """z is sampled deüending on the mean, std (z_mean, z_log_var)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        """
        perform the reparameterization trick
        """
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class CustomVariationalAutoencoderHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def apply_pooling(self, x, pooling_type, padding, pooling_size):
        """
        returns the pooling function's output
        """
        if pooling_type == 'max':
            return layers.MaxPooling2D(pooling_size, padding=padding)(x)
        else:
            return layers.AveragePooling2D(pooling_size, padding=padding)(x)
        
    def get_hyperparameters(self):
        """
        gets the hyperparameters for building and tuning the model.
        """
        learning_rate = 0.0001
        dropout_rate = 0.0
        num_conv_layers = 4
        conv_units = 96
        padding = 'same'
        pooling_type = 'average'
        use_batch_norm = False
        pooling_size = (2,2)
        activation = 'relu'
        optimizer_choice = 'rmsprop'
        dense_units = 12
        latent_dim = 2
        regularization_type = 'l2'
     
        # Define optimizer based on hyperparameter choice
        if optimizer_choice == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

        if regularization_type == 'l2':
            regularizer = regularizers.l2(1.917073014369646e-06)

        return (
            learning_rate, dropout_rate, num_conv_layers, conv_units, padding, pooling_type,
            use_batch_norm, pooling_size, activation, optimizer, dense_units, latent_dim, regularizer
        )


    def build(self, hp):
        """
        builds the autpencoder model.
        """
        (
            learning_rate, dropout_rate, num_conv_layers, conv_units, padding, pooling_type,
            use_batch_norm, pooling_size, activation, optimizer, dense_units, latent_dim, regularizer
        ) = self.get_hyperparameters(hp)

        encoder_inputs = keras.Input(shape=self.input_shape)
        encoder = encoder_inputs

        for i in range(num_conv_layers):
            encoder = layers.Conv2D(conv_units, (3, 1), padding=padding, kernel_regularizer=regularizer)(encoder)
            if use_batch_norm:
                encoder = layers.BatchNormalization()(encoder)
            encoder = layers.Activation(activation)(encoder)
            encoder = self.apply_pooling(encoder, pooling_type, padding, pooling_size)
            conv_units //= 2  # Halve the number of units for each subsequent layer

        # define an additional fully connected layer
        bottleneck_layer = layers.Flatten()(encoder)
        bottleneck_layer = layers.Dense(dense_units, kernel_regularizer=regularizer)(bottleneck_layer)
        if use_batch_norm:
            bottleneck_layer = layers.BatchNormalization()(bottleneck_layer)
        bottleneck_layer = layers.Activation(activation)(bottleneck_layer) 
        bottleneck_layer = layers.Dropout(dropout_rate)(bottleneck_layer)
        z_mean = layers.Dense(latent_dim, name="z_mean", kernel_regularizer=regularizer)(bottleneck_layer)
        z_log_var = layers.Dense(latent_dim, name="z_log_var", kernel_regularizer=regularizer)(bottleneck_layer)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # Decoder
        latent_inputs = keras.Input(shape=(latent_dim,))
        # This size is chosen to match the number of units required for reshaping the output into a 3D tensor suitable for the next deconvolutional (transposed convolution) layers.
        # The multiplication factor 2 ensures that the units are sufficient for the required depth in the 3D reshaped tensor.
        # 10 is a chosen size for one of the dimensions in the reshaped tensor. This can be seen as the height of the tensor in a typical CNN.
        decoder = layers.Dense(dense_units * 2 * 10, kernel_regularizer=regularizer)(latent_inputs)     
        if use_batch_norm:
            decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation(activation)(decoder)   
        decoder = layers.Reshape((10, 1, dense_units * 2))(decoder)

        for i in range(num_conv_layers):
            # double the previously halfed unit number from last iteration of loop
            conv_units *= 2  # Double the number of units for each subsequent layer
            decoder = layers.Conv2DTranspose(conv_units, (3, 1), padding=padding, kernel_regularizer=regularizer)(decoder)
            if use_batch_norm:
                decoder = layers.BatchNormalization()(decoder)
            decoder = layers.Activation(activation)(decoder)
            if i < num_conv_layers - 1:  # Only upsample if not the last conv layer
                decoder = layers.UpSampling2D((2, 1))(decoder)

        # mqk3 sure the final output dimensions match the input dimensions
        output_shape = K.int_shape(decoder)
        if output_shape[1] != 39:
            padding_amount = 39 - output_shape[1]
            if padding_amount > 0:
                decoder = layers.ZeroPadding2D(((padding_amount, 0), (0, 0)))(decoder)
            elif padding_amount < 0:
                decoder = layers.Cropping2D(((-padding_amount, 0), (0, 0)))(decoder)

        decoder_outputs = layers.Conv2DTranspose(1, (3, 1), activation="sigmoid", padding=padding, kernel_regularizer=regularizer)(decoder)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        class VAE(keras.Model):
            def __init__(self, encoder, decoder, **kwargs):
                super().__init__(**kwargs)
                self.encoder = encoder
                self.decoder = decoder
                self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
                self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
                self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
                self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
                self.val_reconstruction_loss_tracker = keras.metrics.Mean(name="val_reconstruction_loss")
                self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")

            @property
            def metrics(self):
                return [
                    self.total_loss_tracker,
                    self.reconstruction_loss_tracker,
                    self.kl_loss_tracker,
                    self.val_total_loss_tracker,
                    self.val_reconstruction_loss_tracker,
                    self.val_kl_loss_tracker
                ]

            def call(self, inputs, training=False):
                """
                Forward pass for the VAE model
                """
                z_mean, z_log_var, z = self.encoder(inputs)
                reconstruction = self.decoder(z)
                
                # Ccalculate the reconstructions loss
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(keras.losses.mean_squared_error(inputs, reconstruction), axis=(1, 2))
                )
                # calculate kullback-leibler loss
                kl_loss = -0.5 * tf.reduce_mean(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
                )
                
                # add both custom losses
                self.add_loss(reconstruction_loss)
                self.add_loss(kl_loss)
                
                return reconstruction

            def train_step(self, data):
                if isinstance(data, tuple):
                    data = data[0]  
                with tf.GradientTape() as tape:
                    reconstruction = self(data, training=True)
                    total_loss = sum(self.losses)
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

                self.total_loss_tracker.update_state(total_loss)
                self.reconstruction_loss_tracker.update_state(self.losses[0])
                self.kl_loss_tracker.update_state(self.losses[1])

                return {
                    "loss": self.total_loss_tracker.result(),
                    "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                    "kl_loss": self.kl_loss_tracker.result(),
                }

            def test_step(self, data):
                if isinstance(data, tuple):
                    data = data[0]  
                reconstruction = self(data, training=False)
                total_loss = sum(self.losses)

                self.val_total_loss_tracker.update_state(total_loss)
                self.val_reconstruction_loss_tracker.update_state(self.losses[0])
                self.val_kl_loss_tracker.update_state(self.losses[1])

                return {
                    "loss": self.val_total_loss_tracker.result(),
                    "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
                    "kl_loss": self.val_kl_loss_tracker.result(),
                }

        vae = VAE(encoder, decoder)
        vae.compile(optimizer=optimizer)
        return vae
    

    def train_autoencoder(self, autoencoder, X_train_reshaped, epochs, batch_size, early_stopping=None, n_splits=5):
        """
        trains the autoencoder model using 5-fold cross-validation.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        training_losses = []
        validation_losses = []

        for train_index, val_index in kf.split(X_train_reshaped):
            X_train_fold, X_val_fold = X_train_reshaped[train_index], X_train_reshaped[val_index]

            history = autoencoder.fit(X_train_fold, X_train_fold, 
                                    epochs=epochs, batch_size=batch_size, 
                                    validation_data=(X_val_fold, X_val_fold), 
                                    callbacks=[early_stopping] if early_stopping else [])

            training_losses.append(history.history['loss'])
            validation_losses.append(history.history['val_loss'])
        
        avg_training_loss = np.mean(training_losses, axis=0)
        avg_validation_loss = np.mean(validation_losses, axis=0)
        
        avg_history = {
            'loss': avg_training_loss,
            'val_loss': avg_validation_loss
        }
        
        return avg_history
    

def main():

    data_path = '../data/original-data/continuous_factory_process.csv'
    pipeline = VariationalAutoencoderPipeline(data_path)

    X_train_reshaped, X_test_reshaped, X_train_good_reshaped, X_train_bad_reshaped, X_test_good_reshaped, X_test_bad_reshaped, y_train, y_test = pipeline.load_and_prepare_data()

    hypermodel = CustomVariationalAutoencoderHyperModel(input_dim=X_train_reshaped.shape[1])
    autoencoder = hypermodel.build()

    start_time = time.time()

    history = hypermodel.train_autoencoder(autoencoder, X_train_good_reshaped, 50, 32)

    end_time = time.time()

    # Calculate the training time
    training_time = end_time - start_time

    autoencoder.save('trained-models/std3-variational-autoencoder.keras')

    helper_methods.plot_loss(history)

    evaluation_metrics = helper_methods.evaluate_autoencoder(autoencoder, X_test_good_reshaped, hypermodel.encoder, labels=None)

    print(evaluation_metrics)

    #helper_methods.plot_reconstruction_error(X_test_good_reshaped, X_test_bad_reshaped, autoencoder)
    #mse = helper_methods.calculate_mse(X_test_reshaped, autoencoder)

    shap_values = helper_methods.calculate_shap_values(X_train_good_reshaped, pipeline.feature_names, autoencoder)

    #helper_methods.evaluate_vae(X_test_reshaped, autoencoder)

    data = {}
    data['metrics'] = evaluation_metrics 
    data['training_time'] = training_time
    data['val-loss'] = history['val_loss']
    data['loss'] = history['loss']
    data['shap'] = shap_values
    data['x_test'] = X_test_reshaped
    data['x_test_bad'] = X_test_bad_reshaped
    data['x_test_good'] = X_test_good_reshaped

    with open('results_std3_variational_ae.pkl', 'wb') as f:
        pickle.dump(data, f) 

if __name__ == '__main__':
    main()
