import os
import shap
import sys
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
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
from matplotlib.backends.backend_pdf import PdfPages

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
        
    def get_hyperparameters(self, hp):
        """
        gets the hyperparameters for building and tuning the model.
        """
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        num_conv_layers = hp.Int('num_conv_layers', min_value=2, max_value=4, step=1)
        conv_units = hp.Int('conv_units', min_value=32, max_value=128, step=32)
        padding = hp.Choice('padding', values=['same'])
        pooling_type = hp.Choice('pooling_type', values=['max', 'average'])
        use_batch_norm = hp.Boolean('use_batch_norm')
        pooling_size = hp.Choice('pooling_size', values=['2x1', '2x2', '3x1', '3x3'])
        activation = hp.Choice('activation', values=['relu', 'tanh', 'selu'])
        optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])

        # convert to tuple, for instance 2x2 to (2,2)
        pooling_size = tuple(map(int, pooling_size.split('x')))

        # Calculate the  conv_units after the last encoded layer
        conv_units_in_last_layer = conv_units // (2 ** (num_conv_layers - 1))
        # dense unites shall be smaller than units form alst endocde layer
        dense_units = hp.Int('dense_units', min_value=2, max_value=conv_units_in_last_layer, step=2)

        # make latent dimension smaller than last dense_units to compress it more
        latent_dim = hp.Int('latent_dim', min_value=2, max_value=dense_units, step=2)
     
        if optimizer_choice == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

        regularization_type = hp.Choice('regularization_type', values=['l1', 'l2', 'elasticnet'])
        if regularization_type == 'l1':
            regularizer = regularizers.l1(hp.Float('l1_reg', min_value=1e-6, max_value=1e-4, sampling='LOG'))
        elif regularization_type == 'l2':
            regularizer = regularizers.l2(hp.Float('l2_reg', min_value=1e-6, max_value=1e-4, sampling='LOG'))
        elif regularization_type == 'elasticnet':
            l1_ratio = hp.Float('l1_ratio', min_value=0.0, max_value=1.0, step=0.1)
            regularizer = regularizers.L1L2(
                l1=hp.Float('l1_reg', min_value=1e-5, max_value=1e-2, sampling='LOG') * l1_ratio,
                l2=hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='LOG') * (1 - l1_ratio)
            )

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

    def fit(self, hp, model, X_train_reshaped, X_train_reshaped_original, num_folds=5, *args, **kwargs):
        """
        trains the model using K-fold cross-validation.
        """
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        val_losses = []

        for train_index, val_index in kfold.split(X_train_reshaped_original):
            X_train_fold, X_val_fold = X_train_reshaped[train_index], X_train_reshaped[val_index]

            history = model.fit(
                X_train_fold, X_train_fold,
                validation_data=(X_val_fold, X_val_fold),
                batch_size=hp.Choice('batch_size', values=[16, 32, 64, 128]),
                epochs=hp.Choice('epochs', values=[15, 30, 40, 50]),
                *args, **kwargs
            )

            val_loss = np.min(history.history['val_loss']) 
            val_losses.append(val_loss)

        # calculate average validation loss across all folds
        avg_val_loss = np.mean(val_losses)

        return {'val_loss': avg_val_loss}
    

def main():
    """
    Main function to run the data preparation and train the VAE.
    """
    data_path = '../data/original-data/continuous_factory_process.csv'
    pipeline = VariationalAutoencoderPipeline(data_path)

    X_train_reshaped, X_test_reshaped, X_train_good_reshaped, X_train_bad_reshaped, X_test_good_reshaped, X_test_bad_reshaped, y_train, y_test = pipeline.load_and_prepare_data()

    hypermodel = CustomVariationalAutoencoderHyperModel(input_dim=X_train_reshaped.shape[1])

    # Set up the tuner
    tuner = kt.RandomSearch(
        hypermodel,
        objective='val_loss',
        max_trials=50,  
        executions_per_trial=1, 
        directory='hyperparameter-tuning-k-fold-3stds',
        project_name='variational-autoencoder-last-layer-transpose2d'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',    
        min_delta=0.0003,      
        patience=5,          
        verbose=1,             
        restore_best_weights=True 
    )

    tuner.search(X_train_good_reshaped, X_train_good_reshaped, num_folds=5, callbacks=[early_stopping])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The optinmal number of convolutional units in the first layer is {best_hps.get('conv_units')}.      
    The optinmal number of layers is {best_hps.get('num_conv_layers')}.
    The optimal latent dimension is {best_hps.get('latent_dim')}.
    The optimal dropout rate is {best_hps.get('dropout_rate')}.
    The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
    The optimal number of epochs is {best_hps.get('epochs')}.
    The optimal batch size is {best_hps.get('batch_size')}.
    The optimal regularization type is {best_hps.get('regularization_type')}.
    The optimal L1 regularization is {best_hps.get('l1_reg') if best_hps.get('regularization_type') == 'l1' else 'N/A'}.
    The optimal L2 regularization is {best_hps.get('l2_reg') if best_hps.get('regularization_type') == 'l2' else 'N/A'}.
    The optimal L1 ratio is {best_hps.get('l1_ratio') if best_hps.get('regularization_type') == 'elasticnet' else 'N/A'}.
    The optimal optimizer is {best_hps.get('optimizer')}.
    The optimal activation function is {best_hps.get('activation')}.
    The optimal number of dense units is {best_hps.get('dense_units')}.
    The optimal padding is {best_hps.get('padding')}.
    The optimal pooling type is  {best_hps.get('pooling_type')}.
    The optimal usage of batch normalization is {best_hps.get('use_batch_norm')}.
    The optimal pooling size is {best_hps.get('pooling_size')}.
    """)


    tuner.reload()
    helper_methods.save_hyperparameters_with_styled_headers(tuner, output_pdf='variational-ae-hyperparameters_table-3stds.pdf', output_csv='variational-ae-hyperparameters-3stds.csv')


if __name__ == '__main__':
    main()
