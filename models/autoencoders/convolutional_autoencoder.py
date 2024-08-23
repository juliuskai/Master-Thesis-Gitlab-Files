import os
import sys
import random
import time
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score, mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras.layers import Flatten, Dense, Reshape, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.python.keras import backend as K
import pickle
import shap
import keras_tuner as kt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.')))
from stats import data_preprocessor
import helper_methods

# Setting seeds for reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class AutoencoderPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
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
    

class CustomAutoencoderHyperModel(kt.HyperModel):
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def apply_pooling(self, x, pooling_type, padding, pooling_size):
        """
        returns the pooling function's output

        Args:
            x (): layer
            pooling_type (string): type of pooling
            padding (string): type of padding

        Returns:
            _type_: layer
        """
        if pooling_type == 'max':
            return layers.MaxPooling2D(pooling_size, padding=padding)(x)
        else:
            return layers.AveragePooling2D(pooling_size, padding=padding)(x)
        
    def get_hyperparameters(self):
        """
        define the hyperparameters for building and tuning the model.
        """
        num_conv_layers = 2
        learning_rate = 0.001
        dropout_rate = 0.0
        optimizer_choice = 'rmsprop'
        activation = 'tanh'
        conv_units = 32
        padding = 'same'
        pooling_type = 'max'
        use_batch_norm = False
        pooling_size = (2,2)
        dense_units = 2
        regularization_type = 'l2'

        if optimizer_choice == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

        if regularization_type == 'l2':
            regularizer = regularizers.l2(1.3768736261195804e-05)

        return (
            learning_rate, dropout_rate, num_conv_layers, conv_units, padding, pooling_type,
            use_batch_norm, pooling_size, activation, optimizer, dense_units, regularizer
        )


    def build(self, hp):
        """
        build and compile the autoencoder model.
        """
        input_layer = Input(shape=(self.input_dim, 1, 1))

        (
            learning_rate, dropout_rate, num_layers, conv_units, padding, pooling_type,
            use_batch_norm, pooling_size, activation, optimizer, dense_units, regularizer
        ) = self.get_hyperparameters(hp)


        # Encoder
        encoded = layers.Conv2D(conv_units, (3, 1), padding=padding, kernel_regularizer=regularizer)(input_layer)
        if use_batch_norm:
            encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Activation(activation)(encoded)
        encoded = self.apply_pooling(encoded, pooling_type, padding, pooling_size)

        for i in range(num_layers - 1):
            conv_units = conv_units // 2  # halve the number of units for each subsequent layer
            encoded = layers.Conv2D(conv_units, (3, 1), padding=padding, kernel_regularizer=regularizer)(encoded)
            if use_batch_norm:
                encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Activation(activation)(encoded)
            encoded = self.apply_pooling(encoded, pooling_type, padding, pooling_size)

        # add a fully connected bottleneck layer to furhter compress
        shape = K.int_shape(encoded)
        bottleneck_layer = Flatten()(encoded)

        bottleneck_layer = Dense(dense_units, kernel_regularizer=regularizer)(bottleneck_layer)
        if use_batch_norm:
            bottleneck_layer = layers.BatchNormalization()(bottleneck_layer) 
        bottleneck_layer = layers.Activation(activation)(bottleneck_layer)  
        bottleneck_layer = Dropout(dropout_rate)(bottleneck_layer)

        # add another fully connected bottleneck layer to decompresses a bit
        decompressed_bn = Dense(shape[1] * shape[2] * shape[3], kernel_regularizer=regularizer)(bottleneck_layer)
        if use_batch_norm:
            decompressed_bn = layers.BatchNormalization()(decompressed_bn)  
        decompressed_bn = layers.Activation(activation)(decompressed_bn) 

        encoded = Reshape((shape[1], shape[2], shape[3]))(decompressed_bn)

        # Decoder
        for i in range(num_layers - 1):
            conv_units = conv_units * 2  # double the number of units for each subsequent layer
            encoded = layers.Conv2DTranspose(conv_units, (3, 1), padding=padding, kernel_regularizer=regularizer)(encoded)
            if use_batch_norm:
                encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Activation(activation)(encoded)
            encoded = layers.UpSampling2D((2, 1))(encoded)

        encoded = layers.Conv2DTranspose(conv_units, (3, 1), padding=padding, kernel_regularizer=regularizer)(encoded)
        if use_batch_norm:
            encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Activation(activation)(encoded)
        encoded = layers.UpSampling2D((2, 1))(encoded)

        # final transpose layer to reconstruct
        encoded = layers.Conv2D(1, (3, 1), activation='sigmoid', padding=padding, kernel_regularizer=regularizer)(encoded)

        # in case the final output dimensions do not match the input dimensions
        output_shape = K.int_shape(encoded)
        if output_shape[1] != self.input_dim:
            padding_amount = self.input_dim - output_shape[1]
            if padding_amount > 0:
                encoded = layers.ZeroPadding2D(((padding_amount, 0), (0, 0)))(encoded)
            elif padding_amount < 0:
                encoded = layers.Cropping2D(((-padding_amount, 0), (0, 0)))(encoded)
        
        # compile the model
        autoencoder = Model(input_layer, encoded)
        autoencoder.summary()
        autoencoder.compile(optimizer=optimizer, loss='mse')

        return autoencoder
    
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
        
        # Calculate the average training and validation loss across folds
        avg_training_loss = np.mean(training_losses, axis=0)
        avg_validation_loss = np.mean(validation_losses, axis=0)
        
        avg_history = {
            'loss': avg_training_loss,
            'val_loss': avg_validation_loss
        }
        
        return avg_history


def main():
    data_path = '../data/original-data/continuous_factory_process.csv'
    pipeline = AutoencoderPipeline(data_path)
    
    X_train_reshaped, X_test_reshaped, X_train_good_reshaped, X_train_bad_reshaped, X_test_good_reshaped, X_test_bad_reshaped, y_train, y_test = pipeline.load_and_prepare_data()

    hypermodel = CustomAutoencoderHyperModel(input_dim=X_train_reshaped.shape[1])
    autoencoder = hypermodel.build()

    start_time = time.time()
    
    history = hypermodel.train_autoencoder(autoencoder, X_train_good_reshaped, 15, 64)

    end_time = time.time()

    # Calculate the training time
    training_time = end_time - start_time

    autoencoder.save('trained-models/3-stds-convolutional-autoencoder.keras')

    helper_methods.plot_loss(history)
  
    #helper_methods.plot_reconstruction_error(autoencoder, X_test_good_reshaped, X_test_bad_reshaped)

    shap_values = helper_methods.calculate_shap_values(X_train_good_reshaped, pipeline.feature_names, autoencoder)

    #helper_methods.evaluate_model(autoencoder, X_test_reshaped)            

    #mse = helper_methods.calculate_mse(X_test_reshaped, autoencoder)

    evaluation_metrics = helper_methods.evaluate_autoencoder(autoencoder, X_test_good_reshaped, labels=None)

    data = {}
    data['metrics'] = evaluation_metrics 
    data['training_time'] = training_time
    data['val-loss'] = history['val_loss']
    data['loss'] = history['loss']
    data['shap'] = shap_values
    data['x_test'] = X_test_reshaped
    data['x_test_bad'] = X_test_bad_reshaped
    data['x_test_good'] = X_test_good_reshaped

    with open('results_3stds_convolutional_ae.pkl', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    main()
