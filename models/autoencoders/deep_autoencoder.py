import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, regularizers
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score, mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import random
import os
import sys 
import pandas as pd
import matplotlib.pyplot as plt
import shap
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


class DeepAutoencoderPipeline:
    def __init__(self, data_path, random_state=42):
        self.data_path = data_path
        self.data_processor = data_preprocessor.DataProcessor(data_path)
        self.scaler = MinMaxScaler()
        self.random_state = random_state

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

    def get_hyperparameters(self):
        """
        get the hps for building and tuning the model.
        """
        learning_rate = 0.001
        num_layers = 3
        dense_size1 = 24
        dropout_rate = 0.1
        optimizer_choice = 'adam'
        activation = 'selu'
        use_batch_norm = True
        regularization_type = 'l2'

        if optimizer_choice == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

        if regularization_type == 'l2':
            regularizer = regularizers.l2(2.012271984736256e-05)
 
        return (
            learning_rate, dropout_rate, num_layers, dense_size1,
            use_batch_norm, activation, optimizer, regularizer
        )

    def build(self, hp):
        """
        builds and compiles the autoencoder model.
        """
        input_layer = keras.Input(shape=(self.input_dim, 1, 1))

        (
            learning_rate, dropout_rate, num_layers, dense_size1,
            use_batch_norm, activation, optimizer, regularizer
        ) = self.get_hyperparameters(hp)

        x = layers.Flatten()(input_layer) 

        # first hidden layer 
        encoded = layers.Dense(dense_size1, kernel_regularizer=regularizer)(x)
        if use_batch_norm:
            encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Activation(activation)(encoded)

        if num_layers == 3:
            dense_size2 = dense_size1 // 2
            dense_size3 = dense_size2 // 2

            encoded = layers.Dense(dense_size2, kernel_regularizer=regularizer)(encoded)
            if use_batch_norm:
                encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Activation(activation)(encoded)
            encoded = layers.Dropout(dropout_rate)(encoded)

            encoded = layers.Dense(dense_size3, kernel_regularizer=regularizer)(encoded)
            if use_batch_norm:
                encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Activation(activation)(encoded)
            encoded = layers.Dropout(dropout_rate)(encoded)

            # bottleneck layer
            bn = layers.Dense(2, kernel_regularizer=regularizer)(encoded) 
            if use_batch_norm:
                bn = layers.BatchNormalization()(bn)
            bn = layers.Activation(activation)(bn)

            # Decoder layers
            decoded = layers.Dense(dense_size3, kernel_regularizer=regularizer)(bn)
            if use_batch_norm:
                decoded = layers.BatchNormalization()(decoded)
            decoded = layers.Activation(activation)(decoded)
            decoded = layers.Dropout(dropout_rate)(decoded)

            decoded = layers.Dense(dense_size2, kernel_regularizer=regularizer)(decoded)
            if use_batch_norm:
                decoded = layers.BatchNormalization()(decoded)
            decoded = layers.Activation(activation)(decoded)
            decoded = layers.Dropout(dropout_rate)(decoded)

        elif num_layers == 2:
            dense_size2 = dense_size1 // 2

            encoded = layers.Dense(dense_size2, kernel_regularizer=regularizer)(encoded)
            if use_batch_norm:
                encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Activation(activation)(encoded)
            encoded = layers.Dropout(dropout_rate)(encoded)

            # bottleneck layer
            bn = layers.Dense(2, kernel_regularizer=regularizer)(encoded) 
            if use_batch_norm:
                bn = layers.BatchNormalization()(bn)
            bn = layers.Activation(activation)(bn)

            # Decoder layers
            decoded = layers.Dense(dense_size2, kernel_regularizer=regularizer)(bn)
            if use_batch_norm:
                decoded = layers.BatchNormalization()(decoded)
            decoded = layers.Activation(activation)(decoded)
            decoded = layers.Dropout(dropout_rate)(decoded)


        # final decoder layer for reconstruction
        decoded = layers.Dense(self.input_dim, activation='sigmoid', kernel_regularizer=regularizer)(decoded)
        
        autoencoder = keras.Model(input_layer, decoded)
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

            # store the training and validation losses
            training_losses.append(history.history['loss'])
            validation_losses.append(history.history['val_loss'])
        
        # calculate the average training and validation loss across folds
        avg_training_loss = np.mean(training_losses, axis=0)
        avg_validation_loss = np.mean(validation_losses, axis=0)
        
        avg_history = {
            'loss': avg_training_loss,
            'val_loss': avg_validation_loss
        }
        
        return avg_history

def main():
    data_path = '../data/original-data/continuous_factory_process.csv'
    pipeline = DeepAutoencoderPipeline(data_path)

    X_train_reshaped, X_test_reshaped, X_train_good_reshaped, X_train_bad_reshaped, X_test_good_reshaped, X_test_bad_reshaped, y_train, y_test = pipeline.load_and_prepare_data()

    hypermodel = CustomAutoencoderHyperModel(input_dim=X_train_reshaped.shape[1])
    autoencoder = hypermodel.build()

    start_time = time.time()
    
    history = hypermodel.train_autoencoder(autoencoder, X_train_good_reshaped, 2, 16)

    end_time = time.time()

    # Calculate the training time
    training_time = end_time - start_time

    autoencoder.save('trained-models/3-stds-deep-autoencoder.keras')

    helper_methods.plot_loss(history)

    shap_values = helper_methods.calculate_shap_values(X_train_good_reshaped, pipeline.feature_names, autoencoder)

    mse = helper_methods.evaluate_model(X_test_reshaped, autoencoder)

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

    with open('results_3stds_deep_ae.pkl', 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
