import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, regularizers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
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
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.')))
from stats import data_preprocessor
import helper_methods

# Setting seeds for reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


class SimpleAutoencoderPipeline:
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

    def get_hyperparameters(self, hp):
        """
        Get the hyperparameters for building and tuning the model.
        """
        dense_units = hp.Int('dense_units', min_value=8, max_value=32, step=8)
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        activation = hp.Choice('activation', values=['relu', 'tanh', 'selu'])
        use_batch_norm = hp.Boolean('use_batch_norm')

        optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])

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
                l1=hp.Float('l1_reg', min_value=1e-5, max_value=1e-4, sampling='LOG') * l1_ratio,
                l2=hp.Float('l2_reg', min_value=1e-5, max_value=1e-4, sampling='LOG') * (1 - l1_ratio)
            )

        return (
            learning_rate, dropout_rate, dense_units,
            use_batch_norm, activation, optimizer, regularizer
        )

    def build(self, hp):
        """
        build and compile the model
        """
        input_layer = keras.Input(shape=(self.input_dim, 1, 1))

        (
            learning_rate, dropout_rate, dense_units,
            use_batch_norm, activation, optimizer, regularizer
        ) = self.get_hyperparameters(hp)

        x = layers.Flatten()(input_layer)  

        # enocder
        encoded = layers.Dense(dense_units, kernel_regularizer=regularizer)(x)
        if use_batch_norm:
            encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Activation(activation)(encoded)
        encoded = layers.Dropout(dropout_rate)(encoded)

        # bottleneack layer added
        bn = layers.Dense(2, kernel_regularizer=regularizer)(encoded) 
        if use_batch_norm:
            bn = layers.BatchNormalization()(bn)
        bn = layers.Activation(activation)(bn)

        # decoder
        decoded = layers.Dense(self.input_dim, activation='sigmoid', kernel_regularizer=regularizer)(bn)
        
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.summary()
        autoencoder.compile(optimizer=optimizer, loss='mse')
        return autoencoder

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

        # cvalculate average validation loss across all folds
        avg_val_loss = np.mean(val_losses)

        return {'val_loss': avg_val_loss}
    

def main():
    data_path = '../data/original-data/continuous_factory_process.csv'
    pipeline = SimpleAutoencoderPipeline(data_path)

    X_train_reshaped, X_test_reshaped, X_train_good_reshaped, X_train_bad_reshaped, X_test_good_reshaped, X_test_bad_reshaped, y_train, y_test = pipeline.load_and_prepare_data()

    hypermodel = CustomAutoencoderHyperModel(input_dim=X_train_reshaped.shape[1])

    # Set up the tuner
    tuner = kt.RandomSearch(
        hypermodel,
        objective='val_loss',
        max_trials=50,  # Number of hyperparameter combinations to try
        executions_per_trial=1,  # Number of models to train for each combination of hyperparameters
        directory='hyperparameter-tuning-k-fold-3stds',
        project_name='simple-autoencoder'
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
    The optimal number of units in the dense layer is {best_hps.get('dense_units')}.
    The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
    The optimal number of epochs is {best_hps.get('epochs')}.
    The optimal batch size is {best_hps.get('batch_size')}.
    The optimal regularization type is {best_hps.get('regularization_type')}.
    The optimal L1 regularization is {best_hps.get('l1_reg') if best_hps.get('regularization_type') == 'l1' else 'N/A'}.
    The optimal L2 regularization is {best_hps.get('l2_reg') if best_hps.get('regularization_type') == 'l2' else 'N/A'}.
    The optimal L1 ratio is {best_hps.get('l1_ratio') if best_hps.get('regularization_type') == 'elasticnet' else 'N/A'}.
    The optimal optimizer is {best_hps.get('optimizer')}.
    The optimal dropout rate is {best_hps.get('dropout_rate')}.
    The optimal activation function is {best_hps.get('activation')}.
    The optimal usage of batch normalization is {best_hps.get('use_batch_norm')}.
    """)

    tuner.reload()
    helper_methods.save_hyperparameters_with_styled_headers(tuner, output_pdf='simple-ae-hyperparameters_table_std3.pdf', output_csv='simple-ae-hyperparameters_std3.csv')


if __name__ == "__main__":
    main()
