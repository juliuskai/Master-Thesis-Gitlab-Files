import os
import sys
import random
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
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
        Get the hyperparameters for building and tuning the model during hp search
        """
        num_conv_layers = hp.Int('num_layers', min_value=2, max_value=4, step=1)
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
        activation = hp.Choice('activation', values=['relu', 'tanh', 'selu'])
        conv_units = hp.Int('conv_units', min_value=32, max_value=128, step=32)
        padding = hp.Choice('padding', values=['same'])
        pooling_type = hp.Choice('pooling_type', values=['max', 'average'])
        use_batch_norm = hp.Boolean('use_batch_norm')
        pooling_size = hp.Choice('pooling_size', values=['2x1', '2x2', '3x1', '3x3'])
        # convert to tuple, for instance 2x2 to (2,2)
        pooling_size = tuple(map(int, pooling_size.split('x')))

        # Calculate the  conv_units after the last encoded layer
        conv_units_in_last_layer = conv_units // (2 ** (num_conv_layers - 1))

        # dense_units should be smaller than the smallest conv_units
        dense_units = hp.Int('dense_units', min_value=2, max_value=conv_units_in_last_layer, step=2)

        if optimizer_choice == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

        regularization_type = hp.Choice('regularization_type', values=['l1', 'l2', 'elasticnet'])
        if regularization_type == 'l1':
            regularizer = regularizers.l1(hp.Float('l1_reg', min_value=1e-6, max_value=1e-2, sampling='LOG'))
        elif regularization_type == 'l2':
            regularizer = regularizers.l2(hp.Float('l2_reg', min_value=1e-6, max_value=1e-2, sampling='LOG'))
        elif regularization_type == 'elasticnet':
            l1_ratio = hp.Float('l1_ratio', min_value=0.0, max_value=1.0, step=0.1)
            regularizer = regularizers.L1L2(
                l1=hp.Float('l1_reg', min_value=1e-5, max_value=1e-2, sampling='LOG') * l1_ratio,
                l2=hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='LOG') * (1 - l1_ratio)
            )

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

            # save the best validation loss for this fold
            val_loss = np.min(history.history['val_loss'])  
            val_losses.append(val_loss)

        # calc average validation loss across all folds
        avg_val_loss = np.mean(val_losses)
        
        # Save the result to the tuner
        return {'val_loss': avg_val_loss}


def main():
    """
        Main function to run the autoencoder pipeline.
    """
    data_path = '../data/original-data/continuous_factory_process.csv'
    pipeline = AutoencoderPipeline(data_path)
    
    X_train_reshaped, X_test_reshaped, X_train_good_reshaped, X_train_bad_reshaped, X_test_good_reshaped, X_test_bad_reshaped, y_train, y_test = pipeline.load_and_prepare_data()
    
    hypermodel = CustomAutoencoderHyperModel(input_dim=X_train_reshaped.shape[1])

    # set up the tuner
    tuner = kt.RandomSearch(
        hypermodel,
        objective='val_loss',
        max_trials=50,  
        executions_per_trial=1,  # number of models to train for each combination of hps
        directory='hyperparameter-tuning-k-fold-3stds',
        project_name='convolutional-autoencoder-final'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',    
        min_delta=0.0003,       # minimum change to qualify asimprovement
        patience=5,           # number of epochs with no improvement after which training will be stopped
        verbose=1,             
        restore_best_weights=True  # Restore  weights from the best epoch
    )

    tuner.search(X_train_good_reshaped, X_train_good_reshaped, num_folds=5, callbacks=[early_stopping])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""      
    The optinmal number of layers is {best_hps.get('num_layers')}.
    The optimal number of units in the first convolutional layer is {best_hps.get('conv_units')}.
    The optimal number of units in the dense layer is {best_hps.get('dense_units')}.
    The optimal dropout rate is {best_hps.get('dropout_rate')}.
    The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
    The optimal number of epochs is {best_hps.get('epochs')}.
    The optimal batch size is {best_hps.get('batch_size')}.
    The optimal number of layers is {best_hps.get('num_layers')}.
    The optimal regularization type is {best_hps.get('regularization_type')}.
    The optimal L1 regularization is {best_hps.get('l1_reg') if best_hps.get('regularization_type') == 'l1' else 'N/A'}.
    The optimal L2 regularization is {best_hps.get('l2_reg') if best_hps.get('regularization_type') == 'l2' else 'N/A'}.
    The optimal L1 ratio is {best_hps.get('l1_ratio') if best_hps.get('regularization_type') == 'elasticnet' else 'N/A'}.
    The optimal optimizer is {best_hps.get('optimizer')}.
    The optimal activation function is {best_hps.get('activation')}.
    The optimal padding is {best_hps.get('padding')}.
    The optimal pooling type is  {best_hps.get('pooling_type')}.
    The optimal usage of batch normalization is {best_hps.get('use_batch_norm')}.
    The optimal pooling size is  {best_hps.get('pooling_size')}.
    """)

    tuner.reload()
    helper_methods.save_hyperparameters_with_styled_headers(tuner, output_pdf='convolutional-ae-hyperparameters_table-std3.pdf', output_csv='convolutional-ae-hyperparameters-std3.csv')


if __name__ == '__main__':
    main()
