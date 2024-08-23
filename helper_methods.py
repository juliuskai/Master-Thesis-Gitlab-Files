


import keras
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import davies_bouldin_score, mean_absolute_error, mean_squared_error, silhouette_score
import tensorflow as tf

def save_hyperparameters_with_styled_headers(tuner, output_pdf, output_csv):
    """
    Extracts the hyperparameters and final validation loss from each trial in the Keras Tuner search and saves them as a CSV and PDF table.
    The validation loss values are rounded to four decimal places in the PDF, and the column headers are styled.
    
    Parameters:
    tuner (kerastuner.Tuner): The Keras Tuner object used for hyperparameter search.
    output_pdf (str): The path to save the output PDF file.
    output_csv (str): The path to save the output CSV file.
    highlight_row (str or None): The 'Trial ID' of the row to highlight. Default is None.
    highlight_color (str): The color to highlight the row. Default is 'lightgray'.
    """
    
    # fetch all the trials
    trials = tuner.oracle.trials
    hp_list = []

    for trial_id, trial in trials.items():
        trial_data = trial.hyperparameters.values.copy()
        trial_data['Trial ID'] = trial_id
        trial_data['Final Validation Loss'] = round(trial.score, 4)  # round final validation loss 
        
        # convert l1 and l2 to scientific notation with 4 digits after the decimal point
        if 'l1_reg' in trial_data:
            trial_data['l1_reg'] = f"{float(trial_data['l1_reg']):.4e}"
        if 'l2_reg' in trial_data:
            trial_data['l2_reg'] = f"{float(trial_data['l2_reg']):.4e}"

        hp_list.append(trial_data)

    hp_df = pd.DataFrame(hp_list)

    # round all other numeric values, except for l1_reg and l2_reg
    for col in hp_df.columns:
        if col not in ['l1_reg', 'l2_reg']:
            hp_df[col] = hp_df[col].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)

    hp_df.to_csv(output_csv, index=False)

    with PdfPages(output_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(len(hp_df.columns) * 1.5, len(hp_df) * 0.3))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=hp_df.values, colLabels=hp_df.columns, cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.2)

        for key, cell in table.get_celld().items():
            if key[0] == 0:  
                cell.set_fontsize(10)
                cell.set_text_props(weight='bold')  # make the header text bold

        pdf.savefig(fig)
        plt.close()

def calculate_shap_values(X_sample, feature_names, autoencoder, nsamples=1000, subset_size=100):
    """
    calculates SHAP values for a autoencoder using KernelExplainer.

    args:
    - X_sample (ndarray): Sample input data to calculate SHAP values for.
    - feature_names (list): List of feature names.
    - nsamples (int): Number of samples to use in KernelExplainer.
    - subset_size (int): Number of samples to use for SHAP value calculation to speed up the process.
    """

    background_sample = shap.sample(X_sample, 100)

    # Select a subset of X_sample for faster SHAP calculation
    X_sample_subset = X_sample[:subset_size]
    
    X_sample_flat = X_sample_subset.reshape((X_sample_subset.shape[0], -1))
    background_sample_flat = background_sample.reshape((background_sample.shape[0], -1))
    
    # prediction function for the autoencoder
    def predict(input_data):
        input_data_reshaped = input_data.reshape((-1, *X_sample.shape[1:]))
        reconstruction = autoencoder.predict(input_data_reshaped)
        return reconstruction.reshape((input_data.shape[0], -1))
    
    # kernelExplainer
    explainer = shap.KernelExplainer(predict, background_sample_flat)
    
    # calculate SHAP values
    shap_values = explainer.shap_values(X_sample_flat, nsamples=nsamples)

    # Summarize feature importance
    # If shap_values is a list of arrays (multi-output), average them
    if len(shap_values.shape) == 3:
        shap_values_mean = np.mean(shap_values, axis=2)
    else:
        shap_values_mean = shap_values        
    # Plot the SHAP values
    shap.summary_plot(shap_values_mean, feature_names=feature_names, plot_type="bar")
    
    return shap_values_mean

def evaluate_model(autoencoder, X_test_reshaped):
    """
    test the model on the test data.
    """
    loss = autoencoder.evaluate(X_test_reshaped, X_test_reshaped)
    print(f'Test Loss: {loss}')
    return loss


def calculate_mse(x_test, autoencoder):
    reconstruction = autoencoder.predict(x_test)
    mse = mean_squared_error(x_test.flatten(), reconstruction.flatten())
    print(f"MSE: {mse}")

    # Assuming x_test has shape (num_samples, height, width, channels)
    num_samples = x_test.shape[0]
    mse_per_sample = mse / num_samples
    print(f"MSE per sample: {mse_per_sample}")
    return mse


def evaluate_autoencoder(autoencoder, X_test, labels=None):
    """
    evaluates an autoencoder on test data.
    """
    reconstruction = autoencoder.predict(X_test)
    
    # Reconstruction error
    mse = mean_squared_error(X_test.flatten(), reconstruction.flatten())
    mae = mean_absolute_error(X_test.flatten(), reconstruction.flatten())
    rmse = np.sqrt(mse)
    
    # Latent space evaluation
    encoder = keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dense_1').output)
    latent_space_test = encoder.predict(X_test)
    
    if labels is not None:
        silhouette = silhouette_score(latent_space_test, labels)
        davies_bouldin = davies_bouldin_score(latent_space_test, labels)
    else:
        silhouette = None
        davies_bouldin = None
    
    results = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin
    }
    
    return results


def plot_loss(history):
    """
    Plot training and validation loss.
    
    Parameters:
        history (History): Training history
    """
    plt.plot(history['loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def plot_reconstruction_error(autoencoder, X_test_good_reshaped, X_test_bad_reshaped):
    """
    plots the reconstruction error for good and bad classes.
    """
    predGood = autoencoder.predict(X_test_good_reshaped)
    predBad = autoencoder.predict(X_test_bad_reshaped)

    rec1 = np.sum((predGood - X_test_good_reshaped) ** 2, axis=(1, 2, 3))
    rec2 = np.sum((predBad - X_test_bad_reshaped) ** 2, axis=(1, 2, 3))

    min_error = min(rec1.min(), rec2.min())
    max_error = max(rec1.max(), rec2.max())

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.hist(rec1, bins=30, range=(min_error, max_error), color='g', alpha=0.7)
    plt.title("Reconstruction Error Distribution for Good Class")
    plt.ylabel("Count of Good Samples")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.hist(rec2, bins=30, range=(min_error, max_error), color='r', alpha=0.7)
    plt.title("Reconstruction Error Distribution for Bad Class")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count of Bad Samples")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_vae(x_test, autoencoder):
    z_mean, z_log_var, z = autoencoder.encoder(x_test)
    reconstruction = autoencoder.decoder(z)
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(keras.losses.binary_crossentropy(x_test, reconstruction), axis=(1, 2))
    )
    kl_loss = tf.reduce_mean(
        -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
    )
    total_loss = reconstruction_loss + kl_loss

    print(f"Variational Autoencoder Total Loss: {total_loss.numpy()}")
    print(f"Variational Autoencoder Reconstruction Loss: {reconstruction_loss.numpy()}")
    print(f"Variational Autoencoder KL Divergence Loss: {kl_loss.numpy()}")