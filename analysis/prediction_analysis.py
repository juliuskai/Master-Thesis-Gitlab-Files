import pickle
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import auc, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/autoencoders')))
import variational_autoencoder

def load_model_and_test_data(model_file, test_file, custom_objects=None):
    """Load a model and its corresponding test data."""
    model = load_model(model_file, custom_objects=custom_objects)
    with open(test_file, 'rb') as f:
        data = pickle.load(f)
    return model, data

def calculate_reconstruction_error(model, X_test):
    """Calculate reconstruction error for a test set using a trained model."""
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    reconstruction = model.predict(X_test)
    reconstruction_reshaped = reconstruction.reshape(reconstruction.shape[0], -1)  

    reconstruction_error = np.mean((X_test_reshaped - reconstruction_reshaped) ** 2, axis=1)
    
    return reconstruction_error


def set_threshold(reconstruction_error_good):
    """set a threshold based on the reconstruction error of the 'good' class."""
    threshold = reconstruction_error_good.mean() + 3 * reconstruction_error_good.std()
    return threshold


def compute_metrics(y_test, predicted_labels):
    """compute precision, recall, f1 score, and accuracy."""
    precision = precision_score(y_test, predicted_labels)
    recall = recall_score(y_test, predicted_labels)
    f1 = f1_score(y_test, predicted_labels)
    accuracy = accuracy_score(y_test, predicted_labels)
    return precision, recall, f1, accuracy

def plot_confusion_matrix_with_labels(cm, ax, title):
    group_labels = ["True Negative (TN)", "False Positive (FP)", "False Negative (FN)", "True Positive (TP)"]
    group_counts = [f"{value}" for value in cm.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm.flatten() / np.sum(cm)]
    labels = [f"{label}\n{count}\n{percentage}" for label, count, percentage in zip(group_labels, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax)
    ax.set_title(title)

def save_confusion_matrices_to_pdf(confusion_matrices, titles):
    pdf_file_paths = []
    num_matrices = len(confusion_matrices)
    num_pages = (num_matrices + 1) // 2  

    for i in range(num_pages):
        pdf_file_path = f"result-charts-3stds/confusion_matrices_{i+1}.pdf"
        pdf = PdfPages(pdf_file_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot first matrix on the left
        plot_confusion_matrix_with_labels(confusion_matrices[i * 2], axes[0], titles[i * 2])

        # Plot second matrix on the right, if available
        if i * 2 + 1 < num_matrices:
            plot_confusion_matrix_with_labels(confusion_matrices[i * 2 + 1], axes[1], titles[i * 2 + 1])
        else:
            # If only one matrix is left, remove the second axis
            fig.delaxes(axes[1])

        pdf.savefig(fig)
        plt.close(fig)
        pdf.close()
        
        pdf_file_paths.append(pdf_file_path)
    
    return pdf_file_paths


def plot_roc_points(models):
    """
    Plots single points on a ROC space for multiple models.
    """
    plt.figure(figsize=(8, 8))
    
    for model_name, (TP, FP, TN, FN) in models.items():
        TPR = TP / (TP + FN)  # True Positive Rate (Sensitivity, Recall)
        FPR = FP / (FP + TN)  # False Positive Rate (1 - Specificity)
        
        plt.scatter(FPR, TPR, label=model_name)
        plt.text(FPR + 0.01, TPR, model_name, fontsize=12)  # Add model name next to the point
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random classifier
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Space: Model Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_roc_curves(roc_data, custom_labels=None, filename="result-charts-3stds/roc_curves_3std.pdf"):
    plt.figure(figsize=(10, 8))
    
    for i, data in enumerate(roc_data):
        label = custom_labels[i] if custom_labels and i < len(custom_labels) else data['model']
        plt.plot(data['fpr'], data['tpr'], label=f"{label} (AUC = {data['roc_auc']:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    
    plt.tight_layout() 
    plt.savefig(filename, format='pdf')
    plt.show()




def main():
    model_files_3stds = ['./trained-models/3-stds-simple-autoencoder.keras', './trained-models/3-stds-deep-autoencoder.keras', './trained-models/3-stds-denoising-autoencoder.keras', './trained-models/3-stds-convolutional-autoencoder.keras', './trained-models/3-stds-variational-autoencoder.keras']
    test_files_3stds = ['./results_3stds_simple_ae.pkl', './results_3stds_deep_ae.pkl', './results_3stds_denoising_ae.pkl', './results_3stds_convolutional_ae.pkl', './results_3stds_variational_ae.pkl']
    
    model_files_1stds = ['./trained-models/simple-autoencoder.keras', './trained-models/deep-autoencoder.keras', './trained-models/denoising-autoencoder.keras', './trained-models/convolutional-autoencoder.keras', './trained-models/variational-autoencoder.keras']
    test_files_1stds = ['./results_simple_ae.pkl', './results_deep_ae.pkl', './results_denoising_ae.pkl', './results_convolutional_ae.pkl', './results_variational_ae.pkl']
    results = []
    confusion_matrices = []
    roc_data = []

    for model_file, test_file in zip(model_files_3stds, test_files_3stds):

        if 'variational' in model_file:
            hypermodel_instance = variational_autoencoder.CustomVariationalAutoencoderHyperModel((39, 1, 1))
            VAE = hypermodel_instance.build()
            custom_objects = {'VAE': VAE, 'Sampling' : variational_autoencoder.Sampling}  
        else:
             custom_objects = None

        model, data = load_model_and_test_data(model_file, test_file, custom_objects)
        
        X_test = data['x_test']
        X_test_good = data['x_test_good']
        X_test_bad = data['x_test_bad']

        X_test_good = X_test_good[:len(X_test_bad)]
        #X_test_bad = X_test_bad[:len(X_test_good)]

        X_combined_test = np.concatenate([X_test_good, X_test_bad], axis=0)
        
        # Create labels: 0 for good, 1 for bad
        y_combined_test = np.array([0] * len(X_test_good) + [1] * len(X_test_bad))
        
        # Calculate reconstruction error for the full test set and good class only
        reconstruction_error = calculate_reconstruction_error(model, X_combined_test)
        reconstruction_error_good = calculate_reconstruction_error(model, X_test_good)
        reconstruction_error_bad = calculate_reconstruction_error(model, X_test_bad)

        # Set threshold based on reconstruction error of the good class
        threshold = set_threshold(reconstruction_error_good)
        
        # Make predictions based on the threshold
        predicted_labels = (reconstruction_error > threshold).astype(int)
        
        precision, recall, f1, accuracy = compute_metrics(y_combined_test, predicted_labels)

        cm = confusion_matrix(y_combined_test, predicted_labels)
        confusion_matrices.append(cm)

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_combined_test, reconstruction_error)
        roc_auc = auc(fpr, tpr)

        # Store the ROC data for later plotting
        roc_data.append({
            'model': model_file,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        })
       
        # Store results
        results.append({
            'model': model_file,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy
        })
    
        # Print the model name and its confusion matrix
        print(f"Confusion Matrix for {model_file}:")
        print(cm)
        print()
    
    # Convert results to a DataFrame for easy viewing
    #results_df = pd.DataFrame(results)
    #print(results_df)

    #titles = ["SAE", "DAE & DenAE", "CAE", "VAE"]
    #confusion_matrices = confusion_matrices[:1] + confusion_matrices[2:]

    # Call the function to save them
    #save_confusion_matrices_to_pdf(confusion_matrices, titles)
        
    titles = ["SAE", "DAE", "DenAE", "CAE", "VAE"] 
    #plot_roc_curves(roc_data, custom_labels=titles)


if __name__ == "__main__":
    main()
