import pickle
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

def load_results(filename):
    # filename = 'results/' + filename
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results


def plot_mse_comparison(mse_values, labels, filename='result-charts-3stds/mse_comparison.pdf'):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, mse_values, color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f'])  
    plt.xlabel('Autoencoders')
    plt.ylabel('Mean Squared Error')
    #plt.title('Comparison of Mean Squared Error')
    plt.savefig(filename, format='pdf')  
    plt.show()


def plot_mae_comparison(mse_values, labels, filename='result-charts-3stds/mae_comparison.pdf'):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, mse_values, color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f'])  
    plt.xlabel('Autoencoders')
    plt.ylabel('Mean Absolute Error')
    #plt.title('Comparison of Mean Absolute Error')
    plt.savefig(filename, format='pdf')  
    plt.show()

def plot_shap_plots(shap_values_list, feature_names, labels):
    for i, shap_values in enumerate(shap_values_list):
        shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f'SHAP values for {labels[i]}')
        plt.show()

def plot_rmse_comparison(rmse_values, labels, filename='result-charts-3stds/rmse_comparison.pdf'):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, rmse_values, color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f'])  
    plt.xlabel('Autoencoders')
    plt.ylabel('Root Mean Squared Error')
    #plt.title('Comparison of Root Mean Squared Error')
    plt.savefig(filename, format='pdf')  
    plt.show()

def plot_training_time_comparison(training_time_values, labels, filename='result-charts-3stds/training-time.pdf'):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, training_time_values, color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f'])  
    plt.xlabel('Autoencoders')
    plt.ylabel('Training Time (seconds)')
    #plt.title('Comparison of Training Time')
    plt.savefig(filename, format='pdf')  
    plt.show()


def plot_losses(val_loss, loss, labels, filename='result-charts-3stds/training-losses.pdf'):
    num_autoencoders = len(val_loss)
    
    plt.figure(figsize=(15, 5)) 
    
    for i, history in enumerate(val_loss):
        plt.subplot(1, num_autoencoders, i + 1)
        plt.plot(loss[i], label='Train Loss')
        plt.plot(val_loss[i], label='Validation Loss')
        plt.title(f'Loss for {labels[i]}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, format='pdf')  
    plt.show()


def plot_loss(loss, val_loss, filename):
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format='pdf')  
    plt.show()


def plot_shap_comparison(shap_values_list, feature_names, labels, filename='result-charts-3stds/shap-comparison.pdf'):

    fig, axes = plt.subplots(nrows=1, ncols=len(shap_values_list), figsize=(20, 6))

    for i, shap_values in enumerate(shap_values_list):
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        sorted_indices = np.argsort(mean_shap_values)[::-1]
        sorted_shap_values = mean_shap_values[sorted_indices]
        sorted_feature_names = [feature_names[j] for j in sorted_indices]
        
        axes[i].barh(sorted_feature_names, sorted_shap_values)
        axes[i].set_title(f'SHAP values for {labels[i]}')
        # put the highest values on top
        axes[i].invert_yaxis()  

    plt.tight_layout()
    plt.savefig(filename, format='pdf')  
    plt.show()


def plot_shap_comparison_pairs(shap_values_list, feature_names, labels, base_filename='result-charts-3stds/shap-comparison.pdf'):

    num_charts = len(shap_values_list)
    num_rows = (num_charts + 1) // 2  

    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(10, 6*num_rows))

    for i, ax in enumerate(axes.flatten()):
        if i < num_charts:
            shap_values = shap_values_list[i]
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)
            sorted_indices = np.argsort(mean_shap_values)[::-1]
            sorted_shap_values = mean_shap_values[sorted_indices]
            sorted_feature_names = [feature_names[j] for j in sorted_indices]

            ax.barh(sorted_feature_names, sorted_shap_values)
            ax.set_title(f'SHAP values for {labels[i]}')
            ax.invert_yaxis()  # Put the highest values on top
        else:
            ax.axis('off')  
    plt.tight_layout()
    plt.savefig(base_filename, format='pdf')
    plt.close(fig)  


def plot_shap_comparison_seaborn_individual(shap_values_list, feature_names, labels, base_dir='result-charts-3stds/'):

    for i, shap_values in enumerate(shap_values_list):
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        sorted_indices = np.argsort(mean_shap_values)[::-1]
        sorted_shap_values = mean_shap_values[sorted_indices]
        sorted_feature_names = [feature_names[j] for j in sorted_indices]

        plt.figure(figsize=(7, 5.3))

        sns.barplot(x=sorted_shap_values, y=sorted_feature_names, palette="viridis")
        plt.xlabel('Mean |SHAP Value|')
        plt.ylabel('Feature')

        plt.gca().spines['left'].set_position(('outward', 1))  
        plt.gca().margins(x=0.01) 

        plt.tight_layout(rect=[0.02, 0, 1, 1])  #

        file_name = f"{base_dir}{labels[i]}-shap-values.pdf"
        plt.savefig(file_name, format='pdf')
        plt.close()





def find_common_features(shap_values_list, feature_names, labels, n, filename='result-charts-3stds/common-features.pdf'):
    """finds all features that are among the top 10 shap values for each autoencoder and lists them in a table ranked by number of occurrences across autoencoders
       and cumulative ascending position of ranks as a secondary ordering metric
    """
    feature_positions = defaultdict(list)

    # find the top n important features for each autoencoder
    for i, shap_values in enumerate(shap_values_list):
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        sorted_indices = np.argsort(mean_shap_values)[::-1]
        top_n_indices = sorted_indices[:n]
        top_n_features = [feature_names[j] for j in top_n_indices]
        
        for rank, feature in enumerate(top_n_features, 1):
            # append tuple in dictionary
            feature_positions[feature].append((labels[i], rank))

    # count the occurrences of each feature
    feature_counts = {feature: len(positions) for feature, positions in feature_positions.items()}

    # create a dataframe to show the common features and their positions
    feature_data = []
    for feature, positions in feature_positions.items():
        row = [feature, feature_counts[feature], sum(rank for _, rank in positions)]
        for label in labels:
            position = next((rank for autoencoder, rank in positions if autoencoder == label), '-')
            row.append(position)
        feature_data.append(row)

    columns = ['Feature', 'Count', 'Sum of Positions'] + labels
    feature_df = pd.DataFrame(feature_data, columns=columns)

    # sort the dataframe by descending count and sum of ascending positions and alphabetical order of names as third criterion
    feature_df.sort_values(by=['Count', 'Sum of Positions', 'Feature'], ascending=[False, True, True], inplace=True)

    plt.figure(figsize=(12, len(feature_df)*0.5))
    plt.axis('off')
    table = plt.table(cellText=feature_df.drop(columns=['Sum of Positions']).values, colLabels=feature_df.drop(columns=['Sum of Positions']).columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.auto_set_column_width(col=list(range(len(feature_df.columns))))
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    for (row, col), cell in table.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    plt.title(f'Top {n} Common Important Features Across Autoencoders')
    
    #plt.subplots_adjust(top=0.6)
    
    plt.savefig(filename, format='pdf')  
    plt.show()

    return feature_df


def find_common_feature_percentages(shap_values_list, feature_names, labels, n, filename='result-charts/common-features-percentage.pdf'):
    """Finds the top n features for each label and computes the percentage of these features
       that are found in the top 10 and top 15 features of other labels.

    """
    top_features_per_label = {}
    for i, shap_values in enumerate(shap_values_list):
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        sorted_indices = np.argsort(mean_shap_values)[::-1]
        
        top_n_indices = sorted_indices[:n]
        top_10_indices = sorted_indices[:10]
        top_15_indices = sorted_indices[:15]

        top_features_per_label[labels[i]] = {
            'top_n': [feature_names[j] for j in top_n_indices],
            'top_10': [feature_names[j] for j in top_10_indices],
            'top_15': [feature_names[j] for j in top_15_indices]
        }

    percentages = {label: {'10': 0, '15': 0} for label in labels}
    for label in labels:
        top_n_features = set(top_features_per_label[label]['top_n'])

        found_in_top_10 = set()
        found_in_top_15 = set()

        for other_label in labels:
            if label != other_label:
                top_10_features_other = set(top_features_per_label[other_label]['top_10'])
                top_15_features_other = set(top_features_per_label[other_label]['top_15'])

                found_in_top_10.update(top_n_features & top_10_features_other)
                found_in_top_15.update(top_n_features & top_15_features_other)

        percentages[label]['10'] = len(found_in_top_10) / len(top_n_features) * 100
        percentages[label]['15'] = len(found_in_top_15) / len(top_n_features) * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    index = np.arange(len(labels))
    bar_width = 0.35

    top_10_means = [percentages[label]['10'] for label in labels]
    top_15_means = [percentages[label]['15'] for label in labels]

    rects1 = ax.bar(index, top_10_means, bar_width, color='#87CEFA', label='Top 10')  
    rects2 = ax.bar(index + bar_width, top_15_means, bar_width, color='#FFA07A', label='Top 15')  

    ax.set_xlabel('Labels')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Percentage of Top {n} Features in Top 10 and Top 15 Features of Other Labels')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, format='pdf') 
    plt.show()


def find_common_feature_percentages_all(shap_values_list, feature_names, labels, n, filename='result-charts/common-features-percentage-all.pdf'):
    """Finds the top n features for each label and computes the percentage of these features
       that are found in the top 10 and top 15 features of ALL other labels.
    """
    top_features_per_label = {}
    for i, shap_values in enumerate(shap_values_list):
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        sorted_indices = np.argsort(mean_shap_values)[::-1]
        
        top_n_indices = sorted_indices[:n]
        top_10_indices = sorted_indices[:10]
        top_15_indices = sorted_indices[:15]

        top_features_per_label[labels[i]] = {
            'top_n': [feature_names[j] for j in top_n_indices],
            'top_10': [feature_names[j] for j in top_10_indices],
            'top_15': [feature_names[j] for j in top_15_indices]
        }

    percentages = {label: {'10': 0, '15': 0} for label in labels}
    for label in labels:
        top_n_features = set(top_features_per_label[label]['top_n'])

        found_in_all_top_10 = top_n_features.copy()
        found_in_all_top_15 = top_n_features.copy()

        for other_label in labels:
            if label != other_label:
                top_10_features_other = set(top_features_per_label[other_label]['top_10'])
                top_15_features_other = set(top_features_per_label[other_label]['top_15'])

                found_in_all_top_10 &= top_10_features_other
                found_in_all_top_15 &= top_15_features_other

        percentages[label]['10'] = len(found_in_all_top_10) / len(top_n_features) * 100
        percentages[label]['15'] = len(found_in_all_top_15) / len(top_n_features) * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    index = np.arange(len(labels))
    bar_width = 0.35

    top_10_means = [percentages[label]['10'] for label in labels]
    top_15_means = [percentages[label]['15'] for label in labels]

    rects1 = ax.bar(index, top_10_means, bar_width, color='#87CEFA', label='Top 10')  
    rects2 = ax.bar(index + bar_width, top_15_means, bar_width, color='#FFA07A', label='Top 15')  

    ax.set_xlabel('Labels')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Percentage of Top {n} Features in Top 10 and Top 15 Features of All Other Labels')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, format='pdf') 
    plt.show()



def main():
    # load results from each autoencoder
    results_convolutional_autoencoder = load_results('results_3stds_convolutional_ae.pkl')
    results_denoising_autpencoder = load_results('results_3stds_denoising_ae.pkl')
    results_variational_autoencoder = load_results('results_3stds_variational_ae.pkl')
    results_simple_autoencoder = load_results('results_3stds_simple_ae.pkl')
    results_deep_autpencoder = load_results('results_3stds_deep_ae.pkl')

    feature_names_res = load_results('results/results_convolutional_ae.pkl')

    labels = ['Simple', 'Deep', 'Denoising', 'Convolutional', 'Variational']

    mse_values = [
        results_simple_autoencoder['metrics']['mse'], 
        results_deep_autpencoder['metrics']['mse'], 
        results_denoising_autpencoder['metrics']['mse'], 
        results_convolutional_autoencoder['metrics']['mse'], 
        results_variational_autoencoder['metrics']['mse']
    ]

    print(mse_values)

    shap_values_list = [
        results_simple_autoencoder['shap'], 
        results_deep_autpencoder['shap'], 
        results_denoising_autpencoder['shap'], 
        results_convolutional_autoencoder['shap'], 
        results_variational_autoencoder['shap']
    ]

    mae_values = [
        results_simple_autoencoder['metrics']['mae'], 
        results_deep_autpencoder['metrics']['mae'], 
        results_denoising_autpencoder['metrics']['mae'], 
        results_convolutional_autoencoder['metrics']['mae'], 
        results_variational_autoencoder['metrics']['mae']
    ]

    print(mae_values)

    rmse_values = [
        results_simple_autoencoder['metrics']['rmse'], 
        results_deep_autpencoder['metrics']['rmse'], 
        results_denoising_autpencoder['metrics']['rmse'], 
        results_convolutional_autoencoder['metrics']['rmse'], 
        results_variational_autoencoder['metrics']['rmse']
    ]

    print(rmse_values)

    silhouette_values = [
        results_simple_autoencoder['metrics']['silhouette'], 
        results_deep_autpencoder['metrics']['silhouette'], 
        results_denoising_autpencoder['metrics']['silhouette'], 
        results_convolutional_autoencoder['metrics']['silhouette'], 
        results_variational_autoencoder['metrics']['silhouette']
    ]

    davies_bouldin_values = [
        results_simple_autoencoder['metrics']['davies_bouldin'], 
        results_deep_autpencoder['metrics']['davies_bouldin'], 
        results_denoising_autpencoder['metrics']['davies_bouldin'], 
        results_convolutional_autoencoder['metrics']['davies_bouldin'], 
        results_variational_autoencoder['metrics']['davies_bouldin']
    ]

    training_time_values = [
        results_simple_autoencoder['training_time'], 
        results_deep_autpencoder['training_time'], 
        results_denoising_autpencoder['training_time'], 
        results_convolutional_autoencoder['training_time'], 
        results_variational_autoencoder['training_time']
    ]

    val_losses = [
        results_simple_autoencoder['val-loss'], 
        results_deep_autpencoder['val-loss'], 
        results_denoising_autpencoder['val-loss'], 
        results_convolutional_autoencoder['val-loss'], 
        results_variational_autoencoder['val-loss']
    ]

    losses = [
        results_simple_autoencoder['loss'], 
        results_deep_autpencoder['loss'], 
        results_denoising_autpencoder['loss'], 
        results_convolutional_autoencoder['loss'], 
        results_variational_autoencoder['loss']
    ]

    # feature names are the same for all autoencoders obviously
    feature_names = feature_names_res['feature_names']

    #plot_mse_comparison(mse_values, labels)

    #plot_mae_comparison(mae_values, labels)

    #plot_rmse_comparison(rmse_values, labels)

    #plot_silhouette_comparison(silhouette_values, labels)

    #plot_davies_bouldin_comparison(davies_bouldin_values, labels)

    #plot_training_time_comparison(training_time_values, labels)    

    #plot_losses(val_losses, losses, labels)

    labels = ['SAE', 'DAE', 'DenAE', 'CAE', 'VAE']

    #plot_loss(losses[4], val_losses[4], 'result-charts-3stds/vae_losses.pdf')

    #plot_shap_comparison_seaborn_individual(shap_values_list, feature_names, labels)
        
    #plot_shap_comparison_pairs(shap_values_list, feature_names, labels)

    n = 10  # number of top features to consider
   #find_common_features(shap_values_list, feature_names, labels, n)

    #find_common_feature_percentages(shap_values_list, feature_names, labels, n)

    #find_common_feature_percentages_all(shap_values_list, feature_names, labels, n)
if __name__ == "__main__":
    main()
