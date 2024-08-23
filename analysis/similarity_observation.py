import os
import random
import sys
from matplotlib.font_manager import FontProperties
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from torch import layer_norm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../stats')))
import numpy as np
import data_preprocessor
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_classif, f_regression, VarianceThreshold



class SimilarityAnalysis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_processor = data_preprocessor.DataProcessor(data_path)
        self.scaler = StandardScaler()      


    def prepare_data(self, normalize=True):
        data = self.data_processor.prepare_data_for_project(add_target=True)  
        X = data.drop(columns=['Y_Target'])
        y = data['Y_Target']   

        self.feature_names = X.columns.tolist()   

        if normalize:
            X = preprocessing.normalize(X)

        X = pd.DataFrame(X)

        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        good_X = X[y == 1]
        bad_X = X[y == 0]

        good_y = y[y == 1]
        bad_y = y[y == 0]

        return (X, y, good_X, bad_X, good_y, bad_y, data)
    

    def plot_mean_differences(self, good_X, bad_X, output_file="result-charts-3stds/mean_differences.pdf"):
        mean_good_X = good_X.mean()
        mean_bad_X = bad_X.mean()
        mean_good_X = mean_good_X.round(2)
        mean_bad_X = mean_bad_X.round(2)

        good_and_bad_means = pd.concat([mean_good_X, mean_bad_X], axis=1)
        good_and_bad_means.columns = ['Good', 'Bad']
        
        good_and_bad_means['Absolute Difference'] = (good_and_bad_means['Good'] - good_and_bad_means['Bad']).abs()
        good_and_bad_means['Relative Difference'] = (good_and_bad_means['Absolute Difference'] / 
                                                     ((good_and_bad_means['Good'] + good_and_bad_means['Bad']) / 2))

        good_and_bad_means = good_and_bad_means.sort_values(by=['Relative Difference'], ascending=False)

        good_and_bad_means = good_and_bad_means.reset_index().rename(columns={'index': 'Feature Name'})

        good_and_bad_means = good_and_bad_means.applymap(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and x != 0 else "0.0" if x == 0 else x)

        #print(good_and_bad_means)

        self._plot_table(good_and_bad_means, 'Differences Of Means Between Classes', output_file)

    def plot_std_differences(self, good_X, bad_X, output_file="result-charts-3stds/std_differences.pdf"):
        std_good_X = good_X.std()
        std_bad_X = bad_X.std()
        std_good_X = std_good_X.round(2)
        std_bad_X = std_bad_X.round(2)

        good_and_bad_stds = pd.concat([std_good_X, std_bad_X], axis=1)
        good_and_bad_stds.columns = ['Good', 'Bad']
        
        good_and_bad_stds['Absolute Difference'] = (good_and_bad_stds['Good'] - good_and_bad_stds['Bad']).abs()
        good_and_bad_stds['Relative Difference'] = (good_and_bad_stds['Absolute Difference'] / 
                                                    ((good_and_bad_stds['Good'] + good_and_bad_stds['Bad']) / 2))

        good_and_bad_stds = good_and_bad_stds.sort_values(by=['Relative Difference'], ascending=False)

        good_and_bad_stds = good_and_bad_stds.reset_index().rename(columns={'index': 'Feature Name'})

        good_and_bad_stds = good_and_bad_stds.applymap(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and x != 0 else "0.0" if x == 0 else x)

        #print(good_and_bad_stds)

        self._plot_table(good_and_bad_stds, 'Differences Of Standard Deviations Between Classes', output_file)

    def _plot_table(self, df, title, output_file):
        fig, ax = plt.subplots(figsize=(10, len(df) * 0.3)) 
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        table.auto_set_column_width(col=list(range(len(df.columns))))
        
        for (row, col), cell in table.get_celld().items():
            if (row == 0) or (col == -1):
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))

        plt.title(title, y=1.0)
        plt.tight_layout()

        plt.savefig(output_file, format='pdf')
        plt.show()


    def plot_top_features_box_plots(self, data, class_column, top_n=10):
        mean_diff = data.groupby(class_column).mean().transpose()
        mean_diff['Relative Difference'] = (mean_diff.iloc[:, 0] - mean_diff.iloc[:, 1]).abs() / ((mean_diff.iloc[:, 0] + mean_diff.iloc[:, 1]) / 2)
        top_features = mean_diff['Relative Difference'].sort_values(ascending=False).head(top_n).index
        
        melted_data = data.melt(id_vars=class_column, var_name='Feature', value_name='Value')
        filtered_data = melted_data[melted_data['Feature'].isin(top_features)]
        
        plt.figure(figsize=(15, 10))
        sns.boxplot(x='Feature', y='Value', hue=class_column, data=filtered_data)
        plt.xticks(rotation=90)
        plt.title(f'Box Plots of Top {top_n} Features by Class')
        plt.show()


    def plot_correlation_heatmaps(self, data, class_column):
        class_labels = data[class_column].unique()
        
        for label in class_labels:
            subset = data[data[class_column] == label]
            correlation_matrix = subset.drop(columns=[class_column]).corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
            plt.title(f'Correlation Heatmap for Class {label}')
            plt.show()

    def plot_top_features_histograms(self, data, class_column, top_n=10):
        mean_diff = data.groupby(class_column).mean().transpose()
        mean_diff['Relative Difference'] = (mean_diff.iloc[:, 0] - mean_diff.iloc[:, 1]).abs() / ((mean_diff.iloc[:, 0] + mean_diff.iloc[:, 1]) / 2)
        top_features = mean_diff['Relative Difference'].sort_values(ascending=False).head(top_n).index
        
        num_rows = (top_n + 2) // 3 
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
        axes = axes.flatten()

        for i, feature in enumerate(top_features):
            sns.histplot(data=data, x=feature, hue=class_column, kde=True, ax=axes[i], palette='coolwarm', element='step')
            axes[i].set_title(f'Histogram of {feature}')


        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def calculate_pearson_correlation(self, df, input_features, output_features):
        correlation_matrix = pd.DataFrame(index=input_features, columns=output_features)
        
        for input_feature in input_features:
            for output_feature in output_features:
                correlation, _ = pearsonr(df[input_feature], df[output_feature])
                correlation_matrix.at[input_feature, output_feature] = correlation

        correlation_matrix = correlation_matrix.astype(float)
        
        return correlation_matrix
    

    def calculate_mutual_information(self, df, input_features, target_class, output_file="result-charts-3stds/mutual_information.pdf"):
        X = df[input_features]
        y = df[target_class]
        
        if len(df[target_class].unique()) > 2:
            mi_scores = mutual_info_regression(X, y)
        else:
            mi_scores = mutual_info_classif(X, y)
        
        mi_series = pd.Series(mi_scores, index=input_features)
        mi_series = mi_series.sort_values(ascending=False)
        
        # Round to two decimals and convert to string format
        mi_series = mi_series.apply(lambda x: f"{x:.2f}")
        
        self._save_as_pdf(mi_series, 'Mutual Information Scores', output_file)
        
        return mi_series

    def perform_anova_test(self, df, input_features, target_class, output_file="result-charts-3stds/anova_results.pdf"):
        X = df[input_features]
        y = df[target_class]
        
        if len(df[target_class].unique()) > 2:
            f_values, p_values = f_regression(X, y)
        else:
            f_values, p_values = f_classif(X, y)
        
        anova_results = pd.DataFrame({
            'F-value': f_values,
            'p-value': p_values
        }, index=input_features)
        
        anova_results = anova_results.sort_values(by='p-value')
        
        # Round to two decimals and convert to string format
        anova_results = anova_results.applymap(lambda x: f"{x:.2f}")
        
        self._save_as_pdf(anova_results, 'ANOVA Test Results', output_file)
        
        return anova_results

    def _save_as_pdf(self, data, title, output_file):
        fig, ax = plt.subplots(figsize=(10, len(data) * 0.3)) 
        ax.axis('tight')
        ax.axis('off')
        
        if isinstance(data, pd.Series):
            table_data = pd.DataFrame(data).reset_index()
            table_data.columns = ['Feature', 'Score']
        else:
            table_data = data.reset_index()
            table_data.columns = ['Feature'] + list(table_data.columns[1:])
        
        table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        table.auto_set_column_width(col=list(range(len(table_data.columns))))
        
        plt.title(title, y=1.0)
        plt.tight_layout()

        plt.savefig(output_file, format='pdf')
        plt.show()
    

    def apply_variance_threshold(self, df, threshold=0.0):
        selector = VarianceThreshold(threshold=threshold)

        reduced_df = selector.fit_transform(df)       

        kept_columns = df.columns[selector.get_support()]
        
        return pd.DataFrame(reduced_df, columns=kept_columns)
    
    
    def plot_correlation_heatmap(self, correlation_matrix, threshold=0.0, output_file="result-charts-3stds/correlation_heatmap.pdf"):
        mask = correlation_matrix.abs() < threshold

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, 
                    mask=mask, annot_kws={"size": 10}, linewidths=.5)

        plt.title('Pearson Correlation Heatmap', fontsize=16)
        plt.xlabel('Output Features', fontsize=12)
        plt.ylabel('Input Features', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_file, format='pdf')

        plt.show()


def main():
    obj = SimilarityAnalysis(
    '../data/original-data/continuous_factory_process.csv')

    (X, y, good_X, bad_X, good_y, bad_y, data) = obj.prepare_data(normalize=False)

    data_all = obj.data_processor.prepare_data_for_project()
    data_all, target_values = obj.data_processor.extract_columns(data_all, 'Setpoint')

    X_all, y_all = obj.data_processor.extract_columns(data_all, 'Actual')

    input_names_list = X_all.columns.tolist()
    output_names_list = y_all.columns.tolist()

    #obj.plot_mean_differences(good_X, bad_X)

    #obj.plot_std_differences(good_X, bad_X)

    #obj.plot_top_features_box_plots(data, 'Y_Target', top_n=10)

    #obj.plot_correlation_heatmaps(data, 'Y_Target')

    #obj.plot_top_features_histograms(data, 'Y_Target')

    #pearson_matrix = obj.calculate_pearson_correlation(data_all, input_names_list, output_names_list)

    #print(pearson_matrix)

    #print(data['Y_Target'].dtype)
    #data['Y_Target'] = data['Y_Target'].astype('category')
    #mutual_information = obj.calculate_mutual_information(data, input_names_list, 'Y_Target')

    #print(mutual_information)

    #anova = obj.perform_anova_test(data, input_names_list, 'Y_Target')

    #print(anova)

    #variance_threshold = obj.apply_variance_threshold(X_all, 0.9)

    #print(variance_threshold)

    #obj.plot_correlation_heatmap(pearson_matrix, threshold=0.0)












if __name__ == "__main__":
    main()








