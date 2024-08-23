import pickle
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scipy.optimize import minimize
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../stats')))
import data_preprocessor

# Setting seeds for reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class InputOptimizer:
    def __init__(self, data_path, random_state=42):
        self.data_path = data_path
        self.data_processor = data_preprocessor.DataProcessor(data_path)
        self.scaler = StandardScaler()
        self.random_state = random_state    
    
    def load_and_prepare_data(self):
        """loads and prepares data for training and testing.
        """
        data = self.data_processor.prepare_data_for_project()

        data, target_values = self.data_processor.extract_columns(data, 'Setpoint')
        self.target_values = target_values.head(1).values.flatten()  

        X, y = self.data_processor.extract_columns(data, 'Actual')

        self.feature_names = X.columns.tolist() 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        # normalize data
        X_train_normalized = self.scaler.fit_transform(X_train.values)
        X_test_normalized = self.scaler.transform(X_test.values)
        
        return X_train, X_test, y_train, y_test, X_train_normalized, X_test_normalized
    

    def build_and_train_model(self, X_train, Y_train):
        """builds and trains a sequential model for a supervised learning approach for regression tasks
        """
        model = Sequential([
            Dense(128, activation='relu'),  
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(15, activation='linear')  
        ])

        model.compile(optimizer='adam', loss='mse')

        history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2)    

        return model
    
    def calculate_loss(self, model, x_test, y_test):
        """
        Calculates the mse loss for a neural network regressor.
        """
        y_pred = model.predict(x_test)
        mse_loss = mean_squared_error(y_test, y_pred)
        
        return mse_loss

    def objective_function(self, input_features, model, feature_indices):
        """Objective function for optimization. can also be used for reduced feature sets.

        Args:
            input_features (numpy array): Array of input features to optimize.
            model: The model used for predictions.
            feature_indices (list of int): Indices of the features being used.

        Returns:
            float: The sum of squared differences between target values and predicted values.
        """
        input_features = np.array(input_features).reshape(1, -1)
        
        #create a full-sized array filled with zeros if reduced feature set is used
        full_input_features = np.zeros((1, len(self.feature_names)))
        
        # place the input features into the correct positions (based on feature_indices)
        full_input_features[:, feature_indices] = input_features
        
        scaled_input_features = self.scaler.transform(full_input_features)
        
        # only select the relevant features after scaling
        scaled_input_features_reduced = scaled_input_features[:, feature_indices]
        
        predicted_values = model.predict(scaled_input_features_reduced)
        
        # calculate the sum of squared differences
        return np.sum((predicted_values.flatten() - self.target_values) ** 2)

    
    def plot_loss(self, base_filename='result-charts/optimization-regressor-loss.pdf'):
        plt.plot(self.history.history['loss'], label='train_loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(base_filename, format='pdf')
        plt.show()

    
    def plot_feature_table(self, feature_names, optimal_values):
        """plots a table of feature names and their corresponding optimal values.
        """
        df = pd.DataFrame({
            'Feature': feature_names,
            'Optimal Value': optimal_values.round(2)
        })

        fig, ax = plt.subplots(figsize=(10, len(feature_names) * 0.5)) 
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

        plt.title('Optimal Feature Values', y=0.8)
        plt.show()


    def plot_feature_table_all(self, feature_names, optimal_values_tuple, names, base_filename='result-charts/optimization-algorithms-comparison.pdf'):
        """Plots a table of feature names and their corresponding optimal values across multiple models.
        """
        df = pd.DataFrame({
            'Feature': feature_names,
            names[0]: optimal_values_tuple[0],
            names[1]: optimal_values_tuple[1],
            names[2]: optimal_values_tuple[2],
            names[3]: optimal_values_tuple[3]
        })

        df[names] = df[names].round(2)

        fig, ax = plt.subplots(figsize=(12, len(feature_names) * 0.5))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        table.auto_set_column_width(col=list(range(len(df.columns))))
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # Header row
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))

        plt.tight_layout()
        plt.savefig(base_filename, format='pdf')
        plt.show()
        plt.close(fig) 


    def find_best_model(self, X_train_normalized, X_test_normalized, y_train, y_test, top_30_features):
        mse_results = {}
        best_model = None
        best_mse = float('inf')

        # Iterate over the number of top features from numver of 22 to 28
        for n_features in range(22, 29):
            selected_features = top_30_features[:n_features]
            feature_indices = [self.feature_names.index(feature) for feature in selected_features]
            
            # reduce the feature set
            X_train_reduced = X_train_normalized[:, feature_indices]
            X_test_reduced = X_test_normalized[:, feature_indices]
            
            model = self.build_and_train_model(X_train_reduced, y_train.values)
            
            mse_loss = self.calculate_loss(model, X_test_reduced, y_test.values)
            
            mse_results[n_features] = mse_loss
            print(f"MSE with top {n_features} features: {mse_loss}")

            if mse_loss < best_mse:
                best_mse = mse_loss
                best_model = model
                best_n_features = n_features

        print(f"\nBest model found with MSE: {best_mse}")
        print(mse_results)
        
        return best_model, best_mse, best_n_features
    

    def optimize_for_perfect_targets(self, initial_input_values, model, feature_indices, file_name='optimization_results.pdf'):
        """Optimize input features to achieve perfect target values using different optimization methods.
        Returns:
            dict: Dictionary containing the best input values for each optimization method.
        """
        results = {}
        results_normalized = {}

        # List of optimization methods
        methods = ['BFGS', 'CG', 'Nelder-Mead', 'trust-constr']
        #methods = ['BFGS', 'CG', 'trust-constr']
        
        for method in methods:
            input_values = initial_input_values.copy()
            
            options = {}
            if method == 'Nelder-Mead':
                options = {'maxiter': 1000, 'disp': True, 'xatol': 1e-4, 'fatol': 1e-4}
            elif method == 'trust-constr':
                options = {'disp': True}

            # optimize
            result = minimize(self.objective_function, input_values, args=(model, feature_indices), method=method, options=options)
            
            # Get the best input features
            best_input_normalized = result.x
            
            # create a full-sized array filled with zeros as feature set might be reduced
            full_input_normalized = np.zeros(len(self.feature_names))
            
            # place the normalized reduced features into the correct positions (based on feature_indices) toahve th eoriginal feature set
            full_input_normalized[feature_indices] = best_input_normalized
            
            # Invert the normalization to get actual values
            full_input_original = self.scaler.inverse_transform(full_input_normalized.reshape(1, -1)).flatten()
            
            # extract the relevant original feature values of the reduced set
            best_input = full_input_original[feature_indices]

            results[method] = best_input
            results_normalized[f'{method}_normalized'] = best_input_normalized

            print(f"Optimization method: {method}")
            print(f"Best input (normalized): {best_input_normalized}")
            print(f"Best input (original): {best_input}\n")

        reduced_feature_names = [self.feature_names[i] for i in feature_indices]
        results_df = pd.DataFrame(results, index=reduced_feature_names).round(2)


        with PdfPages(file_name) as pdf:
            fig, ax = plt.subplots(figsize=(8, len(reduced_feature_names) * 0.3))  
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=results_df.values, colLabels=results_df.columns, rowLabels=results_df.index, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(0.8, 0.8)
            pdf.savefig(fig)
            plt.close()

        return results, results_normalized


    def save_predictions_to_pdf(self, predictions, file_name):
        """saves a pdf table with the prediction values for each algorithm.
        """
        df = pd.DataFrame(predictions)

        df = df.round(2)

        with PdfPages(file_name) as pdf:
            fig, ax = plt.subplots(figsize=(8, len(df.columns) * 0.5)) 
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.0, 1.0)  
            pdf.savefig(fig)
            plt.close()


    def save_to_pkl(self, file_name, **kwargs):
        with open(file_name, 'wb') as file:
            pickle.dump(kwargs, file)
    

def main():  
    data_path = '../data/original-data/continuous_factory_process.csv'
    pipeline = InputOptimizer(data_path)

    X_train, X_test, y_train, y_test, X_train_normalized, X_test_normalized = pipeline.load_and_prepare_data()

    all_features = pipeline.feature_names

    model_all_features = pipeline.build_and_train_model(X_train_normalized, y_train.values)
    #model_all_features.save('trained-models/all_feature_regressor.keras')

    top_30_features = [
        "Machine3RawMaterialProperty2",
        "Machine3RawMaterialProperty4",
        "Machine3RawMaterialProperty3",
        "Machine2RawMaterialProperty1",
        "Machine1RawMaterialProperty1",
        "Machine2RawMaterialProperty4",
        "FirstStageombinerOperationTemperature2",
        "Machine1RawMaterialProperty1",
        "Machine1ExitZoneTemperature",
        "Machine1MotorRPM",
        "Machine3RawMaterialProperty1",
        "Machine3MotorRPM",
        "Machine3MaterialPressure",
        "Machine1MaterialPressure",
        "Machine1RawMaterialProperty3",
        "Machine1MaterialPressure",
        "Machine1RawMaterialProperty2",
        "Machine1MaterialTemperature",
        "Machine3MaterialTemperature",
        "Machine2RawMaterialProperty3",
        "Machine1Zone2Temperature",
        "Machine1MotorAmperage",
        "Machine3MotorAmperage",
        "Machine2RawMaterialProperty2",
        "FirstStageombinerOperationTemperature1",
        "Machine1Zone1Temperature",
        "Machine3Zone2Temperature",
        "Machine3Zone1Temperature",
        "Machine2Zone1Temperature",
        "Machine2MotorRPM"
    ]

    reduced_model, optimal_mse, optimal_n_features = pipeline.find_best_model(X_train_normalized, X_test_normalized, y_train, y_test, top_30_features)

    #reduced_model.save('trained-models/reduced_feature_regressor.keras')

    top_features = top_30_features[:optimal_n_features]

    mse_loss_all_features = pipeline.calculate_loss(model_all_features, X_test_normalized, y_test.values)

    #print(mse_loss_all_features)
    #print(optimal_n_features)
    #print(optimal_mse)

    feature_indices_full = list(range(39))

    feature_indices_reduced = [all_features.index(feature) for feature in top_features]

    random_initial_input_values = np.random.rand(39)

    input_values = random_initial_input_values

    optimization_results_all_features_model, optimization_results_all_features_model_normalized = pipeline.optimize_for_perfect_targets(input_values, model_all_features, feature_indices_full, file_name='optimization-results/all_features_optimization.pdf')

    optimization_results_reduced_features_model, optimization_results_reduced_features_model_normalized = pipeline.optimize_for_perfect_targets(input_values[:optimal_n_features], reduced_model, feature_indices_reduced, file_name='optimization-results/reduced_features_optimization.pdf')

    #print(optimization_results_reduced_features_model)
    #print(optimization_results_reduced_features_model_normalized)

    predictions_reduced = {}
    predictions_all = {}

    for method, normalized_input in optimization_results_reduced_features_model_normalized.items():
        normalized_input = np.array(normalized_input).reshape(1, -1)
        prediction = reduced_model.predict(normalized_input)
        predictions_reduced[method] = prediction.flatten() 
        print(f"Prediction for {method}: {prediction.flatten()}")

    for method, normalized_input in optimization_results_all_features_model_normalized.items():
        normalized_input = np.array(normalized_input).reshape(1, -1)
        prediction = model_all_features.predict(normalized_input)
        predictions_all[method] = prediction.flatten() 
        print(f"Prediction for {method}: {prediction.flatten()}")


    pipeline.plot_feature_table_all(top_30_features[:26], (optimization_results_reduced_features_model['BFGS'], optimization_results_reduced_features_model['CG'], optimization_results_reduced_features_model['Nelder-Mead'], optimization_results_reduced_features_model['trust-constr']), ['BFGS', 'CG', 'NM', 'TR'], base_filename='result-charts-3stds/optimization-algorithms-comparison_reduced.pdf')

    pipeline.plot_feature_table_all(all_features, (optimization_results_all_features_model['BFGS'], optimization_results_all_features_model['CG'], optimization_results_all_features_model['NM'], optimization_results_all_features_model['trust-constr']), ['BFGS', 'CG', 'NM', 'TR'], base_filename='result-charts-3stds/optimization-algorithms-comparison_reduced.pdf')


    #combined_input_all_features = pipeline.scaler.transform(combined_input_all_features)
 
    #combined_input_all_features = pd.DataFrame(combined_input_all_features)

    #prediction_all_features = model_all_features.predict(combined_input_all_features)


    #print(prediction_all)  
    #print(prediction_reduced)  


    algorithms = ['BFGS', 'CG', 'Nelder-Mead', 'trust-constr']

    pipeline.save_predictions_to_pdf(predictions_all, file_name='optimization-results/all-algorithm-results.pdf')
    pipeline.save_predictions_to_pdf(predictions_reduced, file_name='optimization-results/reduced-algorithm-results.pdf')

    file_name = 'optimization-results/optimization_results_redcued_model.pkl'
    pipeline.save_to_pkl(
        file_name, 
        optimal_n_features=optimal_n_features, 
        mse_loss_reduced=optimal_mse, 
        optimization_results_reduced_features_model=optimization_results_reduced_features_model, 
        optimization_results_reduced_features_model_normalized=optimization_results_reduced_features_model_normalized, 
        prediction_reduced_features=prediction, 
        reduced_history=reduced_model.history
    )

if __name__ == "__main__":
    main()