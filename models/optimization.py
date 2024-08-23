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
        """constructor which inintialites a dataProcessor object

        Args:
            data_path (str): path of data
            random_state (int, optional): Defaults to 42.
        """
        self.data_path = data_path
        self.data_processor = data_preprocessor.DataProcessor(data_path)
        self.scaler = StandardScaler()
        self.random_state = random_state    
    
    def load_and_prepare_data(self):
        """loads and prepares data for training and testing.
        
        Returns:
            tuple: X_train, x_val, X_test, y_train, y_val, y_test, X_train_normalized, X_test_normalized, X_val_normalized
        """
        data = self.data_processor.prepare_data_for_project()

        data, setpoint = self.data_processor.extract_columns(data, 'Setpoint')
        setpoint = setpoint.head(1).values.flatten()  # Convert to numpy array and flatten

        self.target_values = data.filter(like='Actual').mean()
        print(self.target_values)
        print('aaaaaaaaaa')
        print(setpoint)

        # Split data
        X, y = self.data_processor.extract_columns(data, 'Actual')

        self.feature_names = X.columns.tolist()  # Store feature names

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Further split the training set into training and validation sets
        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) 

        # Normalize the data
        X_train_normalized = self.scaler.fit_transform(X_train.values)
        # X_val_normalized = self.scaler.transform(X_val)
        X_test_normalized = self.scaler.transform(X_test.values)
        
        return X_train, X_test, y_train, y_test, X_train_normalized, X_test_normalized
    

    def build_and_train_model(self, X_train, Y_train):
        """builds and trains a sequential model for a supervised learning approach

        Args:
            X_train (dataframe): training data features
            Y_train (dataframe): training data outcomes

        Returns:
            Trained model.
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
        Calculates the mean squared error loss for a neural network regressor.

        Args:
            model: Trained neural network model.
            x_test (numpy array or pandas DataFrame): Test data features.
            y_test (numpy array or pandas Series): True target values for the test data.

        Returns:
            float: The mean squared error loss.
        """
        # Predict the values using the model
        y_pred = model.predict(x_test)
        
        # Calculate the mean squared error
        mse_loss = mean_squared_error(y_test, y_pred)
        
        return mse_loss

    def objective_function(self, input_features, model, feature_indices):
        """Objective function for optimization using the given scaler, with support for reduced feature sets.

        Args:
            input_features (numpy array): Array of input features to optimize.
            model: The model used for predictions.
            feature_indices (list of int): Indices of the features being used.

        Returns:
            float: The sum of squared differences between target values and predicted values.
        """
        # Convert input_features to a 2D array
        input_features = np.array(input_features).reshape(1, -1)
        
        # Create a full-sized array filled with zeros
        full_input_features = np.zeros((1, len(self.feature_names)))
        
        # Place the input features into the correct positions (based on feature_indices)
        full_input_features[:, feature_indices] = input_features
        
        # Scale the full-sized array using the scaler fitted on the full dataset
        scaled_input_features = self.scaler.transform(full_input_features)
        
        # Now select only the relevant features after scaling
        scaled_input_features_reduced = scaled_input_features[:, feature_indices]
        
        # Predict using the model
        predicted_values = model.predict(scaled_input_features_reduced)
        
        # Calculate the sum of squared differences
        return np.sum((predicted_values.flatten() - self.target_values) ** 2)

    
    def plot_loss(self, base_filename='result-charts/optimization-regressor-loss.pdf'):
        """
        Plot training and validation loss.

        Parameters:
            history (History): Training history
        """
        plt.plot(self.history.history['loss'], label='train_loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(base_filename, format='pdf')
        plt.show()

    
    def plot_feature_table(self, feature_names, optimal_values):
        """plots a table of feature names and their corresponding optimal values.

        Parameters:
            feature_names (list of str): List of feature names.
            optimal_values (list of float): List of optimal values for each feature.
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

        Parameters:
            feature_names (list of str): List of feature names.
            optimal_values_tuple (tuple of lists of float): Tuple containing four lists of optimal values for each feature.
            names (list of str): List of names corresponding to each list in the tuple.
        """
        df = pd.DataFrame({
            'Feature': feature_names,
            names[0]: optimal_values_tuple[0],
            names[1]: optimal_values_tuple[1],
            names[2]: optimal_values_tuple[2],
            names[3]: optimal_values_tuple[3],
            names[4]: optimal_values_tuple[4]
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

        # Iterate over the number of top features from 22 to 28
        for n_features in range(22, 29):
            # Select the top n_features
            selected_features = top_30_features[:n_features]
            feature_indices = [self.feature_names.index(feature) for feature in selected_features]
            
            # Reduce the feature set
            X_train_reduced = X_train_normalized[:, feature_indices]
            X_test_reduced = X_test_normalized[:, feature_indices]
            
            # Train the model with the reduced feature set
            model = self.build_and_train_model(X_train_reduced, y_train.values)
            
            # Calculate the MSE
            mse_loss = self.calculate_loss(model, X_test_reduced, y_test.values)
            
            # Store the result
            mse_results[n_features] = mse_loss
            print(f"MSE with top {n_features} features: {mse_loss}")

            # Check if this model is the best so far
            if mse_loss < best_mse:
                best_mse = mse_loss
                best_model = model
                best_n_features = n_features

        print(f"\nBest model found with MSE: {best_mse}")
        print(mse_results)
        
        return best_model, best_mse, best_n_features
    

    def optimize_for_perfect_targets(self, initial_input_values, model, feature_indices, file_name='optimization_results.pdf'):
        """Optimize input features to achieve perfect target values using different optimization methods.

        Args:
            pipeline (InputOptimizer): Instance of the InputOptimizer class.
            initial_input_values (numpy array): Initial guess for the input features.

        Returns:
            dict: Dictionary containing the best input values for each optimization method.
        """
        results = {}
        results_normalized = {}

        # List of optimization methods
        methods = ['BFGS', 'CG', 'Nelder-Mead', 'trust-constr']
        #methods = ['BFGS', 'CG', 'trust-constr']
        
        for method in methods:
            # Reset input values for each optimization run
            input_values = initial_input_values.copy()
            
            # Select options based on the method
            options = {}
            if method == 'Nelder-Mead':
                options = {'maxiter': 1000, 'disp': True, 'xatol': 1e-4, 'fatol': 1e-4}
            elif method == 'trust-constr':
                options = {'disp': True}

            # Perform optimization
            result = minimize(self.objective_function, input_values, args=(model, feature_indices), method=method, options=options)
            
            # Get the normalized best input features
            best_input_normalized = result.x
            
            # Create a full-sized array filled with zeros
            full_input_normalized = np.zeros(len(self.feature_names))
            
            # Place the normalized reduced features into the correct positions (based on feature_indices)
            full_input_normalized[feature_indices] = best_input_normalized
            
            # Invert the normalization on the full-sized array using the scaler
            full_input_original = self.scaler.inverse_transform(full_input_normalized.reshape(1, -1)).flatten()
            
            # Extract the relevant original feature values
            best_input = full_input_original[feature_indices]

            # Store the result
            results[method] = best_input
            results_normalized[f'{method}_normalized'] = best_input_normalized

            print(f"Optimization method: {method}")
            print(f"Best input (normalized): {best_input_normalized}")
            print(f"Best input (original): {best_input}\n")

        # Create a DataFrame with the reduced feature names and round values to 2 decimal places
        reduced_feature_names = [self.feature_names[i] for i in feature_indices]
        results_df = pd.DataFrame(results, index=reduced_feature_names).round(2)

        # Save the DataFrame to a PDF file
        with PdfPages(file_name) as pdf:
            fig, ax = plt.subplots(figsize=(8, len(reduced_feature_names) * 0.3))  # Adjusted size for better column visibility
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=results_df.values, colLabels=results_df.columns, rowLabels=results_df.index, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(0.8, 0.8)  # Reduce the table size to make columns narrower
            pdf.savefig(fig)
            plt.close()

        return results, results_normalized


    def print_optimization_results(self, optimization_results):
        """Prints the best input features for each optimization method.

        Args:
            optimization_results (dict): Dictionary containing the best input values for each optimization method.
            pipeline (InputOptimizer): Instance of the InputOptimizer class containing the feature names.
        """
        for method, best_input in optimization_results.items():
            print(f"Best input features for perfect target values using {method}:")
            for name, value in zip(self.feature_names, best_input):
                print(f"{name}: {value:.4f}")
            print("\n")  # Add a newline for readability between methods


    def create_combined_dataframe(self, optimization_results, feature_indices=None):
        """Creates a DataFrame containing the recommended input features for each optimization method.

        Args:
            optimization_results (dict): Dictionary containing the best input values for each optimization method.
            feature_indices (list, optional): List of indices for the reduced set of features. 
                                            If None, assumes full feature set.

        Returns:
            pd.DataFrame: DataFrame with each row representing the best input values for a different optimization method.
        """
        # Prepare the data for the DataFrame
        data = []
        method_names = []

        for method, best_input in optimization_results.items():
            data.append(best_input)
            method_names.append(method)

        # Determine which feature names to use based on feature_indices
        if feature_indices is not None:
            selected_feature_names = [self.feature_names[i] for i in feature_indices]
        else:
            selected_feature_names = self.feature_names  # Use full feature set if no indices are provided

        # Create the DataFrame with the correct feature names
        combined_recommended_input_features_all_features = pd.DataFrame(data, columns=selected_feature_names, index=method_names)

        return combined_recommended_input_features_all_features
    

    def save_predictions_to_pdf(self, predictions, file_name):
        """Saves a PDF table with the prediction values for each algorithm.

        Args:
            predictions (dict): Dictionary containing prediction results for each algorithm.
            file_name (str): The name of the PDF file to save.
        """
        # Convert the dictionary into a DataFrame
        df = pd.DataFrame(predictions)

        # Round the values to two decimal places for readability
        df = df.round(2)

        # Save the DataFrame to a PDF file
        with PdfPages(file_name) as pdf:
            fig, ax = plt.subplots(figsize=(8, len(df.columns) * 0.5))  # Adjust the size as needed
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.0, 1.0)  # Adjust the scale to fit the table into the PDF page
            pdf.savefig(fig)
            plt.close()


    def save_to_pkl(self, file_name, **kwargs):
        """Saves multiple variables to a single pickle file.
        
        Args:
            file_name (str): The name of the pickle file to save.
            **kwargs: Named variables to save.
        """
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
    """test = [211.82675162, 205.11679483, 432.04233628,  12.85755822,
        12.29048741, 256.28261115,  86.01178237,  12.29048741,
        77.2251003 ,  11.52654865,   9.09786711,  13.68952778,
       251.84996994, 412.78172171, 974.51470229, 412.78172171,
       209.23886927,  81.96439341,  75.03276882, 571.7312197 ,
        72.26595251,  71.11619933, 347.75706827, 240.66564312,
       111.52706009,  72.06236334
       ]
    
    test2 = [191.62559332,  206.68943402,  476.59395878,   12.5452423 ,
         12.73169732,  256.72004436,   84.17434397,   12.73169732,
         78.53119051,    9.49940785,    9.13953357,   11.19349111,
        229.37500141,  448.14752719, 1653.93955834,  448.14752719,
        144.97626338,   80.60452397,   71.79392406,  557.21519192,
         66.40790009,   67.97842705,  335.02193096,  240.87383913,
        117.54230177,   73.57662664]
    
    test3 = [190.56736817,  207.09046449,  480.86311334,   12.55022379,   12.8483538,
            256.74028335,   84.60005638,   12.8483538,    78.00704708,    9.43263204,
            9.14521709,   11.55275842,  233.255311,    449.81821303, 1624.40206523,
            449.81821303,  142.62898312,   80.7767634,    71.24857486,  557.18994531,
            66.00161476,   67.7800658,   334.27307085,  240.69350509,  117.74184572,
            73.58217034

    ]"""
    cg_values = [
    12.04, 216.77, 1044.59, 250.86, 1257.66, 72.02, 72.04, 75.12, 11.46,
    423.54, 81.50, 77.99, 12.88, 240.35, 569.72, 256.29, 207.24, 69.03,
    69.15, 73.51, 13.91, 226.54, 77.06, 60.06, 9.27, 218.53, 426.95, 204.17,
    211.69, 78.01, 78.08, 346.65, 13.30, 252.56, 76.13, 65.06, 110.67, 86.75,
    80.08
    ]

    nelder_mead_values = [
        12.75, 226.44, 1411.36, 253.21, 1256.54, 72.03, 72.00, 117.89, 10.38,
        384.32, 81.55, 82.68, 13.34, 241.05, 582.77, 256.06, 192.23, 68.51,
        69.18, 71.56, 14.07, 226.59, 76.43, 59.96, 8.93, 61.46, 437.30, 224.50,
        275.57, 78.01, 78.84, 352.17, 13.37, 259.34, 77.07, 65.54, 114.46, 95.72,
        76.77
    ]

    nm_target = [
        11.05, 225.15, 1257.08, 252.12, 1235.34, 72.04, 72.02, 78.83, 6.19, 387.71,
        81.52, 90.40, 12.97, 240.12, 564.64, 256.49, 224.21, 68.70, 69.31, 72.73,
        14.02, 228.80, 76.66, 60.08, 9.25, 197.89, 435.10, 234.45, 226.59, 78.02,
        78.46, 337.39, 13.37, 230.22, 91.60, 65.61, 114.01, 91.66, 76.76
        ]

    combined_df = pd.DataFrame([nelder_mead_values])
    input_features = pipeline.scaler.transform(combined_df) 

 
    reshaped_df = pd.DataFrame(input_features)

    pred = model_all_features.predict(reshaped_df)
    print(pred)

    sys.exit()

    pipeline.plot_feature_table_all(pipeline.feature_names, (cg_values, cg_values, cg_values, nelder_mead_values, nm_target), ['BFGS', 'CG', 'TR', 'NM based on mean', 'NM based on target'], base_filename='result-charts-3stds/optimization-algorithms-comparison_all_both.pdf')

    reduced_model, optimal_mse, optimal_n_features = pipeline.find_best_model(X_train_normalized, X_test_normalized, y_train, y_test, top_30_features)

    reduced_model.save('trained-models/reduced_feature_regressor.keras')

    top_features = top_30_features[:optimal_n_features]

    mse_loss_all_features = pipeline.calculate_loss(model_all_features, X_test_normalized, y_test.values)


    #print(mse_loss_all_features)
    #sys.exit()
    print(optimal_n_features)
    print(optimal_mse)


    feature_indices_full = list(range(39))

    feature_indices_reduced = [all_features.index(feature) for feature in top_features]

    # Initial guess for the input features
    random_initial_input_values = np.random.rand(39)

    input_values = random_initial_input_values

    optimization_results_all_features_model, optimization_results_all_features_model_normalized = pipeline.optimize_for_perfect_targets(input_values, model_all_features, feature_indices_full, file_name='optimization-results/all_features_optimization_mean_values.pdf')

    optimization_results_reduced_features_model, optimization_results_reduced_features_model_normalized = pipeline.optimize_for_perfect_targets(input_values[:optimal_n_features], reduced_model, feature_indices_reduced, file_name='optimization-results/reduced_features_optimization_mean_values.pdf')

    #print(optimization_results_reduced_features_model)
    #print(optimization_results_reduced_features_model_normalized)

    #pipeline.plot_feature_table_all(top_30_features[:26], (test, test, test2, test), ['BFGS', 'CG', 'NM', 'TR'], base_filename='result-charts-3stds/optimization-algorithms-comparison_reduced.pdf')


    predictions_reduced = {}
    predictions_all = {}

    for method, normalized_input in optimization_results_reduced_features_model_normalized.items():
        # Convert the normalized input to a 2D array (needed for prediction)
        normalized_input = np.array(normalized_input).reshape(1, -1)

        # If necessary, inverse transform the normalized input to get the original scale
        # This step is optional depending on whether your model expects normalized or original scale inputs
        #original_input = pipeline.scaler.inverse_transform(normalized_input)

        # Make predictions using the reduced model
        prediction = reduced_model.predict(normalized_input)

        # Store the prediction
        predictions_reduced[method] = prediction.flatten()  # Flatten to get a 1D array of predictions

        print(f"Prediction for {method}: {prediction.flatten()}")


    print('predictions for all features model:\n')
    for method, normalized_input in optimization_results_all_features_model_normalized.items():
        # Convert the normalized input to a 2D array (needed for prediction)
        normalized_input = np.array(normalized_input).reshape(1, -1)

        # If necessary, inverse transform the normalized input to get the original scale
        # This step is optional depending on whether your model expects normalized or original scale inputs
        #original_input = pipeline.scaler.inverse_transform(normalized_input)

        # Make predictions using the reduced model
        prediction = reduced_model.predict(normalized_input)

        # Store the prediction
        predictions_all[method] = prediction.flatten()  # Flatten to get a 1D array of predictions

        print(f"Prediction for {method}: {prediction.flatten()}")









    #combined_input_all_features = pipeline.scaler.transform(combined_input_all_features)
 
    #combined_input_all_features = pd.DataFrame(combined_input_all_features)

    #prediction_all_features = model_all_features.predict(combined_input_all_features)


    #print(prediction_all_features)  
    print('aaaaa') 
    algorithms = ['BFGS', 'CG', 'Nelder-Mead', 'trust-constr']

    pipeline.save_predictions_to_pdf(predictions_reduced, file_name='optimization-results/reduced-algorithm-results_mean_values.pdf')
    pipeline.save_predictions_to_pdf(predictions_all, file_name='optimization-results/all-algorithm-results_mean_values.pdf')

    file_name_reduced = 'optimization-results/optimization_results_mean_values_reduced_model.pkl'
    pipeline.save_to_pkl(
        file_name_reduced, 
        optimal_n_features=25, 
        mse_loss_reduced=optimal_mse, 
        optimization_results_reduced_features_model=optimization_results_reduced_features_model, 
        optimization_results_reduced_features_model_normalized=optimization_results_reduced_features_model_normalized, 
        prediction_reduced_features=predictions_reduced, 
        reduced_history=reduced_model.history
    )

    file_name_all = 'optimization-results/optimization_results_mean_values_all_model.pkl'
    pipeline.save_to_pkl(
        file_name_all, 
        mse_loss_all_features=mse_loss_all_features, 
        optimization_results_all_features_model=optimization_results_all_features_model, 
        optimization_results_all_features_model_normalized=optimization_results_all_features_model_normalized, 
        prediction_all_features=predictions_all, 
        reduced_history=model_all_features.history
    )

    #print('all features:')
    #pipeline.print_optimization_results(optimization_results_all_features_model)

    #print('\n\nreduced features:')
    #pipeline.print_optimization_results(optimization_results_reduced_features_model)  

    sys.exit()

    """ feature_indices_15 = [all_features.index(feature) for feature in top_15_features]
    feature_indices_20 = [all_features.index(feature) for feature in top_20_features]
    feature_indices_25 = [all_features.index(feature) for feature in top_25_features]
    feature_indices_30 = [all_features.index(feature) for feature in top_30_features]

    X_train_reduced_15 = X_train_normalized[:, feature_indices_15]
    X_train_reduced_20 = X_train_normalized[:, feature_indices_20]
    X_train_reduced_25 = X_train_normalized[:, feature_indices_25]
    X_train_reduced_30 = X_train_normalized[:, feature_indices_30]
    
    x_test_reduced_15 = X_test_normalized[:, feature_indices_15]
    x_test_reduced_20 = X_test_normalized[:, feature_indices_20]
    x_test_reduced_25 = X_test_normalized[:, feature_indices_25]
    x_test_reduced_30 = X_test_normalized[:, feature_indices_30]

    model_cae_shap_features_15 = pipeline.build_and_train_model(X_train_reduced_15, y_train.values)
    model_cae_shap_features_20 = pipeline.build_and_train_model(X_train_reduced_20, y_train.values)
    model_cae_shap_features_25 = pipeline.build_and_train_model(X_train_reduced_25, y_train.values)
    model_cae_shap_features_30 = pipeline.build_and_train_model(X_train_reduced_30, y_train.values)

    mse_loss_all_features = pipeline.calculate_loss(model_all_features, X_test_normalized, y_test.values)

    mse_loss_reduced_feauters_15 = pipeline.calculate_loss(model_cae_shap_features_15, x_test_reduced_15, y_test.values)
    mse_loss_reduced_feauters_20 = pipeline.calculate_loss(model_cae_shap_features_20, x_test_reduced_20, y_test.values)
    mse_loss_reduced_feauters_25 = pipeline.calculate_loss(model_cae_shap_features_25, x_test_reduced_25, y_test.values)
    mse_loss_reduced_feauters_30 = pipeline.calculate_loss(model_cae_shap_features_30, x_test_reduced_30, y_test.values)

    print(f"MSE 20: {mse_loss_reduced_feauters_20}\nMSE 15: {mse_loss_reduced_feauters_15}\nMSE 25: {mse_loss_reduced_feauters_25}\nMSE 30: {mse_loss_reduced_feauters_30}\nMSE all: {mse_loss_all_features}") """



    sample1_values = [
        12.04, 216.77, 1044.59, 250.86, 1257.66, 72.02, 72.04, 75.12, 11.46, 423.54,
        81.5, 77.99, 12.88, 240.35, 569.72, 256.29, 207.24, 69.03, 69.15, 73.51,
        13.91, 226.54, 77.06, 60.06, 9.27, 218.53, 426.95, 204.17, 211.69, 78.01,
        78.08, 346.65, 13.3, 252.56, 76.13, 65.06, 110.67, 86.75, 80.08
    ]
    sample2_values = [
        11.05, 225.15, 1257.08, 252.12, 1235.34, 72.04, 72.02, 78.83, 6.19, 387.71,
        81.52, 90.4, 12.97, 240.12, 564.64, 256.49, 224.21, 68.7, 69.31, 72.73,
        14.02, 228.8, 76.66, 60.08, 9.25, 197.89, 435.1, 234.45, 226.59, 78.02,
        78.46, 337.39, 13.37, 230.22, 91.6, 65.61, 114.01, 91.66, 76.76
    ]



    combined_df = pd.DataFrame([sample1_values, sample2_values])

    input_features = pipeline.scaler.transform(combined_df) 

 
    reshaped_df = pd.DataFrame(input_features)

    pred = model_all_features.predict(reshaped_df)
    print(pred)




    print("Best input features for perfect target values:")
    #for name, value in zip(pipeline.feature_names, best_input_perfect_bfgs):
    #    print(f"{name}: {value}")

    #pipeline.plot_loss()
    #pipeline.plot_feature_table(pipeline.feature_names, best_input_perfect_tr)

    #pipeline.plot_feature_table_all(pipeline.feature_names, (best_input_perfect_bfgs, best_input_perfect_cg, best_input_perfect_nm, best_input_perfect_tr), ['BFGS', 'CG', 'NM', 'TR'])


if __name__ == "__main__":
    main()