import os
import numpy as np
import pandas as pd


class DataProcessor:
    def __init__(self, file_name):
        # Get the directory of the current script
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the absolute path to the input CSV file
        self.file_path = os.path.join(self.current_dir, file_name)
        self.df = pd.read_csv(self.file_path)
        self.df = self.remove_timestamp(self.df)


    def remove_timestamp(self, dataframe):
        """
        removes the timestamp from a dataframe
        """
        dataframe.drop(columns=['time_stamp'], inplace=True)
        return dataframe
    

    def remove_columns_by_keywords(self, df, keywords):
        """
        removes columns from a datafrmae whose names contain specified keywords specified as a list
        """
        columns_to_remove = [col for col in df.columns if any(keyword in col for keyword in keywords)]
        cleaned_df = df.drop(columns=columns_to_remove)
        return cleaned_df
        
    
    def rename_columns_by_keywords(self, df, rename_dict):
        """
        renames columns in a datafrmae based on the provided dictionary. each key is a keyword potentially found in column names,
        and the corresponding value is a list of words to be removed from those columns.
        """
        renamed_df = df.copy()
        
        for keyword, remove_words in rename_dict.items():
            matching_columns = [col for col in renamed_df.columns if keyword in col]
            
            for col in matching_columns:
                new_col = col
                for word in remove_words:
                    new_col = new_col.replace(word, "")
                renamed_df.rename(columns={col: new_col}, inplace=True)
        
        return renamed_df
    

    def drop_columns_by_keyword(self, df, keyword):
        """
        drops columns from a dataframe based on the presence of a keyword in the column name.
        """
        columns_to_drop = [col for col in df.columns if keyword in col]
        return df.drop(columns=columns_to_drop, inplace=False)

    
    def clean_data(self, dataframe, remove_zeros=False, col_suffix=None):
        """
        cleans the input DataFrame by removing rows containing NaN values,
        rows containing negative values in numeric columns, and optionally
        rows containing zeros in columns with a specified suffix.

        """
        cleaned_df = dataframe.dropna()

        # Coerce columns to numeric dtype (excluding datetime columns)
        numeric_columns = cleaned_df.select_dtypes(include=['number']).columns
        cleaned_df[numeric_columns] = cleaned_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # remove rows containing negative values 
        cleaned_df = cleaned_df[(cleaned_df[numeric_columns] >= 0).all(axis=1)]

        # remove entries containing zeros from columns with the specified suffix: Setpint as these sometimes contained falsely values as zeros
        if remove_zeros and col_suffix:
            for col in cleaned_df.columns:
                if col.endswith(col_suffix):
                    cleaned_df = cleaned_df[cleaned_df[col] != 0]

        cleaned_df = cleaned_df.reset_index(drop=True)

        return cleaned_df
       

    def add_target_and_cleanup(self, dataframe, stds):
        """
        adds a new column called Y_Target to the df. Rows where each 'Actual' value lies within one or three standard
        deviation of its corresponding 'Setpoint' value are labeled 'good' in Y_Target, otherwise labeled 'bad'.
        Finally, remove all 'Actual' and 'Setpoint' columns from the DataFrame.
        """
        dataframe['Y_Target'] = 'bad'
        means = dataframe.filter(like='Actual').mean()
        stds = dataframe.filter(like='Actual').std()
    
        for index, row in dataframe.iterrows():
            within_std = True
            
            # grab according ot index
            for col in means.index:
                actual_value = row[col]
                mean = means[col]
                std_dev = stds[col]
                
                # check for criterion; break out of loop if not satisfied for one column
                if not (actual_value >= mean - (stds * std_dev) and actual_value <= mean + (stds * std_dev)):
                    within_std = False
                    dataframe.at[index, 'Y_Target'] = 0
                    break
            
            # if all 'Actual' values meet the criteria, add the index of the row to the list
            if within_std:
                dataframe.at[index, 'Y_Target'] = 1
        
        # drop all 'Actual' and 'Setpoint' columns from the df
        filtered_df = dataframe.loc[:, ~dataframe.columns.str.contains('Actual|Setpoint')]
        
        return filtered_df
    

    def add_target_by_percentage_and_cleanup(self, dataframe, percentage):
        """
        adds a new column called Y_Target to the DataFrame. Rows where each 'Actual' value lies within a
        certain oercentage of its corresponding 'Setpoint' value are labeled 'good' in Y_Target, otherwise labeled 'bad'.
        Finally, remove all 'Actual' and 'Setpoint' columns from the DataFrame.
        """
        dataframe['Y_Target'] = 'bad'
        setpoints = dataframe.filter(like='Setpoint').mean()
        stds = dataframe.filter(like='Actual').std()
    
        for index, row in dataframe.iterrows():
            within_percentage = True
            
            for col in setpoints.index:
                setpoint_col = col.replace('Setpoint', 'Actual')
                actual_value = row[setpoint_col]
                setpoint = setpoints[col]
                # calculate the threshold and check
                threshold = setpoint * (percentage / 100)
                if abs(actual_value - setpoint) > threshold:
                        within_percentage = False
                        dataframe.at[index, 'Y_Target'] = 0
                        break

            # if all 'Actual' values meet the criteria, add the index of the row to the list
            if within_percentage:
                dataframe.at[index, 'Y_Target'] = 1
        
        # drop all 'Actual' and 'Setpoint' columns from the df
        filtered_df = dataframe.loc[:, ~dataframe.columns.str.contains('Actual|Setpoint')]
        
        return filtered_df
    

    def extract_columns(self, df, keyword):
        """
        extracts columns containing a keyword 
        """
        keyword_columns = [col for col in df.columns if keyword in col]

        keyword_df = df[keyword_columns]

        non_keyword_df = df.drop(columns=keyword_columns)

        return non_keyword_df, keyword_df
    

    def prepare_data_for_project(self, add_target=False):
        self.working_df = self.remove_columns_by_keywords(self.df, ['Stage2', 'Machine4', 'Machine5'])

        rename_dict = {'Machine1': ['Actual', 'U', 'C', '.'], 'Machine2': ['Actual', 'U', 'C', '.'],
                       'Machine3': ['Actual', 'U', 'C', '.'], 'FirstStage': ['Actual', 'U', '.', 'C'],
                       'Stage1': ['Output', 'Measurement', 'Stage1', 'U', '.']}
        self.working_df = self.rename_columns_by_keywords(self.working_df, rename_dict)
        self.working_df = self.drop_columns_by_keyword(self.working_df, 'AmbientCondition')
        self.working_df = self.clean_data(self.working_df, True, 'Setpoint')

        if add_target:
            self.working_df = self.add_target_and_cleanup(self.working_df, 3)

        return self.working_df       


    def reshape_and_categorize_data_for_autoencoder(self, X_train, X_test, y_train, y_test):
        """
        Reshape data for training the autoencoder.
        
        Parameters:
            X_train_normalized (array): Normalized training data
            X_test_normalized (array): Normalized test data
            y_train (array): Training labels
            y_test (array): Test labels
        
        Returns:
            tuple: Reshaped data for training and testing, validation
        """        
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        trainGoodX = X_train[y_train == 1]
        trainBadX = X_train[y_train == 0]
        testGoodX = X_test[y_test == 1]
        testBadX = X_test[y_test == 0]

        # Reshape for the autoencoder
        X_train_reshaped = np.expand_dims(np.expand_dims(X_train, axis=-1), axis=-1)
        X_test_reshaped = np.expand_dims(np.expand_dims(X_test, axis=-1), axis=-1)
        
        X_train_good_reshaped = np.expand_dims(np.expand_dims(trainGoodX, axis=-1), axis=-1)
        X_train_bad_reshaped = np.expand_dims(np.expand_dims(trainBadX, axis=-1), axis=-1)
        X_test_good_reshaped = np.expand_dims(np.expand_dims(testGoodX, axis=-1), axis=-1)
        X_test_bad_reshaped = np.expand_dims(np.expand_dims(testBadX, axis=-1), axis=-1)

        return X_train_reshaped, X_test_reshaped, X_train_good_reshaped, X_train_bad_reshaped, X_test_good_reshaped, X_test_bad_reshaped