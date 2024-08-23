import os
import pandas as pd


class StatsGenerator:
    def __init__(self, file_name):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(self.current_dir, file_name)
        self.df = pd.read_csv(self.file_path)
        self.df = self.remove_timestamp(self.df)

    
    def get_names_of_columns(self, df):
        return df.columns.tolist()
    
    def generate_basic_stats(self, dataframe):
        basic_stats = dataframe.describe()
        return basic_stats
    

    def find_outliers(self, dataframe):
        """
        Find outliers and count zeros per column in a DataFrame.

        Parameters:
            dataframe (DataFrame): The DataFrame for which outliers and zero counts are to be calculated.

        Returns:
            DataFrame: A DataFrame with counts of outliers, negative values, and zeros per column.
        """
        # Create a DataFrame to store outlier counts, negative value counts, and zero counts
        outlier_df = pd.DataFrame(index=['count', 'negative', 'zero'], columns=dataframe.columns)

        # Count outliers, negative values, and zeros for each column
        for col in dataframe.columns:
            if pd.api.types.is_numeric_dtype(dataframe[col]):
                outlier_df.loc['count', col] = dataframe[col].count()
                outlier_df.loc['negative', col] = (dataframe[col] < 0).sum()
                outlier_df.loc['zero', col] = (dataframe[col] == 0).sum()
            else:
                outlier_df.loc[:, col] = 0

        return outlier_df
    

    def calculate_within_std(self, dataframe, keywords):
        """
        Calculate the number of values within one, two, and three standard deviations
        for columns containing specified keywords.

        Parameters:
            dataframe (DataFrame): The input DataFrame.
            keywords (list): A list of keywords to search for in column names.

        Returns:
            DataFrame: A new DataFrame containing columns with specified keywords
                and rows indicating the counts of values within one, two, and three
                standard deviations.
        """
        result_data = {}
        
        for keyword in keywords:
            matching_cols = [col for col in dataframe.columns if keyword in col]
            
            for col in matching_cols:
                values = dataframe[col]
                mean = values.mean()
                std_dev = values.std()
                
                within_one_std = ((values >= mean - std_dev) & (values <= mean + std_dev)).sum()
                within_two_std = ((values >= mean - 2 * std_dev) & (values <= mean + 2 * std_dev)).sum()
                within_three_std = ((values >= mean - 3 * std_dev) & (values <= mean + 3 * std_dev)).sum()
                
                result_data[col] = [within_one_std, within_two_std, within_three_std]
        
        result_df = pd.DataFrame.from_dict(result_data, orient='index', columns=['Within 1 SD', 'Within 2 SD', 'Within 3 SD'])
        return result_df
    

    def count_matching_values(self, dataframe, num_pairs=15):
        """
        Count the number of matching values between setpoint and actual for each pair.

        Parameters:
            dataframe (DataFrame): The input DataFrame containing setpoint and actual columns.
            num_pairs (int, optional): The number of value pairs to consider. Default is 15.

        Returns:
            DataFrame: A new DataFrame containing the counts of matching values for each pair.
        """
        matching_counts = {}
        
        for i in range(num_pairs):
            setpoint_col = f'{i}Setpoint'
            actual_col = f'{i}Actual'
            
            matching_count = (dataframe[setpoint_col] == dataframe[actual_col]).sum()
            matching_counts[f'Pair {i}'] = matching_count
        
        result_df = pd.DataFrame.from_dict(matching_counts, orient='index', columns=['Matching Count'])
        return result_df
    

    def count_within_std_for_all_pairs(self, dataframe):
        """
        Count the number of rows where each 'Actual' value lies within one standard deviation
        of its column's mean.

        Parameters:
            dataframe (DataFrame): The input DataFrame containing 'Actual' columns.

        Returns:
            int: The count of rows where each 'Actual' value lies within one standard deviation
            of its column's mean.
        """
        num_rows = len(dataframe)
        
        # Calculate the mean and standard deviation for each 'Actual' column
        means = dataframe.filter(like='Actual').mean()
        stds = dataframe.filter(like='Actual').std()
        
        # Initialize a counter for the number of rows meeting the criteria
        count_within_std = 0
        
        # Iterate over each row in the DataFrame
        for index, row in dataframe.iterrows():
            # Initialize a flag to indicate whether all 'Actual' values meet the criteria
            within_std = True
            
            # Check if each 'Actual' value lies within one standard deviation of its column's mean
            for col in means.index:
                actual_value = row[col]
                mean = means[col]
                std_dev = stds[col]
                
                # Check if the actual value lies within one standard deviation of the mean
                if not (actual_value >= mean - std_dev and actual_value <= mean + std_dev):
                    within_std = False
                    break
            
            # If all 'Actual' values meet the criteria, increment the counter
            if within_std:
                count_within_std += 1
        
        return count_within_std
    

    def filter_within_std_for_all_columns(self, dataframe):
        """
        Filter the DataFrame to include only the rows where each 'Actual' value
        lies within one standard deviation of its column's mean.

        Parameters:
            dataframe (DataFrame): The input DataFrame containing 'Actual' columns.

        Returns:
            DataFrame: A new DataFrame consisting only of the rows where each 'Actual'
            value lies within one standard deviation of its column's mean.
        """
        # Calculate the mean and standard deviation for each 'Actual' column
        means = dataframe.filter(like='Actual').mean()
        stds = dataframe.filter(like='Actual').std()
        
        # Initialize an empty list to store the indices of matching rows
        matching_indices = []
        
        # Iterate over each row in the DataFrame
        for index, row in dataframe.iterrows():
            # Initialize a flag to indicate whether all 'Actual' values meet the criteria
            within_std = True
            
            # Check if each 'Actual' value lies within one standard deviation of its column's mean
            for col in means.index:
                actual_value = row[col]
                mean = means[col]
                std_dev = stds[col]
                
                # Check if the actual value lies within one standard deviation of the mean
                if not (actual_value >= mean - std_dev and actual_value <= mean + std_dev):
                    within_std = False
                    break
            
            # If all 'Actual' values meet the criteria, add the index of the row to the list
            if within_std:
                matching_indices.append(index)
        
        # Create a new DataFrame consisting only of the rows that matched the criteria
        filtered_df = dataframe.loc[matching_indices]
        
        return filtered_df


    def filter_within_percentage_of_setpoint(self, dataframe, percentage):
        """
        Filter the DataFrame to include only the rows where each 'Actual' value
        lies within a certain percentage of its corresponding 'Setpoint' value.

        Parameters:
            dataframe (DataFrame): The input DataFrame containing 'Actual' and 'Setpoint' columns.
            percentage (float): The percentage within which the 'Actual' value should lie relative to the 'Setpoint'.

        Returns:
            DataFrame: A new DataFrame consisting only of the rows where each 'Actual'
            value lies within a certain percentage of its corresponding 'Setpoint' value.
        """
        # List to store the indices of matching rows
        matching_indices = []
        
        # Iterate over each row in the DataFrame
        for index, row in dataframe.iterrows():
            # Initialize a flag to indicate whether all 'Actual' values meet the criteria
            within_percentage = True
            
            # Iterate over 'Actual' columns
            for col in dataframe.columns:
                if col.endswith('Actual'):
                    # Get the corresponding 'Setpoint' column
                    setpoint_col = col.replace('Actual', 'Setpoint')
                    # Calculate the threshold based on the percentage
                    threshold = row[setpoint_col] * (percentage / 100)
                    # Check if the 'Actual' value is within the threshold
                    if abs(row[col] - row[setpoint_col]) > threshold:
                        within_percentage = False
                        break
            
            # If all 'Actual' values meet the criteria, add the index of the row to the list
            if within_percentage:
                matching_indices.append(index)
        
        # Create a new DataFrame consisting only of the rows that matched the criteria
        filtered_df = dataframe.loc[matching_indices]
        
        return filtered_df
    

    def count_zeros_in_data(self, df, col_suffix):
        # Initialize an empty dictionary to store counts for each 'Actual' column
        zero_counts = {}

        # Iterate over the columns
        for column in df.columns.levels[1]:
            # Check if the second level of the MultiIndex is 'Actual'
            if column == col_suffix:
                # Get the column data
                column_data = df.xs((col_suffix), axis=1, level=1)

                # Count the number of zeros in the column
                zero_count = (column_data == 0).sum()
                # print(zero_count)
                # Store the count in the dictionary
                zero_counts[column] = zero_count

        return zero_counts
    
    def count_unique_for_setpoint(self, df):
        # Initialize an empty dictionary to store unique value counts for each 'Setpoint' column
        unique_counts = {}

        # Iterate over the columns
        for column in df.columns.levels[1]:
            # Check if the second level of the MultiIndex is 'Setpoint'
            if column == 'Setpoint':
                # Get the column data
                column_data = df.xs(('Setpoint'), axis=1, level=1)
            # Count the number of unique values and store them in a dictionary
                unique_counts[column] = {
                    'unique_values': column_data.drop_duplicates(),
                    'count': column_data.nunique()
                }

        return unique_counts
    