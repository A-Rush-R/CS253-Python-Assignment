import pandas as pd
from sklearn.utils import resample

def undersample_df(dataset_path : str) -> pd.DataFrame :
    # Assuming your data is in a CSV file named 'data.csv'
    data = pd.read_csv(dataset_path)

    # Identify the target variable
    target_variable = 'Education'  

    # Get the number of data points for each target label
    target_counts = data[target_variable].value_counts()

    # Identify the class with the least number of data points
    min_class_size = target_counts.min()


    # Undersample each target class using random sampling to match the minimum class size
    undersampled_data = pd.DataFrame()
    for target_label in target_counts.index:
        # Select data points for this target label
        target_data = data[data[target_variable] == target_label]
        # Randomly sample data points to match the minimum class size
        undersampled_target = resample(target_data, replace=False, n_samples=min_class_size)
        # Append the undersampled target class to the final DataFrame
        undersampled_data = pd.concat([undersampled_data, undersampled_target], ignore_index=True)

    return undersample_df

