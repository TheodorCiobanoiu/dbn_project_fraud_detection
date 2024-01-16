import pandas as pd


# SCRIPT TO CHECK FOR MISSING VALUES AND, IF NECESSARY, HANDLE THE MISSING DATA
def handle_missing_values(data):
    # Dataset loading
    file_path = 'creditcard.csv'
    data = pd.read_csv(file_path)

    # Check for missing values in each column
    missing_values = data.isnull().sum()

    # Results
    print("Number of missing values:")
    print(missing_values)

    if missing_values.any():
        print("\nThere are missing values in the dataset. Beginning the correction process")
        # In case there were missing values, we could've handle this scenario by replacing missing values with the
        # median or mode, or dropping rows/columns with missing values. Here is an example of how we would've replaced
        # missing values if we had any in the "Amount" column with the median
        median_value = data['Amount'].median()
        data['Amount'].fillna(median_value, inplace=True)
    else:
        print("\nNo missing values in the dataset, therefore there is no need to process the data further")

    return data