from sklearn.model_selection import train_test_split


# Loading the normalized dataset
def split_data(data):
    # Separating features (X) and target variable (y)
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Splitting the dataset into training (60%), validation (20%), and testing (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Saving the datasets to CSV files in the specified folder
    X_train.to_csv('datasets/X_train.csv', index=False)
    y_train.to_csv('datasets/y_train.csv', index=False)
    X_val.to_csv('datasets/X_val.csv', index=False)
    y_val.to_csv('datasets/y_val.csv', index=False)
    X_test.to_csv('datasets/X_test.csv', index=False)
    y_test.to_csv('datasets/y_test.csv', index=False)