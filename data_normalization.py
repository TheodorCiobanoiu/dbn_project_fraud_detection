from sklearn.preprocessing import StandardScaler


# SCRIPT FOR NORMALIZING THE DATA, FOR THE COLUMNS "AMOUNT" AND "TIME"
def normalize_data(data):
    # Initializing the StandardScaler
    scaler = StandardScaler()

    # Normalizing the 'Time' and 'Amount' columns
    data['NormalizedTime'] = scaler.fit_transform(data[['Time']])
    data['NormalizedAmount'] = scaler.fit_transform(data[['Amount']])

    # Dropping the original 'Time' and 'Amount' columns
    data = data.drop(['Time', 'Amount'], axis=1)

    # Displaying the first few rows of the updated dataset
    print("First few rows of the normalized dataset:")
    print(data.head())

    return data