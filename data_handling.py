import pandas as pd
from sklearn.decomposition import PCA

from class_imbalance_handling import handle_class_imbalance
from data_normalization import normalize_data
from data_split import split_data
from missing_values import handle_missing_values


def process_data(file_path):
    data = pd.read_csv(file_path)
    data = handle_missing_values(data)
    # perform_eda(data)
    data = normalize_data(data)
    # Apply PCA if required
    pca = PCA(n_components=0.95)
    data_pca = pca.fit_transform(data.drop('Class', axis=1))
    data_pca = pd.DataFrame(data_pca, columns=[f'PCA_{i}' for i in range(pca.n_components_)])
    data_pca['Class'] = data['Class']
    data = data_pca

    # Split the data
    split_data(data)

    # Handle class imbalance
    handle_class_imbalance()


def handle_data():
    dataset_choice = input("Choose the dataset ('creditcard' or 'baf'): ")
    file_path = 'creditcard.csv' if dataset_choice == 'creditcard' else 'Base.csv'
    process_data(file_path)
    print("Data processing complete")


if __name__ == "__main__":
    handle_data()