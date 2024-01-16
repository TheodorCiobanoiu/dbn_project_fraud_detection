import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline


def handle_class_imbalance():
    # Loading the training data
    X_train = pd.read_csv('datasets/X_train.csv')
    y_train = pd.read_csv('datasets/y_train.csv')

    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('nearmiss', NearMiss(version=1))
    ])

    X_train_hybrid, y_train_hybrid = pipeline.fit_resample(X_train, y_train)

    X_train_hybrid.to_csv('datasets/X_train_hybrid.csv', index=False)
    y_train_hybrid.to_csv('datasets/y_train_hybrid.csv', index=False)