import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from data_normalization import normalize_data
from missing_values import handle_missing_values

file_path = 'creditcard.csv'
data = pd.read_csv(file_path)
data = handle_missing_values(data)
data = normalize_data(data)

X = data.drop('Class', axis=1)
y = data['Class']

# Split data in training set and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train.values.ravel())
selector = SelectFromModel(rf, prefit=True)
X_train_selected = selector.transform(X_train.values)
X_val_selected = selector.transform(X_val.values)

# PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

# Logistic Regression selected characteristics RandomForest
lr_selected = LogisticRegression(max_iter=1000, random_state=42)
lr_selected.fit(X_train_selected, y_train.values.ravel())
y_pred_selected = lr_selected.predict(X_val_selected)

# Logistic Regression selected characteristics PCA
lr_pca = LogisticRegression(max_iter=1000, random_state=42)
lr_pca.fit(X_train_pca, y_train.values.ravel())
y_pred_pca = lr_pca.predict(X_val_pca)

# Evaluation
print("Evaluation on selected characteristics by RandomForest")
print(classification_report(y_val, y_pred_selected))
print("ROC AUC Score: ", roc_auc_score(y_val, y_pred_selected))

print("Evaluation on selected characteristics by PCA")
print(classification_report(y_val, y_pred_pca))
print("ROC AUC Score: ", roc_auc_score(y_val, y_pred_pca))