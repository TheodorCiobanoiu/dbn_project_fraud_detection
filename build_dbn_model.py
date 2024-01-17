import time

import pandas as pd
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import cross_val_score, KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from data_handling import handle_data

# Process the raw dataset
handle_data()

# Load the processed data
X_train_path = 'datasets/X_train_hybrid.csv'
y_train_path = 'datasets/y_train_hybrid.csv'
X_val_path = 'datasets/X_val.csv'
y_val_path = 'datasets/y_val.csv'

X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)
X_val = pd.read_csv(X_val_path)
y_val = pd.read_csv(y_val_path)


# Build the DBN model
def build_dbn_model(input_shape):
    model = Sequential()
    model.add(Dense(256, input_shape=input_shape, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


print("Cross-Validation process started")
# KerasClassifier for cross-validation
model_cv = KerasClassifier(
    build_fn=lambda: build_dbn_model((X_train.shape[1],)),
    epochs=100,
    batch_size=7500,
    verbose=0
)

# KFold cross-validation
k_fold = KFold(n_splits=5,
               shuffle=True,
               random_state=42
               )
cv_results = cross_val_score(model_cv, X_train, y_train, cv=k_fold)
print("Cross-Validation process finished")

start_training_time = time.time()
# Train and evaluate the model
dbn_model = build_dbn_model((X_train.shape[1],))
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=20,
    min_lr=0.00001)
dbn_model.fit(X_train, y_train,
              epochs=400,
              batch_size=2000,
              validation_data=(X_val, y_val),
              callbacks=[early_stopping, reduce_lr]
              )
end_training_time = time.time()
training_time = end_training_time - start_training_time

threshold = 0.8
y_val_probs = dbn_model.predict(X_val)
y_val_pred = (y_val_probs > threshold).astype(int)

# Metrics calculation
classification_rep = classification_report(y_val, y_val_pred, output_dict=True)
roc_auc = roc_auc_score(y_val, y_val_probs)
precision, recall, _ = precision_recall_curve(y_val, y_val_probs)
auc_pr = auc(recall, precision)

print("Cross-Validation Accuracy: %.2f%% (%.2f%%)" % (cv_results.mean() * 100, cv_results.std() * 100))
print("Classification report at threshold", threshold)
print(classification_report(y_val, y_val_pred))
print("ROC AUC Score:", roc_auc)
print("Precision-Recall AUC:", auc_pr)
print(training_time)

report_df = pd.DataFrame(classification_rep).transpose()
with (pd.ExcelWriter('model_evaluation.xlsx') as writer):
    report_df.to_excel(writer, sheet_name='Classification Report')
    pd.DataFrame({
        'Precision-Recall AUC': [auc_pr],
        'ROC AUC Score': [roc_auc],
        # 'Cross-Validation Accuracy': [cv_results.mean()],
        'Training time': [training_time],
    }).to_excel(writer, sheet_name='Other Metrics', index=False)
print("Results saved to Excel file.")
