import matplotlib.pyplot as plt
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
        metrics=['accuracy'])
    return model


# KerasClassifier for cross-validation
model_cv = KerasClassifier(
    build_fn=lambda: build_dbn_model((X_train.shape[1],)),
    epochs=100,
    batch_size=4000,
    verbose=0)

# KFold cross-validation
k_fold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42)
results = cross_val_score(model_cv, X_train, y_train, cv=k_fold)
print("Cross-Validation Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

# Early Stopping and ReduceLROnPlateau callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.00001)

# Train the model with the whole training set
dbn_model = build_dbn_model((X_train.shape[1],))
dbn_model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=6000,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr]
)
# Evaluate the model on the validation set with custom threshold
threshold = 0.8
y_val_probs = dbn_model.predict(X_val)
y_val_pred = (y_val_probs > threshold).astype(int)
print("Classification report at threshold", threshold)
print(classification_report(y_val, y_val_pred))
print("ROC AUC Score:", roc_auc_score(y_val, y_val_probs))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, y_val_probs)
plt.figure()
plt.plot(recall, precision, marker='.', label='DBN')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()

# Calculating AUC for Precision-Recall Curve
auc_score = auc(recall, precision)
print("Precision-Recall AUC:", auc_score)