import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

# Check if data file exists, if not create sample data
data_path = "data/german_credit.csv"
if not os.path.exists(data_path):
    os.makedirs("data", exist_ok=True)
    # Create sample dataset with required columns
    np.random.seed(42)
    n_samples = 1000
    sample_data = {
        'age': np.random.randint(18, 80, n_samples),
        'amount': np.random.randint(100, 10000, n_samples),
        'duration': np.random.randint(1, 72, n_samples),
        'credit_risk': np.random.randint(0, 2, n_samples)
    }
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv(data_path, index=False)

data = pd.read_csv(data_path)

print(data.head())
print(data.info())
data = data.dropna()
data = pd.get_dummies(data, drop_first=True)
X = data.drop("credit_risk", axis=1)
y = data["credit_risk"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Curve")
plt.show()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Curve")
plt.show()
pred = model.predict(X_test)
pred = (pred > 0.5).flatten()

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
