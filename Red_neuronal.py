# a. Importación de bibliotecas necesarias
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers

Adam = optimizers.Adam
from sklearn.model_selection import train_test_split

# b. Carga y preprocesamiento de los datos desde un archivo CSV
df = pd.read_csv('synthetic_data_with_outliers.csv')
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.values)

# Dividir los datos en entrenamiento y prueba
X_train, X_test = train_test_split(scaled_features, test_size=0.2, random_state=42)

# c. Definición y compilación del modelo de red neuronal adecuado para la detección de anomalías
input_dim = X_train.shape[1]
encoding_dim = 14  # Dimensión de la capa codificada

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Continuación: Definición y compilación del modelo de red neuronal
input_dim = X_train.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Entrenamiento del modelo
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test),
                verbose=1)

# Detección de valores atípicos
X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
threshold = np.quantile(mse, 0.95)
outliers = mse > threshold

import matplotlib.pyplot as plt

# Asumiendo que 'feature1' y 'feature2' son las primeras dos características
feature1 = X_test[:, 0]
feature2 = X_test[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(feature1[~outliers], feature2[~outliers], label='Normal', alpha=0.7)
plt.scatter(feature1[outliers], feature2[outliers], color='r', label='Outlier', alpha=0.7)
plt.title('Visualización de Valores Atípicos')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

