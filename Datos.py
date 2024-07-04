import numpy as np
import pandas as pd
import random

# Configurar la semilla aleatoria para reproducibilidad
np.random.seed(42)

# Generar datos sintéticos
num_rows = 1000
feature1 = np.random.normal(loc=0, scale=1, size=num_rows)  # Distribución normal
feature2 = np.random.uniform(low=-2, high=2, size=num_rows)  # Distribución uniforme
feature3 = np.random.lognormal(mean=0, sigma=1, size=num_rows)  # Distribución log-normal

# Crear DataFrame
df = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'feature3': feature3
})

# Introducir valores atípicos
outliers_count = int(0.05 * num_rows)  # 5% de valores atípicos
for _ in range(outliers_count):
    idx = random.randint(0, num_rows-1)  # Seleccionar un índice aleatorio
    col = random.choice(['feature1', 'feature2', 'feature3'])  # Seleccionar una columna aleatoria
    df.at[idx, col] = df[col].mean() + 10 * df[col].std()  # Añadir valor atípico

# Guardar en CSV
df.to_csv('synthetic_data_with_outliers.csv', index=False)