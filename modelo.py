# usen el siguiente comando para instalar las dependencias necesarias:
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn jupyterlab tqdm

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

# Cargar los datos de transacciones
transacciones = pd.read_csv('HeyBancoDatathonDAGA/datos/datadetrans.csv', delimiter=';')

# Cargar los datos de clientes
clientes = pd.read_csv('HeyBancoDatathonDAGA/datos/dataclientes.csv', delimiter=';')

# Unir la tabla de transacciones con la de clientes usando la clave 'id'
data = pd.merge(transacciones, clientes, on='id')

# Convertir variables categóricas en variables dummy
data = pd.get_dummies(data, columns=['comercio', 'giro_comercio', 'tipo_venta'])  # Suponiendo que estas son las variables categóricas

# Seleccionar características y variable objetivo
X = data.drop(['monto', 'id', 'fecha_nacimiento', 'fecha', 'fecha_alta'], axis=1)
y = data['monto']

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular errores
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')

# Gráfica de predicciones vs valores reales
plt.scatter(y_test, y_pred)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.grid(True)
plt.show()