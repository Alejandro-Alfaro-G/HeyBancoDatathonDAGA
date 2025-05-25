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

pd.set_option('display.max_rows', None)

# Cargar los datos de transacciones
transacciones = pd.read_csv('HeyBancoDatathonDAGA/datos/datadetrans.csv', delimiter=';')

# Cargar los datos de clientes
clientes = pd.read_csv('HeyBancoDatathonDAGA/datos/dataclientes.csv', delimiter=';')

# Unir la tabla de transacciones con la de clientes usando la clave 'id'
data = pd.merge(transacciones, clientes, on='id')

# Convertir variables categóricas en variables dummy
data = pd.get_dummies(data, columns=['comercio_id', 'actividad_empresarial_id', 'tipo_venta', 'tipo_persona', 'genero'])  # Suponiendo que estas son las variables categóricas

# Seleccionar características y variable objetivo
X = data.drop(['monto', 'id', 'dia', 'mes'], axis=1)
y = data['monto']

X = X.sample(3000, random_state=42)
y = y.loc[X.index]

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ENTRENAR RANDOM FOREST PARA VER IMPORTANCIA DE VARIABLES
rf = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

importancia = pd.Series(rf.feature_importances_, index=X.columns)
importancia.sort_values(ascending=False).head(20).plot(kind='barh', figsize=(10, 6))
plt.title("Top 20 Variables más importantes (Random Forest)")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()

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

# Elegir 30 índices aleatorios del conjunto de prueba
indices = np.random.choice(len(y_test), size=30, replace=False)

# Subconjuntos para graficar
y_test_sample = y_test.iloc[indices]
y_pred_sample = y_pred[indices]

# Gráfica
plt.figure(figsize=(8, 6))
plt.scatter(range(30), y_test_sample, color='blue', label='Valor Real', alpha=0.6)
plt.scatter(range(30), y_pred_sample, color='orange', label='Predicción', alpha=0.6)
plt.xlabel('Índice de muestra')
plt.ylabel('Monto')
plt.title('Valor Real vs Predicción (30 muestras aleatorias)')
plt.legend()
plt.grid(True)
plt.show()

# Obtener coeficientes e intercepto
coeficientes = model.coef_
intercepto = model.intercept_

# Obtener los nombres de las variables (columnas)
nombres_columnas = X.columns

# Imprimir la ecuación de la regresión
print("Ecuación de la regresión lineal:")
ecuacion = f"monto = {intercepto:.4f}"
for nombre, coef in zip(nombres_columnas, coeficientes):
    ecuacion += f" + ({coef:.4f} * {nombre})"
print(ecuacion)


# Gráfica de predicciones vs valores reales
'''plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valor Real', alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, color='orange', label='Predicción', alpha=0.6)
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title('Valor Real vs Predicción')
plt.legend()
plt.grid(True)
plt.show()'''

