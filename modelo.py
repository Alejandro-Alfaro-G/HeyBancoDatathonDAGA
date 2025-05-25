# usen el siguiente comando para instalar las dependencias necesarias:
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn jupyterlab tqdm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Mostrar todas las filas si lo necesitas
pd.set_option('display.max_rows', None)

# Cargar los datos
transacciones = pd.read_csv('HeyBancoDatathonDAGA/datos/datadetrans.csv', delimiter=';')
clientes = pd.read_csv('HeyBancoDatathonDAGA/datos/dataclientes.csv', delimiter=';')

# Unir por 'id'
data = pd.merge(transacciones, clientes, on='id')

# Convertir variables categóricas en dummies
data = pd.get_dummies(data, columns=['comercio_id', 'actividad_empresarial_id', 'tipo_venta',
                                     'tipo_persona', 'genero', 'id_municipio', 'id_estado'])

# Eliminar columnas no relevantes y definir X, y
X = data.drop(['monto', 'fecha', 'id', 'dia', 'mes'], axis=1)
y = data['monto']

# Submuestreo para velocidad
X = X.sample(3000, random_state=42)
y = y.loc[X.index]

# División preliminar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo de Random Forest
rf = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Importancia de variables
importancia = pd.Series(rf.feature_importances_, index=X.columns)
importancia.sort_values(ascending=False).head(20).plot(kind='barh', figsize=(10, 6))
plt.title("Top 20 Variables más importantes (Random Forest)")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()

# FILTRAR VARIABLES SEGÚN IMPORTANCIA
umbral = 0.01
columnas_importantes = importancia[importancia > umbral].index.tolist()
X = X[columnas_importantes]

# Dividir de nuevo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Métricas de error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'MAE: {mae}')
print(f'MSE: {mse}')

# Gráfica de 30 muestras aleatorias
indices = np.random.choice(len(y_test), size=30, replace=False)
y_test_sample = y_test.iloc[indices]
y_pred_sample = y_pred[indices]

plt.figure(figsize=(8, 6))
plt.scatter(range(30), y_test_sample, color='blue', label='Valor Real', alpha=0.6)
plt.scatter(range(30), y_pred_sample, color='orange', label='Predicción', alpha=0.6)
plt.xlabel('Índice de muestra')
plt.ylabel('Monto')
plt.title('Valor Real vs Predicción (30 muestras aleatorias)')
plt.legend()
plt.grid(True)
plt.show()

# Imprimir ecuación de la regresión
coeficientes = model.coef_
intercepto = model.intercept_
nombres_columnas = X.columns

print("\nEcuación de la regresión lineal:")
ecuacion = f"monto = {intercepto:.4f}"
for nombre, coef in zip(nombres_columnas, coeficientes):
    ecuacion += f" + ({coef:.4f} * {nombre})"
print(ecuacion)
