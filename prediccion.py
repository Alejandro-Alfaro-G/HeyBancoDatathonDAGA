import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# Cargar los datos de transacciones y clientes
transacciones = pd.read_csv('HeyBancoDatathonDAGA/datos/datadetrans.csv', delimiter=';')
clientes = pd.read_csv('HeyBancoDatathonDAGA/datos/dataclientes.csv', delimiter=';')


# Unir la tabla de transacciones con la de clientes usando la clave 'id'
data = pd.merge(transacciones, clientes, on='id')

# Convertir las fechas a datetime
data['fecha'] = pd.to_datetime(data['fecha'])

# Ordenar los datos por 'id' y 'fecha'
data = data.sort_values(by=['id', 'fecha'])

# Calcular el intervalo de tiempo entre transacciones sucesivas para cada cliente
data['intervalo'] = data.groupby('id')['fecha'].diff().dt.days

# Remover valores nulos que ocurren debido a la diferencia de la primera compra
data = data.dropna()

# Crear una columna para la próxima compra ('compra_siguiente') usando shift()
data['compra_siguiente'] = data.groupby('id')['comercio'].shift(-1)

# Remover filas con valores nulos en la columna 'compra_siguiente'
data = data.dropna(subset=['compra_siguiente'])

# Convertir variables categóricas en variables dummy
data = pd.get_dummies(data, columns=['actividad_empresarial', 'giro_comercio', 'compra_siguiente'])

# Predecir Cuándo Será la Próxima Compra
X_time = data.drop(['id', 'fecha', 'intervalo', 'monto', 'compra_siguiente'], axis=1)
y_time = data['intervalo']

X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(X_time, y_time, test_size=0.2, random_state=42)

model_time = RandomForestRegressor(n_estimators=100, random_state=42)
model_time.fit(X_train_time, y_train_time)
y_pred_time = model_time.predict(X_test_time)

mae_time = mean_absolute_error(y_test_time, y_pred_time)
mse_time = mean_squared_error(y_test_time, y_pred_time)

print(f'MAE (Tiempo): {mae_time}')
print(f'MSE (Tiempo): {mse_time}')

# Predecir Cuánto Será la Próxima Compra
X_amount = data.drop(['id', 'fecha', 'intervalo', 'monto', 'compra_siguiente'], axis=1)
y_amount = data['monto']

X_train_amount, X_test_amount, y_train_amount, y_test_amount = train_test_split(X_amount, y_amount, test_size=0.2, random_state=42)

model_amount = RandomForestRegressor(n_estimators=100, random_state=42)
model_amount.fit(X_train_amount, y_train_amount)
y_pred_amount = model_amount.predict(X_test_amount)

mae_amount = mean_absolute_error(y_test_amount, y_pred_amount)
mse_amount = mean_squared_error(y_test_amount, y_pred_amount)

print(f'MAE (Monto): {mae_amount}')
print(f'MSE (Monto): {mse_amount}')

# Predecir Dónde Será la Próxima Compra
X_location = data.drop(['id', 'fecha', 'intervalo', 'monto', 'compra_siguiente'], axis=1)
y_location = data.filter(regex='^compra_siguiente_', axis=1).idxmax(axis=1)

X_train_location, X_test_location, y_train_location, y_test_location = train_test_split(X_location, y_location, test_size=0.2, random_state=42)

model_location = RandomForestClassifier(n_estimators=100, random_state=42)
model_location.fit(X_train_location, y_train_location)
y_pred_location = model_location.predict(X_test_location)

accuracy_location = accuracy_score(y_test_location, y_pred_location)
print(f'Accuracy (Ubicación): {accuracy_location}')

print(classification_report(y_test_location, y_pred_location))

conf_matrix_location = confusion_matrix(y_test_location, y_pred_location)
print('Confusion Matrix (Ubicación):')
print(conf_matrix_location)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_location, annot=True, fmt='d', cmap='Blues', xticklabels=model_location.classes_, yticklabels=model_location.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Ubicación)')
plt.show()