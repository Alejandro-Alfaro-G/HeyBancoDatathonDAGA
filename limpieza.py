#initial commit

import pandas as pd
import numpy as np
import os


def codificar_columna(df, columna, ruta_csv, comercio=False, separador=';'):
    """
    Codifica una columna categórica (o una combinación con giro_comercio) con IDs únicos.
    
    Si comercio=True, también se utiliza la columna 'giro_comercio' para crear combinaciones únicas.

    Args:
        df (pd.DataFrame): DataFrame original.
        columna (str): Nombre de la columna a codificar.
        ruta_csv (str): Ruta donde guardar o cargar el mapeo.
        comercio (bool): Si True, codifica combinaciones de columna + giro_comercio.
        separador (str): Separador del CSV (default ';').

    Returns:
        pd.DataFrame: DataFrame con nueva columna codificada como {columna}_id.
    """

    # Generar archivo si no existe
    if not os.path.exists(ruta_csv):
        if comercio:
            if 'giro_comercio' not in df.columns:
                raise ValueError("Falta la columna 'giro_comercio' en el DataFrame.")

            df_temp = df[[columna, 'giro_comercio']].dropna().copy()
            df_temp[columna] = df_temp[columna].astype(str).str.replace(',', '|').str.strip()
            df_temp['giro_comercio'] = df_temp['giro_comercio'].astype(str).str.replace(',', '|').str.strip()
            df_temp = df_temp.drop_duplicates()
            df_temp['id'] = range(1, len(df_temp) + 1)
            df_temp.to_csv(ruta_csv, sep=separador, index=False)
        else:
            valores_unicos = df[columna].dropna().astype(str).str.replace(',', '|').str.strip().unique()
            df_mapeo = pd.DataFrame({
                columna: valores_unicos,
                'id': range(1, len(valores_unicos) + 1)
            })
            df_mapeo.to_csv(ruta_csv, sep=separador, index=False)

    # Cargar el mapeo
    df_mapeo = pd.read_csv(ruta_csv, sep=separador)

    if comercio:
        # Limpiar columnas en df original
        df[columna] = df[columna].astype(str).str.replace(',', '|').str.strip()
        df['giro_comercio'] = df['giro_comercio'].astype(str).str.replace(',', '|').str.strip()

        # Renombrar 'id' del mapeo para evitar colisiones con otras columnas
        df_mapeo = df_mapeo.rename(columns={'id': f'{columna}_id'})

        # Merge y asignación de ID
        df = df.merge(df_mapeo, how='left', on=[columna, 'giro_comercio'])

        # Llenar nulos con 0 y convertir a int
        df[f'{columna}_id'] = df[f'{columna}_id'].fillna(0).astype(int)
    else:
        mapa = dict(zip(df_mapeo[columna], df_mapeo['id']))
        # Crear columna auxiliar con limpieza de texto
        columna_limpia = f"{columna}_limpia"
        df[columna_limpia] = df[columna].astype(str).str.replace(',', '|').str.strip()

        # Aplicar el mapeo con NaN → 0
        df[columna] = df[columna_limpia].map(mapa).fillna(0).astype(int)

        # Eliminar columna auxiliar
        df.drop(columns=[columna_limpia], inplace=True)

    return df


''' DATOS DE LA BASE DE CLIENTES '''

print("DATOS DE LA BASE DE CLIENTES")

# Cargar el dataset
df = pd.read_csv('HeyBancoDatathonDAGA/datos/base_clientes_final.csv')

#Limpiar fecha_nacimiento y fecha_alta
df['fecha_nacimiento'] = pd.to_datetime(df['fecha_nacimiento'], format='%Y-%m-%d')
df['fecha_alta'] = pd.to_datetime(df['fecha_alta'], format='%Y-%m-%d')


#Limpiar genero y tipo_persona
df['genero'] = df['genero'].map({
    'F': 1,
    'M': 2,
    '': 0  # caso explícito de string vacío
}).fillna(0).astype(int)

df['tipo_persona'] = df['tipo_persona'].map({
    'Persona Fisica Sin Actividad Empresarial': 1,
    'Persona Fisica Con Actividad Empresarial': 2,
})

# Convertir 'actividad_empresarial' a valores categóricos
df = codificar_columna(df, 'actividad_empresarial', 'HeyBancoDatathonDAGA/datos/actividades_empresariales.csv', False)

# Calcular edad
df['edad'] = (pd.to_datetime('today') - df['fecha_nacimiento']).dt.days // 365

print(df.dtypes)
print(df.describe())


''' DATOS DE LA BASE DE TRANSACCIONES '''

print("DATOS DE LA BASE DE TRANSACCIONES")

# Carga del dataset
df_t = pd.read_csv('HeyBancoDatathonDAGA/datos/base_transacciones_final.csv')

##LIMPIEZA DE DATOS
df_t['fecha'] = pd.to_datetime(df_t['fecha'], format='%Y-%m-%d')

# Convert 'tipo_venta' to categorical values 1: fisica, 0: digital
df_t['tipo_venta'] = df_t['tipo_venta'].map({
    'digital': 1,
    'fisica': 2,
})

# Convertir 'comercio' a valores categóricos, poniendo 'giro_comercio' como atributo
df_t = codificar_columna(df_t, 'comercio', 'HeyBancoDatathonDAGA/datos/comercios_codificados.csv', True)

print(df_t.describe())
print(df_t.dtypes)

# Guardar los DataFrames limpios
df_t.to_csv('HeyBancoDatathonDAGA/datos/datadetrans.csv', sep=';', index=False)
df.to_csv('HeyBancoDatathonDAGA/datos/dataclientes.csv', sep=';', index=False)




df_u = pd.merge(df,df_t , on="id")


# Ordenar por cliente y fecha (aunque ya esté ordenado, es una precaución)
df_u = df_u.sort_values(by=["id", "fecha"])

# --- Paso 1: Generar Identificador Único de Transacción ---
# Crear secuencia de 4 dígitos por cliente (empezando en 0001)
df_u["secuencia"] = df_u.groupby("id").cumcount() + 1  # +1 para que empiece en 1

# Formatear secuencia con ceros a la izquierda (4 dígitos)
df_u["secuencia_str"] = df_u["secuencia"].astype(str).str.zfill(4)

# Combinar "id" (string) + secuencia_str para crear el identificador único
df_u["id_transaccion"] = df_u["id"] + df_u["secuencia_str"]

# Eliminar columnas temporales
df_u = df_u.drop(columns=["secuencia", "secuencia_str"])

# --- Paso 2: Calcular "Días desde Última Transacción" ---
# Calcular diferencia de días con la transacción anterior del mismo cliente
df_u["dias_desde_ultima_transaccion"] = (
    df_u.groupby("id")["fecha"]
    .diff()                   # Diferencia entre fechas consecutivas
    .dt.days                  # Convertir a días
    .fillna(0)                # Primera transacción = 0
)

# --- Verificación Final ---
# 1. Asegurar que id_transaccion sea único
print("¿IDs de transacción únicos?:", df_u["id_transaccion"].is_unique)  # Debe ser True

# 2. Verificar primera transacción de un cliente (ejemplo)
cliente_ejemplo = df_u["id"].iloc[0]
print("\nEjemplo para cliente:", cliente_ejemplo)
print(df_u[df_u["id"] == cliente_ejemplo][["id", "fecha", "dias_desde_ultima_transaccion"]].head(2))

## seccion de analisis de primera fase
df=df_u

df = df.drop(columns=["id_municipio", "id_estado"])

df["fecha"] = pd.to_datetime(df["fecha"])
fecha_max = df["fecha"].max()  # Fecha más reciente en el dataset
df["compras_ultimos_30d"] = df.groupby("id")["fecha"].transform(
    lambda x: x[(x >= fecha_max - pd.DateOffset(days=30))].count()
)

df["es_digital"] = df["tipo_venta"].apply(lambda x: 1 if x == 1 else 0)
df["pct_digital"] = df.groupby("id")["es_digital"].transform("mean") * 100

df["recencia"] = (fecha_max - df.groupby("id")["fecha"].transform("max")).dt.days

df["monto_promedio"] = df.groupby("id")["monto"].transform("mean")

df["dia_semana"] = df["fecha"].dt.dayofweek  # 0=Lunes, 6=Domingo
df["mes"] = df["fecha"].dt.month

# Ordenar datos por cliente y fecha
df = df.sort_values(by=["id", "fecha"])

# Calcular días hasta la próxima transacción (dentro de 30 días)
df["dias_hasta_proxima"] = df.groupby("id")["fecha"].shift(-1) - df["fecha"]
df["dias_hasta_proxima"] = df["dias_hasta_proxima"].dt.days

# Si la próxima transacción es fuera de 30 días, establecer como 30
df.loc[df["dias_hasta_proxima"] > 30, "dias_hasta_proxima"] = 30

# Si no hay próxima transacción, usar 30 (asumir que no comprará en el mes)
df["dias_hasta_proxima"] = df["dias_hasta_proxima"].fillna(30)



df_modelo = df.drop(columns=[
    "comercio", "giro_comercio", "comercio_id", "id_transaccion",
    "fecha_nacimiento", "fecha_alta", "fecha", "es_digital"
])

df_modelo = pd.get_dummies(df_modelo, columns=["genero", "actividad_empresarial", "tipo_persona"])

from sklearn.model_selection import train_test_split
X = df_modelo.drop(columns=["dias_hasta_proxima"])
y = df_modelo["dias_hasta_proxima"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





## FASE DE MODELADO CON XGBREGRESSOR
'''
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200]
}
grid = GridSearchCV(XGBRegressor(), param_grid, cv=3)
grid.fit(X_train, y_train)

import joblib
joblib.dump(model, "modelo_dias_hasta_proxima.pkl")

def predecir(nuevos_datos):
    modelo = joblib.load("modelo_dias_hasta_proxima.pkl")
    return modelo.predict(nuevos_datos)

'''



# Eliminar columnas no relevantes (incluyendo 'id')
df_modelo = df.drop(columns=[
    "id", "comercio", "giro_comercio", "id_transaccion",
    "fecha_nacimiento", "fecha_alta", "fecha", "es_digital"
])

# Codificar variables categóricas con pd.get_dummies
df_modelo = pd.get_dummies(
    df_modelo, 
    columns=["genero", "actividad_empresarial", "tipo_persona" ,"comercio_id"], 
    drop_first=True  # Evitar multicolinealidad
)

# Verificar tipos de datos
print(df_modelo.dtypes)  # Todas deben ser numéricas (int/float)
X = df_modelo.drop(columns=["dias_hasta_proxima"])
y = df_modelo["dias_hasta_proxima"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Entrenar modelo
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # <--- Calcular RMSE así

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")







'''
importances = model.feature_importances_
features = X.columns
feat_importances = pd.Series(importances, index=features)
feat_importances.nlargest(10).plot(kind='barh')

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
grid.fit(X_train, y_train)
print("Mejores parámetros:", grid.best_params_)'''

df.to_csv('HeyBancoDatathonDAGA/datos/dataclientesMEGA.csv')

