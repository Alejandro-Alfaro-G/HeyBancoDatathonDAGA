#initial commit

import pandas as pd
import numpy as np

print("DATOS DE LA BASE DE CLIENTES")
# Load the dataset
df = pd.read_csv('HeyBancoDatathonDAGA/datos/base_clientes_final.csv')

print("limpieza de datos")

df['fecha_nacimiento'] = pd.to_datetime(df['fecha_nacimiento'], format='%Y-%m-%d')
df['fecha_alta'] = pd.to_datetime(df['fecha_alta'], format='%Y-%m-%d')

df['genero'] = df['genero'].map({
    'F': 1,
    'M': 2,
    ' ': np.nan,  # Convert empty strings to NaN
})

df['tipo_persona'] = df['tipo_persona'].map({
    'Persona Fisica Sin Actividad Empresarial': 1,
    'Persona Fisica Con Actividad Empresarial': 2,
})

#Convertir 'actividad_empresarial' to categorical values from CSV file
df_aux = pd.read_csv('HeyBancoDatathonDAGA/datos/actividades_empresariales.csv')
mapa_aux = dict(zip(df_aux['actividad_empresarial'], df_aux['id']))
df['actividad_empresarial'] = df['actividad_empresarial'].map(mapa_aux)

#print(df.dtypes)
#print(df.describe())

print("DATOS DE LA BASE DE TRANSACCIONES")

# Load the dataset
df_t = pd.read_csv('HeyBancoDatathonDAGA/datos/base_transacciones_final.csv')

# Display the shape of the dataset
#print(df_t.shape)
# Display the columns of the dataset
#print(df_t.columns)
# Display the data types of the columns
#print(df_t.dtypes)
# Display the summary statistics of the dataset
#print(df_t.describe())


##LIMPIEZA DE DATOS
print("limpieza de datos")
df_t['fecha'] = pd.to_datetime(df_t['fecha'], format='%Y-%m-%d')

# Convert 'tipo_venta' to categorical values 1: fisica, 0: digital
df_t['tipo_venta'] = df_t['tipo_venta'].map({
    'digital': 0,
    'fisica': 1,
})


df_aux = pd.read_csv("HeyBancoDatathonDAGA/datos/relacionalGiros.csv", sep=';', header=None, names=['giro', 'id'])
df_aux['giro'] = df_aux['giro'].astype(str).str.replace('|', ',', regex=False).str.strip()
mapa_aux = dict(zip(df_aux['giro'], df_aux['id']))
df_t['giro_comercio'] = df_t['giro_comercio'].map(mapa_aux)

''' CORRER SOLO SI NO EXISTE EL CSV
# 1. Obtener valores únicos
#valores_unicos = df_t['comercio'].dropna().unique()

# 2. Crear DataFrame con reemplazo de comas y numeración desde 1
df_aux = pd.DataFrame({
    'comercio': [str(v).replace(',', '|') for v in valores_unicos],
    'id': range(1, len(valores_unicos) + 1)
})

# 3. Guardar a CSV con ; como separador
#df_aux.to_csv('HeyBancoDatathonDAGA/datos/comercios_codificados.csv', sep=';', index=False)
'''

# 4. Cargar el CSV como tabla de mapeo
df_aux = pd.read_csv('HeyBancoDatathonDAGA/datos/comercios_codificados.csv', sep=';')

#5. Crear un diccionario de mapeo
mapa_aux = dict(zip(df_aux['comercio'], df_aux['id']))

# 6. Preparar columna original reemplazando comas por '|'
df_t['comercio_limpio'] = df_t['comercio'].astype(str).str.replace(',', '|').str.strip()

# 7. Aplicar mapeo y generar columna con ID
df_t['comercio'] = df_t['comercio_limpio'].map(mapa_aux)

# 8. Eliminar columna auxiliar
df_t.drop(columns=['comercio_limpio'], inplace=True)

#print(df_t['giro_comercio'].value_counts())

print(df_t.dtypes)
