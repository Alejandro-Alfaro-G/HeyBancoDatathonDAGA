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

df = pd.read_csv('HeyBancoDatathonDAGA/datos/base_clientes_final.csv')

print("limpieza de datos")

df['fecha_nacimiento'] = pd.to_datetime(df['fecha_nacimiento'], format='%Y-%m-%d')
df['fecha_alta'] = pd.to_datetime(df['fecha_alta'], format='%Y-%m-%d')

df['genero'] = df['genero'].map({
    'F': 1,
    'M': 2,
    '': 0  # caso explícito de string vacío
}).fillna(0).astype(int)

df['tipo_persona'] = df['tipo_persona'].map({
    'Persona Fisica Sin Actividad Empresarial': 1,
    'Persona Fisica Con Actividad Empresarial': 2,
})

df = codificar_columna(df, 'actividad_empresarial', 'HeyBancoDatathonDAGA/datos/actividades_empresariales.csv', False)

df['edad'] = (pd.to_datetime('today') - df['fecha_nacimiento']).dt.days // 365

print(df.dtypes)
print(df.describe())


''' DATOS DE LA BASE DE TRANSACCIONES '''

print("DATOS DE LA BASE DE TRANSACCIONES")

# Load the dataset
df_t = pd.read_csv('HeyBancoDatathonDAGA/datos/base_transacciones_final.csv')

# Display the shape of the dataset
#print(df_t.shape)
# Display the columns of the dataset
#print(df_t.columns)

##LIMPIEZA DE DATOS
print("limpieza de datos")
df_t['fecha'] = pd.to_datetime(df_t['fecha'], format='%Y-%m-%d')

# Convert 'tipo_venta' to categorical values 1: fisica, 0: digital
df_t['tipo_venta'] = df_t['tipo_venta'].map({
    'digital': 1,
    'fisica': 2,
})


df_t = codificar_columna(df_t, 'comercio', 'HeyBancoDatathonDAGA/datos/comercios_codificados.csv', True)

print(df_t.describe())
print(df_t.dtypes)


df_t.to_csv('HeyBancoDatathonDAGA/datos/datadetrans.csv', sep=';', index=False)