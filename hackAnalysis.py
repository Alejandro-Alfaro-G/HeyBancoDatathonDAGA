#initial commit

import pandas as pd
import numpy as np
import os


def codificar_columna(df, columna, ruta_csv, separador=';'):
    """
    Crea o carga un archivo de mapeo para una columna categórica,
    reemplaza comas por '|', asigna un ID numérico y actualiza la columna original con el ID.
    Los valores que no coincidan o NaN se codifican como 0.

    Args:
        df (pd.DataFrame): DataFrame original.
        columna (str): Nombre de la columna a codificar.
        ruta_csv (str): Ruta del archivo CSV de mapeo.
        separador (str): Separador usado en el CSV (por defecto ';').

    Returns:
        pd.DataFrame: DataFrame actualizado con la columna codificada.
    """

    # Si el archivo no existe, crearlo
    if not os.path.exists(ruta_csv):
        valores_unicos = df[columna].dropna().unique()
        df_mapeo = pd.DataFrame({
            columna: [str(v).replace(',', '|').strip() for v in valores_unicos],
            'id': range(1, len(valores_unicos) + 1)
        })
        df_mapeo.to_csv(ruta_csv, sep=separador, index=False)

    # Leer el mapeo
    df_mapeo = pd.read_csv(ruta_csv, sep=separador)
    mapa = dict(zip(df_mapeo[columna], df_mapeo['id']))

    # Crear columna auxiliar con limpieza de texto
    columna_limpia = f"{columna}_limpia"
    df[columna_limpia] = df[columna].astype(str).str.replace(',', '|').str.strip()

    # Aplicar el mapeo con NaN → 0
    df[columna] = df[columna_limpia].map(mapa).fillna(0).astype(int)

    # Eliminar columna auxiliar
    df.drop(columns=[columna_limpia], inplace=True)

    return df

print("DATOS DE LA BASE DE CLIENTES")

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

df = codificar_columna(df, 'actividad_empresarial', 'HeyBancoDatathonDAGA/datos/actividades_empresariales.csv')

print(df.dtypes)
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

''' LIMPIEZA DE GIRO_COMERCIO '''

df_t = codificar_columna(df_t, 'giro_comercio', 'HeyBancoDatathonDAGA/datos/giro_comercio_codificado.csv')


''' LIMPIEZA DE COMERCIO '''
df_t = codificar_columna(df_t, 'comercio', 'HeyBancoDatathonDAGA/datos/comercios_codificados.csv')

dt_t.count()
print(df_t.dtypes)
