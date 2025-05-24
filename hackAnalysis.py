#initial commit

import pandas as pd
import numpy as np

print("DATOS DE LA BASE DE CLIENTES")
# Load the dataset
df = pd.read_csv('HeyBancoDatathonDAGA/datos/base_clientes_final.csv')
# Display the shape of the dataset
#print(df.shape)
# Display the columns of the dataset
#print(df.columns)     

#valores_unicos = sorted(df['actividad_empresarial'].unique())
#print("Valores unicos :", valores_unicos)
#print(df['genero'].value_counts())

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


actividades = [
    'ADMINISTRACION DE INMUEBLES',
    'AGENCIA DE PUBLICIDAD',
    'AGENTE DE SEGUROS',
    'AMA DE CASA',
    'COMPRA VENTA DE ARTICULOS NO CLASIFICADOS EN OTRA PARTE',
    'COMPRA VENTA DE HARDWARE SOFTWARE Y ARTCULOS COMPUTACIONALES',
    'COMPRA VENTA DE ROPA',
    'COMPRAVENTA DE ARTICULOS DE FERRETERIA',
    'CONSTRUCCION DE INMUEBLES',
    'DESPACHO DE OTROS PROFESIONISTAS',
    'EMPLEADO DE GOBIERNO',
    'EMPLEADO DEL SECTOR INDUSTRIAL',
    'EMPLEADO DEL SECTOR SERVICIOS',
    'ESTABLECIMIENTOS PRIVADOS DE INSTRUCCION EDUCACION CULTURA E INVESTIGACION',
    'ESTABLECIMIENTOS PUBLICOS DE INSTRUCCION EDUCACION SUBPROFESIONAL YPROFESIONAL CULTURA E INVESTIGACION',
    'ESTUDIANTE',
    'JUBILADO',
    'ORGANIZACIONES DE ABOGADOS MEDICOS INGENIEROS Y OTRAS ASOCIACIONES DE PROFESIONALES',
    'PREPARACION DE TIERRAS DE CULTIVO Y OTROS SERVICIOS AGRICOLAS',
    'PRESTACION DE OTROS SERVICIOS TECNICOS',
    'PROFESIONISTA INDEPENDIENTE',
    'SERVICIOS ADMINISTRATIVOS DE TRAMITE Y COBRANZA INCLUSO ESCRITORIOS PUBLICOS',
    'SERVICIOS DE ANALISIS DE SISTEMAS Y PROCESAMIENTO ELECTRONICO DE DATOS',
    'SERVICIOS DE ASESORIA Y ESTUDIOS TECNICOS DE ARQUITECTURA E INGENIERIA INCLUSO DISEO INDUSTRIAL',
    'SERVICIOS DE CONTADURIA Y AUDITORIA INCLUSO TENEDURIA DE LIBROS',
    'SERVICIOS MEDICO GENERAL Y ESPECIALIZADO EN CONSULTORIOS',
    'TELEDIFUSORA',
    'TIENDA DE ABARROTES Y MISCELANEA',
    'TRANSPORTE DE CARGA FORANEA'
]

# Crear diccionario de mapeo empezando desde 1
actividad_map = {nombre: i + 1 for i, nombre in enumerate(actividades)}

# Aplicar el mapeo
df['actividad_empresarial'] = df['actividad_empresarial'].map(actividad_map)


print(df.dtypes)
print(df.describe())

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

print(df_t.dtypes)


#valores_unicos = df_t['giro_comercio'].unique()
#print("Valores únicos :", valores_unicos)
#print(df_t['giro_comercio'].value_counts())



print("vamos a por toda en este hack")