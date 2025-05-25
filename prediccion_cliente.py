'''import pandas as pd
import numpy as np

# Cargar los datos
transacciones = pd.read_csv('HeyBancoDatathonDAGA/datos/datadetrans.csv', delimiter=';')
clientes = pd.read_csv('HeyBancoDatathonDAGA/datos/dataclientes.csv', delimiter=';')

#conteo_ordenado = clientes['edad'].value_counts().sort_index()
#print(conteo_ordenado)


# Unir datos de clientes con transacciones (asumiendo que 'id' en transacciones corresponde al hash de cliente)
data = pd.merge(transacciones, clientes, left_on='id', right_on='id_municipio')

# Convertir fecha a datetime y extraer características
data['fecha'] = pd.to_datetime(data['fecha'])
data['dia_mes'] = data['fecha'].dt.day
data['dia_semana'] = data['fecha'].dt.dayofweek
data['semana_anio'] = data['fecha'].dt.isocalendar().week

print(data.dtypes)         # Tipos de datos de cada columna


# Preprocesamiento
numeric_features = ['monto', 'edad', 'antiguedad']
categorical_features = ['giro_comercio', 'tipo_venta', 'actividad_empresarial', 'genero']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])'''


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

def cargar_datos():
    """Carga y prepara los datos"""
    transacciones = pd.read_csv('HeyBancoDatathonDAGA/datos/datadetrans.csv', delimiter=';')
    clientes = pd.read_csv('HeyBancoDatathonDAGA/datos/dataclientes.csv', delimiter=';')

    clientes = clientes.rename(columns={clientes.columns[0]: 'id_cliente'})
    return clientes, transacciones

def predecir_futuro(cliente_id, clientes, transacciones, dias_futuro=90):
    """Predice gastos recurrentes FUTUROS para un cliente"""
    try:
        # Filtrar datos del cliente
        transacciones['fecha'] = pd.to_datetime(transacciones['fecha'])
        cliente_data = transacciones[transacciones['id'] == cliente_id].copy()
        ultima_fecha = cliente_data['fecha'].max()
        
        if cliente_data.empty:
            return None, "Cliente no encontrado"
        
        # Identificar patrones (simplificado)
        patrones = cliente_data.groupby(['comercio_id', 'giro_comercio']).agg(
            frecuencia=('fecha', 'count'),
            monto_promedio=('monto', 'mean'),
            ultima_transaccion=('fecha', 'max'),
            primera_transaccion=('fecha', 'min')
        ).reset_index()
        
        # Calcular periodicidad y filtrar recurrentes (mínimo 3 transacciones)
        patrones = patrones[patrones['frecuencia'] >= 3].copy()
        patrones['periodicidad_dias'] = (patrones['ultima_transaccion'] - patrones['primera_transaccion']).dt.days / (patrones['frecuencia'] - 1)
        
        # Generar predicciones FUTURAS (solo después de última fecha)
        predicciones = []
        hoy = datetime.now().date()
        
        for _, patron in patrones.iterrows():
            # Calcular próximas fechas posibles (solo futuras)
            siguiente_fecha = patron['ultima_transaccion'] + timedelta(days=patron['periodicidad_dias'])
            
            while siguiente_fecha.date() <= (ultima_fecha.date() + timedelta(days=dias_futuro)):
                if siguiente_fecha.date() > ultima_fecha.date():  # Solo futuro
                    # Modelo predictivo simplificado
                    X = cliente_data[['dia_mes', 'dia_semana', 'mes']]
                    y = cliente_data['monto']
                    model = RandomForestRegressor(n_estimators=30, random_state=42)
                    model.fit(X, y)
                    
                    X_pred = pd.DataFrame({
                        'dia_mes': [siguiente_fecha.day],
                        'dia_semana': [siguiente_fecha.weekday()],
                        'mes': [siguiente_fecha.month]
                    }, index=[0])
                    
                    predicciones.append({
                        'comercio': patron['giro_comercio'],
                        'fecha_predicha': siguiente_fecha.strftime('%Y-%m-%d'),
                        'dias_desde_ultima': (siguiente_fecha.date() - ultima_fecha.date()).days,
                        'monto_esperado': round(model.predict(X_pred)[0], 2),
                        'probabilidad': min(0.95, patron['frecuencia'] / 10),
                        'periodicidad_dias': round(patron['periodicidad_dias'], 1),
                        'ultimas_transacciones': patron['frecuencia']
                    })
                
                siguiente_fecha += timedelta(days=patron['periodicidad_dias'])
        
        if not predicciones:
            return None, "No se encontraron patrones recurrentes futuros"
            
        # Convertir a DataFrame y ordenar
        df_pred = pd.DataFrame(predicciones)
        df_pred = df_pred.sort_values(['probabilidad', 'dias_desde_ultima'], ascending=[False, True])
        
        return df_pred, "Predicciones futuras generadas"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

# EJECUCIÓN
if __name__ == "__main__":
    clientes, transacciones = cargar_datos()
    
    # Configuración
    cliente_id = "91477f382c3cf63ab5cd9263b502109243741158"  # Reemplaza con tu ID
    dias_prediccion = 180  # Predicción para los próximos X días
    
    # Obtener resultados
    resultados, mensaje = predecir_futuro(cliente_id, clientes, transacciones, dias_prediccion)
    
    # Mostrar
    print("\n" + "="*60)
    print(f"PREDICCIONES FUTURAS - Cliente: {cliente_id}")
    print(f"Última transacción registrada: {transacciones[transacciones['id'] == cliente_id]['fecha'].max().date()}")
    print(f"Rango de predicción: Próximos {dias_prediccion} días")
    print("="*60)
    
    if resultados is not None:
        print("\nGastos recurrentes esperados (ordenados por probabilidad):")
        print(resultados.to_string(index=False))
    else:
        print(f"\n{mensaje}")
    
    print("\n" + "="*60)
