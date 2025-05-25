import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

def cargar_datos():
    """Carga y prepara los datos"""
    transacciones = pd.read_csv('HeyBancoDatathonDAGA/datos/datadetrans.csv', delimiter=';')
    clientes = pd.read_csv('HeyBancoDatathonDAGA/datos/dataclientes.csv', delimiter=';')
    clientes = clientes.rename(columns={clientes.columns[0]: 'id_cliente'})
    return clientes, transacciones

def predecir_desde_hoy(cliente_id, clientes, transacciones, dias_prediccion=90):
    """Predice gastos recurrentes FUTUROS comenzando desde HOY"""
    try:
        # Filtrar datos del cliente
        transacciones['fecha'] = pd.to_datetime(transacciones['fecha'])
        cliente_data = transacciones[transacciones['id'] == cliente_id].copy()
        
        if cliente_data.empty:
            return None, "Cliente no encontrado"
        
        # 1. Identificar patrones recurrentes
        patrones = cliente_data.groupby(['comercio_id', 'comercio', 'giro_comercio']).agg(
            frecuencia=('fecha', 'count'),
            monto_promedio=('monto', 'mean'),
            ultima_transaccion=('fecha', 'max'),
            primera_transaccion=('fecha', 'min')
        ).reset_index()
        
        # Filtrar patrones con suficiente historial (≥3 transacciones)
        patrones = patrones[patrones['frecuencia'] >= 3].copy()
        
        # Calcular periodicidad en días
        patrones['periodicidad_dias'] = (patrones['ultima_transaccion'] - patrones['primera_transaccion']).dt.days / (patrones['frecuencia'] - 1)
        
        # 2. Generar predicciones desde HOY
        predicciones = []
        hoy = datetime.now().date()
        fecha_fin = hoy + timedelta(days=dias_prediccion)
        
        for _, patron in patrones.iterrows():
            # Calcular primera ocurrencia después de HOY
            siguiente_fecha = patron['ultima_transaccion']
            while siguiente_fecha.date() < hoy:
                siguiente_fecha += timedelta(days=patron['periodicidad_dias'])
            
            # Generar todas las ocurrencias dentro del período de predicción
            while siguiente_fecha.date() <= fecha_fin:
                # Preparar datos para el modelo
                cliente_data['dia_mes'] = cliente_data['fecha'].dt.day
                cliente_data['dia_semana'] = cliente_data['fecha'].dt.dayofweek
                cliente_data['mes'] = cliente_data['fecha'].dt.month
                
                X = cliente_data[['dia_mes', 'dia_semana', 'mes']]
                y = cliente_data['monto']
                
                # Entrenar modelo (simplificado para el ejemplo)
                model = RandomForestRegressor(n_estimators=30, random_state=42)
                model.fit(X, y)
                
                X_pred = pd.DataFrame({
                    'dia_mes': [siguiente_fecha.day],
                    'dia_semana': [siguiente_fecha.weekday()],
                    'mes': [siguiente_fecha.month]
                }, index=[0])
                
                predicciones.append({
                    'comercio_nombre': patron['comercio'],  # Nombre específico (ej: "AMAZON")
                    'comercio_giro': patron['giro_comercio'],  # Categoría (ej: "COMERCIO ELECTRÓNICO")
                    'fecha_predicha': siguiente_fecha.strftime('%Y-%m-%d'),
                    'dias_desde_hoy': (siguiente_fecha.date() - hoy).days,
                    'monto_esperado': round(model.predict(X_pred)[0], 2),
                    'probabilidad': min(0.95, patron['frecuencia'] / 10),
                    'frecuencia_dias': round(patron['periodicidad_dias'], 1),
                    'ultimas_transacciones': patron['frecuencia']
                })
                
                siguiente_fecha += timedelta(days=patron['periodicidad_dias'])
        
        if not predicciones:
            return None, "No se encontraron gastos recurrentes proyectados"
            
        # Convertir a DataFrame y ordenar
        df_pred = pd.DataFrame(predicciones)
        df_pred = df_pred.sort_values(['fecha_predicha', 'probabilidad'], ascending=[True, False])
        
        return df_pred, "Predicciones generadas desde hoy"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

# EJECUCIÓN
if __name__ == "__main__":
    clientes, transacciones = cargar_datos()
    
    # Configuración
    cliente_id = "91477f382c3cf63ab5cd9263b502109243741158"  # Reemplaza con tu ID
    dias_prediccion = 30  # Predicción para los próximos X días
    
    # Obtener resultados
    resultados, mensaje = predecir_desde_hoy(cliente_id, clientes, transacciones, dias_prediccion)
    
    # Mostrar
    print("\n" + "="*70)
    print(f"PREDICCIONES DESDE HOY - Cliente: {cliente_id}")
    print(f"Fecha actual: {datetime.now().date()}")
    print(f"Rango de predicción: {dias_prediccion} días hacia adelante")
    print("="*70)
    
    if resultados is not None:
        print("\nGastos recurrentes esperados (desde hoy):")
        print(resultados.to_string(index=False))
        
        # Guardar resultados
        nombre_archivo = f"predicciones_{cliente_id}_{datetime.now().date()}.csv"
        resultados.to_csv("HeyBancoDatathonDAGA/resultados/" + nombre_archivo, index=False)
        print(f"\nResultados guardados en: HeyBancoDatathonDAGA/resultados/{nombre_archivo}")
    else:
        print(f"\n{mensaje}")
    
    print("\n" + "="*70)