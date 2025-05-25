import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def cargar_datos():
    """Carga y prepara los datos"""
    transacciones = pd.read_csv('HeyBancoDatathonDAGA/datos/datadetrans.csv', delimiter=';')
    clientes = pd.read_csv('HeyBancoDatathonDAGA/datos/dataclientes.csv', delimiter=';')
    clientes = clientes.rename(columns={clientes.columns[0]: 'id_cliente'})
    return clientes, transacciones

def analizar_patrones_cliente(cliente_id, transacciones, umbral_frecuencia=8):
    """Analiza patrones de gasto recurrente del cliente con mejores métricas"""
    
    # Filtrar y preparar datos del cliente
    transacciones['fecha'] = pd.to_datetime(transacciones['fecha'])
    cliente_data = transacciones[transacciones['id'] == cliente_id].copy()
    
    if cliente_data.empty:
        return None, "Cliente no encontrado"
    
    # Análisis por giro comercial
    patrones = cliente_data.groupby(['giro_comercio', 'comercio']).agg({
        'fecha': ['count', 'min', 'max'],
        'monto': ['mean', 'std', 'median'],
        'dia_semana': lambda x: x.mode().iloc[0] if not x.mode().empty else x.mean(),
        'dia_mes': lambda x: x.mode().iloc[0] if not x.mode().empty else x.mean()
    }).reset_index()
    
    # Aplanar columnas
    patrones.columns = ['giro_comercio', 'comercio', 'frecuencia', 'fecha_min', 'fecha_max', 
                       'monto_promedio', 'monto_std', 'monto_mediana', 'dia_semana_comun', 'dia_mes_comun']
    
    # Filtrar patrones con suficiente historial
    patrones = patrones[patrones['frecuencia'] >= umbral_frecuencia].copy()
    
    if patrones.empty:
        return None, f"No se encontraron patrones recurrentes (mínimo {umbral_frecuencia} transacciones)"
    
    # Calcular métricas de regularidad
    patrones['dias_span'] = (patrones['fecha_max'] - patrones['fecha_min']).dt.days
    patrones['periodicidad_promedio'] = patrones['dias_span'] / (patrones['frecuencia'] - 1)
    patrones['regularidad'] = np.where(patrones['monto_std'] / patrones['monto_promedio'] < 0.3, 'Alta', 
                                     np.where(patrones['monto_std'] / patrones['monto_promedio'] < 0.6, 'Media', 'Baja'))
    
    # Calcular score de confianza
    patrones['score_confianza'] = (
        (patrones['frecuencia'] / patrones['frecuencia'].max()) * 0.4 +
        (1 - (patrones['monto_std'] / patrones['monto_promedio']).fillna(1)) * 0.3 +
        (patrones['dias_span'] / patrones['dias_span'].max()) * 0.3
    ).round(3)
    
    return patrones, cliente_data

def predecir_gastos_futuros(cliente_id, clientes, transacciones, dias_prediccion=90, usar_ml=True):
    """Predice gastos futuros con modelo mejorado"""
    
    try:
        # Analizar patrones
        patrones, cliente_data = analizar_patrones_cliente(cliente_id, transacciones)
        
        if patrones is None:
            return None, cliente_data  # cliente_data contiene el mensaje de error
        
        # Preparar predicciones
        predicciones = []
        hoy = datetime.now().date()
        fecha_fin = hoy + timedelta(days=dias_prediccion)
        
        # Preparar modelo ML si está habilitado
        if usar_ml and len(cliente_data) > 20:
            # Codificar variables categóricas
            le_giro = LabelEncoder()
            le_comercio = LabelEncoder()
            
            cliente_data_ml = cliente_data.copy()
            cliente_data_ml['giro_encoded'] = le_giro.fit_transform(cliente_data_ml['giro_comercio'])
            cliente_data_ml['comercio_encoded'] = le_comercio.fit_transform(cliente_data_ml['comercio'])
            
            # Features para ML
            X = cliente_data_ml[['dia_mes', 'dia_semana', 'mes', 'giro_encoded', 'comercio_encoded']]
            y = cliente_data_ml['monto']
            
            # Entrenar modelo
            modelo_ml = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            modelo_ml.fit(X, y)
        
        # Generar predicciones por patrón
        for _, patron in patrones.iterrows():
            # Calcular siguiente fecha esperada desde hoy
            ultima_fecha = patron['fecha_max']
            periodicidad = max(1, patron['periodicidad_promedio'])  # Evitar división por 0
            
            # Encontrar próxima fecha
            siguiente_fecha = ultima_fecha
            while siguiente_fecha.date() <= hoy:
                siguiente_fecha += timedelta(days=periodicidad)
            
            # Generar predicciones futuras
            contador = 0
            while siguiente_fecha.date() <= fecha_fin and contador < 10:  # Limitar iteraciones
                
                # Predecir monto
                if usar_ml and len(cliente_data) > 20:
                    try:
                        # Preparar datos para predicción ML
                        giro_encoded = le_giro.transform([patron['giro_comercio']])[0]
                        comercio_encoded = le_comercio.transform([patron['comercio']])[0]
                        
                        X_pred = pd.DataFrame({
                            'dia_mes': [siguiente_fecha.day],
                            'dia_semana': [siguiente_fecha.weekday()],
                            'mes': [siguiente_fecha.month],
                            'giro_encoded': [giro_encoded],
                            'comercio_encoded': [comercio_encoded]
                        })
                        
                        monto_predicho = modelo_ml.predict(X_pred)[0]
                    except:
                        monto_predicho = patron['monto_promedio']
                else:
                    # Usar promedio histórico con variación
                    variacion = np.random.normal(0, patron['monto_std'] if pd.notna(patron['monto_std']) else 0)
                    monto_predicho = patron['monto_promedio'] + variacion
                
                predicciones.append({
                    'comercio': patron['comercio'],
                    'giro_comercio': patron['giro_comercio'],
                    'fecha_predicha': siguiente_fecha.strftime('%Y-%m-%d'),
                    'dias_desde_hoy': (siguiente_fecha.date() - hoy).days,
                    'monto_esperado': round(max(0, monto_predicho), 2),
                    'probabilidad': round(patron['score_confianza'], 3),
                    'frecuencia_historica': int(patron['frecuencia']),
                    'periodicidad_dias': round(periodicidad, 1),
                    'regularidad': patron['regularidad'],
                    'dia_semana_comun': int(patron['dia_semana_comun']),
                    'dia_mes_comun': int(patron['dia_mes_comun'])
                })
                
                siguiente_fecha += timedelta(days=periodicidad)
                contador += 1
        
        if not predicciones:
            return None, "No se generaron predicciones futuras"
        
        # Convertir a DataFrame y ordenar
        df_predicciones = pd.DataFrame(predicciones)
        df_predicciones = df_predicciones.sort_values(['dias_desde_hoy', 'probabilidad'], 
                                                    ascending=[True, False])
        
        return df_predicciones, f"Predicciones generadas: {len(df_predicciones)} transacciones esperadas"
    
    except Exception as e:
        return None, f"Error en predicción: {str(e)}"

def generar_resumen_cliente(cliente_id, clientes, predicciones):
    """Genera resumen del análisis del cliente"""
    
    # Buscar info del cliente
    cliente_info = clientes[clientes['id_cliente'] == cliente_id]
    
    if not cliente_info.empty:
        cliente = cliente_info.iloc[0]
        info_texto = f"""
INFORMACIÓN DEL CLIENTE:
- ID: {cliente_id}
- Tipo: {'Persona Física' if cliente['tipo_persona'] == 1 else 'Persona Moral'}
- Actividad: {cliente['actividad_empresarial']}
- Edad: {cliente['edad']} años
- Antigüedad: {cliente['antiguedad']} años
        """
    else:
        info_texto = f"Cliente ID: {cliente_id}"
    
    if predicciones is not None:
        # Estadísticas de predicciones
        gasto_total = predicciones['monto_esperado'].sum()
        comercios_unicos = predicciones['comercio'].nunique()
        giros_unicos = predicciones['giro_comercio'].nunique()
        
        resumen_gastos = f"""
RESUMEN DE PREDICCIONES:
- Gasto total esperado: ${gasto_total:,.2f}
- Comercios diferentes: {comercios_unicos}
- Tipos de giro: {giros_unicos}
- Transacciones esperadas: {len(predicciones)}

TOP 3 GASTOS ESPERADOS:
"""
        top_gastos = predicciones.nlargest(3, 'monto_esperado')[['comercio', 'fecha_predicha', 'monto_esperado', 'probabilidad']]
        for _, gasto in top_gastos.iterrows():
            resumen_gastos += f"- {gasto['comercio']}: ${gasto['monto_esperado']:.2f} el {gasto['fecha_predicha']} (Prob: {gasto['probabilidad']:.2f})\n"
        
        return info_texto + resumen_gastos
    
    return info_texto

# EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    # Cargar datos
    print("Cargando datos...")
    clientes, transacciones = cargar_datos()
    
    # Configuración
    cliente_id = "8cf7b6a6cc74205009f63e37c438cc23181ee2c6"
    dias_prediccion = 60
    usar_ml = True
    
    print(f"\nAnalizando cliente: {cliente_id}")
    print(f"Período de predicción: {dias_prediccion} días")
    print("="*80)
    
    # Generar predicciones
    resultados, mensaje = predecir_gastos_futuros(cliente_id, clientes, transacciones, 
                                                dias_prediccion, usar_ml)
    
    if resultados is not None:
        # Mostrar resumen del cliente
        resumen = generar_resumen_cliente(cliente_id, clientes, resultados)
        print(resumen)
        
        # Mostrar predicciones detalladas
        print("\nPREDICCIONES DETALLADAS:")
        print("="*80)
        columnas_mostrar = ['comercio', 'fecha_predicha', 'dias_desde_hoy', 'monto_esperado', 
                          'probabilidad', 'frecuencia_historica', 'regularidad']
        print(resultados[columnas_mostrar].to_string(index=False))
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"predicciones_{cliente_id}_{datetime.now().date()}.csv"
        resultados.to_csv(f"HeyBancoDatathonDAGA/resultados/{nombre_archivo}", index=False)
        print(f"\nResultados guardados en: HeyBancoDatathonDAGA/resultados/{nombre_archivo}")
        
        # Estadísticas adicionales
        print(f"\nESTADÍSTICAS:")
        print(f"- Gasto promedio por transacción: ${resultados['monto_esperado'].mean():.2f}")
        print(f"- Gasto total esperado: ${resultados['monto_esperado'].sum():.2f}")
        print(f"- Confianza promedio: {resultados['probabilidad'].mean():.2f}")
        
    else:
        print(f" {mensaje}")
    
    print("\n" + "="*80)