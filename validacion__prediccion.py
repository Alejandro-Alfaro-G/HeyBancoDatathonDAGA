import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def cargar_datos():
    """Carga datos del sistema"""
    transacciones = pd.read_csv('HeyBancoDatathonDAGA/datos/datadetrans.csv', delimiter=';')
    clientes = pd.read_csv('HeyBancoDatathonDAGA/datos/dataclientes.csv', delimiter=';')
    clientes = clientes.rename(columns={clientes.columns[0]: 'id_cliente'})
    return clientes, transacciones

def dividir_datos_temporalmente(transacciones, fecha_corte=None, dias_entrenamiento=180, dias_test=60):
    """
    Divide los datos en entrenamiento y test de forma temporal
    """
    transacciones['fecha'] = pd.to_datetime(transacciones['fecha'])
    
    if fecha_corte is None:
        # Usar una fecha que permita tener datos de test
        fecha_max = transacciones['fecha'].max()
        fecha_corte = fecha_max - timedelta(days=dias_test)
    
    fecha_inicio_entrenamiento = fecha_corte - timedelta(days=dias_entrenamiento)
    
    # Dividir datos
    datos_entrenamiento = transacciones[
        (transacciones['fecha'] >= fecha_inicio_entrenamiento) & 
        (transacciones['fecha'] <= fecha_corte)
    ].copy()
    
    datos_test = transacciones[
        (transacciones['fecha'] > fecha_corte) & 
        (transacciones['fecha'] <= fecha_corte + timedelta(days=dias_test))
    ].copy()
    
    return datos_entrenamiento, datos_test, fecha_corte

def generar_predicciones_validacion(cliente_id, datos_entrenamiento, fecha_corte, dias_prediccion=60):
    """
    Genera predicciones usando solo datos de entrenamiento
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    
    # Filtrar datos del cliente
    cliente_data = datos_entrenamiento[datos_entrenamiento['id'] == cliente_id].copy()
    
    if cliente_data.empty:
        return None
    
    # Identificar patrones recurrentes
    patrones = cliente_data.groupby(['giro_comercio', 'comercio']).agg({
        'fecha': ['count', 'min', 'max'],
        'monto': ['mean', 'std']
    }).reset_index()
    
    # Aplanar columnas
    patrones.columns = ['giro_comercio', 'comercio', 'frecuencia', 'fecha_min', 'fecha_max', 'monto_promedio', 'monto_std']
    
    # Filtrar patrones válidos
    patrones = patrones[patrones['frecuencia'] >= 5].copy()
    
    if patrones.empty:
        return None
    
    # Calcular periodicidad
    patrones['dias_span'] = (patrones['fecha_max'] - patrones['fecha_min']).dt.days
    patrones['periodicidad_promedio'] = patrones['dias_span'] / (patrones['frecuencia'] - 1)
    
    # Generar predicciones
    predicciones = []
    fecha_inicio = fecha_corte
    fecha_fin = fecha_corte + timedelta(days=dias_prediccion)
    
    for _, patron in patrones.iterrows():
        # Calcular siguiente fecha esperada
        ultima_fecha = patron['fecha_max']
        periodicidad = max(1, patron['periodicidad_promedio'])
        
        siguiente_fecha = ultima_fecha
        while siguiente_fecha <= fecha_inicio:
            siguiente_fecha += timedelta(days=periodicidad)
        
        # Generar predicciones para el período
        contador = 0
        while siguiente_fecha <= fecha_fin and contador < 10:
            predicciones.append({
                'cliente_id': cliente_id,
                'comercio': patron['comercio'],
                'giro_comercio': patron['giro_comercio'],
                'fecha_predicha': siguiente_fecha,
                'monto_esperado': patron['monto_promedio'],
                'frecuencia_historica': patron['frecuencia'],
                'periodicidad_dias': periodicidad
            })
            
            siguiente_fecha += timedelta(days=periodicidad)
            contador += 1
    
    return pd.DataFrame(predicciones) if predicciones else None

def calcular_metricas_precision(predicciones, datos_reales, tolerancia_dias=3):
    """
    Calcula métricas de precisión comparando predicciones vs datos reales
    """
    if predicciones is None or predicciones.empty:
        return None
    
    # Preparar datos reales para comparación
    datos_reales['fecha'] = pd.to_datetime(datos_reales['fecha'])
    
    # Métricas por cliente
    resultados_metricas = []
    
    for cliente_id in predicciones['cliente_id'].unique():
        pred_cliente = predicciones[predicciones['cliente_id'] == cliente_id]
        real_cliente = datos_reales[datos_reales['id'] == cliente_id]
        
        if real_cliente.empty:
            continue
        
        # Métricas de coincidencia
        coincidencias_exactas = 0
        coincidencias_tolerancia = 0
        errores_monto = []
        errores_fecha = []
        
        for _, pred in pred_cliente.iterrows():
            # Buscar transacciones reales similares
            transacciones_similares = real_cliente[
                (real_cliente['comercio'] == pred['comercio']) |
                (real_cliente['giro_comercio'] == pred['giro_comercio'])
            ]
            
            if not transacciones_similares.empty:
                # Encontrar la transacción más cercana en fecha
                diferencias_fecha = abs((transacciones_similares['fecha'] - pred['fecha_predicha']).dt.days)
                idx_mas_cercana = diferencias_fecha.idxmin()
                transaccion_cercana = transacciones_similares.loc[idx_mas_cercana]
                
                dias_diferencia = abs((transaccion_cercana['fecha'] - pred['fecha_predicha']).days)
                
                # Contar coincidencias
                if dias_diferencia == 0:
                    coincidencias_exactas += 1
                if dias_diferencia <= tolerancia_dias:
                    coincidencias_tolerancia += 1
                
                # Calcular errores
                errores_monto.append(abs(transaccion_cercana['monto'] - pred['monto_esperado']))
                errores_fecha.append(dias_diferencia)
        
        # Calcular métricas del cliente
        total_predicciones = len(pred_cliente)
        total_reales = len(real_cliente)
        
        if total_predicciones > 0:
            precision_exacta = coincidencias_exactas / total_predicciones
            precision_tolerancia = coincidencias_tolerancia / total_predicciones
            recall = coincidencias_tolerancia / total_reales if total_reales > 0 else 0
            
            mae_monto = np.mean(errores_monto) if errores_monto else float('inf')
            mae_fecha = np.mean(errores_fecha) if errores_fecha else float('inf')
            
            resultados_metricas.append({
                'cliente_id': cliente_id,
                'precision_exacta': precision_exacta,
                'precision_tolerancia': precision_tolerancia,
                'recall': recall,
                'f1_score': 2 * (precision_tolerancia * recall) / (precision_tolerancia + recall) if (precision_tolerancia + recall) > 0 else 0,
                'mae_monto': mae_monto,
                'mae_fecha_dias': mae_fecha,
                'total_predicciones': total_predicciones,
                'total_reales': total_reales,
                'coincidencias_tolerancia': coincidencias_tolerancia
            })
    
    return pd.DataFrame(resultados_metricas)

def evaluar_sistema_completo(clientes_muestra=None, dias_entrenamiento=180, dias_test=60):
    """
    Evalúa el sistema completo con múltiples clientes
    """
    print("Cargando datos...")
    clientes, transacciones = cargar_datos()
    
    # Seleccionar muestra de clientes
    if clientes_muestra is None:
        # Obtener clientes con suficientes transacciones
        conteo_transacciones = transacciones['id'].value_counts()
        clientes_activos = conteo_transacciones[conteo_transacciones >= 20].head(10).index.tolist()
        clientes_muestra = clientes_activos
    
    print(f"Evaluando {len(clientes_muestra)} clientes...")
    
    # Dividir datos temporalmente
    datos_entrenamiento, datos_test, fecha_corte = dividir_datos_temporalmente(
        transacciones, dias_entrenamiento=dias_entrenamiento, dias_test=dias_test
    )
    
    print(f"Fecha de corte: {fecha_corte.date()}")
    print(f"Datos entrenamiento: {len(datos_entrenamiento)} transacciones")
    print(f"Datos test: {len(datos_test)} transacciones")
    
    # Evaluar cada cliente
    todas_las_metricas = []
    
    for i, cliente_id in enumerate(clientes_muestra):
        print(f"Evaluando cliente {i+1}/{len(clientes_muestra)}: {cliente_id[:8]}...")
        
        # Generar predicciones
        predicciones = generar_predicciones_validacion(cliente_id, datos_entrenamiento, fecha_corte, dias_test)
        
        if predicciones is not None:
            # Calcular métricas
            metricas_cliente = calcular_metricas_precision(predicciones, datos_test)
            if metricas_cliente is not None and not metricas_cliente.empty:
                todas_las_metricas.append(metricas_cliente)
    
    if not todas_las_metricas:
        print(" No se pudieron calcular métricas para ningún cliente")
        return None
    
    # Consolidar resultados
    df_metricas = pd.concat(todas_las_metricas, ignore_index=True)
    
    return df_metricas, datos_entrenamiento, datos_test, fecha_corte

def generar_reporte_metricas(df_metricas):
    """
    Genera reporte completo de métricas del sistema
    """
    if df_metricas is None or df_metricas.empty:
        return "No hay métricas para reportar"
    
    # Estadísticas generales
    print("\n" + "="*80)
    print("REPORTE DE MÉTRICAS DEL SISTEMA DE PREDICCIÓN")
    print("="*80)
    
    # Métricas promedio
    precision_exacta_avg = df_metricas['precision_exacta'].mean()
    precision_tolerancia_avg = df_metricas['precision_tolerancia'].mean()
    recall_avg = df_metricas['recall'].mean()
    f1_avg = df_metricas['f1_score'].mean()
    mae_monto_avg = df_metricas['mae_monto'].mean()
    mae_fecha_avg = df_metricas['mae_fecha_dias'].mean()
    
    print(f"\n MÉTRICAS DE PRECISIÓN:")
    print(f"   • Precisión Exacta (fecha exacta): {precision_exacta_avg:.2%}")
    print(f"   • Precisión con Tolerancia (±3 días): {precision_tolerancia_avg:.2%}")
    print(f"   • Recall (cobertura): {recall_avg:.2%}")
    print(f"   • F1-Score: {f1_avg:.3f}")
    
    print(f"\n ERRORES PROMEDIO:")
    print(f"   • Error Absoluto en Monto: ${mae_monto_avg:.2f}")
    print(f"   • Error Absoluto en Fecha: {mae_fecha_avg:.1f} días")
    
    print(f"\n ESTADÍSTICAS GENERALES:")
    print(f"   • Clientes evaluados: {len(df_metricas)}")
    print(f"   • Predicciones totales: {df_metricas['total_predicciones'].sum()}")
    print(f"   • Transacciones reales: {df_metricas['total_reales'].sum()}")
    print(f"   • Coincidencias encontradas: {df_metricas['coincidencias_tolerancia'].sum()}")
    
    # Top y Bottom performers
    print(f"\n MEJORES CLIENTES (por F1-Score):")
    top_clientes = df_metricas.nlargest(3, 'f1_score')
    for _, cliente in top_clientes.iterrows():
        cliente_id_short = cliente['cliente_id'][:8] + "..."
        print(f"   • {cliente_id_short}: F1={cliente['f1_score']:.3f}, Precisión={cliente['precision_tolerancia']:.2%}")
    
    print(f"\n CLIENTES CON DESAFÍOS:")
    bottom_clientes = df_metricas.nsmallest(3, 'f1_score')
    for _, cliente in bottom_clientes.iterrows():
        cliente_id_short = cliente['cliente_id'][:8] + "..."
        print(f"   • {cliente_id_short}: F1={cliente['f1_score']:.3f}, Precisión={cliente['precision_tolerancia']:.2%}")
    
    # Distribución de errores
    print(f"\nDISTRIBUCIÓN DE ERRORES:")
    print(f"   • Monto - Mediana: ${df_metricas['mae_monto'].median():.2f}")
    print(f"   • Monto - P90: ${df_metricas['mae_monto'].quantile(0.9):.2f}")
    print(f"   • Fecha - Mediana: {df_metricas['mae_fecha_dias'].median():.1f} días")
    print(f"   • Fecha - P90: {df_metricas['mae_fecha_dias'].quantile(0.9):.1f} días")
    
    # Clasificación del rendimiento
    print(f"\nCLASIFICACIÓN DE RENDIMIENTO:")
    excelente = (df_metricas['f1_score'] >= 0.7).sum()
    bueno = ((df_metricas['f1_score'] >= 0.5) & (df_metricas['f1_score'] < 0.7)).sum()
    regular = ((df_metricas['f1_score'] >= 0.3) & (df_metricas['f1_score'] < 0.5)).sum()
    bajo = (df_metricas['f1_score'] < 0.3).sum()
    
    total = len(df_metricas)
    print(f"Excelente (F1 >= 0.7): {excelente} clientes ({excelente/total:.1%})")
    print(f"Bueno (0.5 <= F1 < 0.7): {bueno} clientes ({bueno/total:.1%})")
    print(f"Regular (0.3 <= F1 < 0.5): {regular} clientes ({regular/total:.1%})")
    print(f"Bajo (F1 < 0.3): {bajo} clientes ({bajo/total:.1%})")
    
    print("\n" + "="*80)
    
    return df_metricas

# EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    print("INICIANDO EVALUACIÓN DEL SISTEMA DE PREDICCIÓN")
    print("="*60)
    
    # Evaluar sistema completo
    resultado = evaluar_sistema_completo(
        clientes_muestra=None,  # None = selección automática
        dias_entrenamiento=120,
        dias_test=30
    )
    
    if resultado is not None:
        df_metricas, datos_entrenamiento, datos_test, fecha_corte = resultado
        
        # Generar reporte
        reporte = generar_reporte_metricas(df_metricas)
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_metricas.to_csv(f"HeyBancoDatathonDAGA/resultados/metricas_evaluacion_{timestamp}.csv", index=False)
        print(f"\nMétricas guardadas en: HeyBancoDatathonDAGA/resultados/metricas_evaluacion_{timestamp}.csv")
        
    else:
        print("No se pudieron calcular las métricas del sistema")
    
    print("\nEVALUACIÓN COMPLETADA")