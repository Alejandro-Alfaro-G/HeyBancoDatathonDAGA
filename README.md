# HeyBancoDatathonDAGA
Reto del datathon edición 2025 en el reto proporcionado por HeyBanco

Se busca, a partir de bases de datos de compras y datos de clientes, llegar a predecir futuras compras. Este modelo llega a predecir hasta un tiempo de 30 días después. 

# Información de la programación:

Se crean nuevos archivos con distintos usos para poder simplificar el reto. Los archivos originales dados por la organización son los archivos de "base_clientes_final.csv", "base_transacciones_final.csv". 


El archivo actividades_empresariales.csv  sirve para cambiar las actividades empresariales de una categoría a un número. Cada valor del número es indicado en el archivo csv.  

El archivo comercios_codificados.csv sirve para númerar las categorías de los negocios que se nos han dado y asignarles un número. En esta entrega, son un total de 97 distintos comercios.

Asimismo, se crean los archivos de dataclientes.csv y datadetrans.csv consideradas como archivos "limpios", el cual se tiene la información lista y preparada. A partir de esto se llega a entrenar en los modelos en estos últimos dos archivos. 

# Explicación de Variables

En cualquier variable, si no se tiene un dato (nulo), se considera el dato "0". En algunos datos de categoría, se utilizan los valores de 1,2,3,..

Esto debido a que nuestro modelo usa regresiones considerando datos categóricos al igual que continuos. Los únicos datos continuos usados en el modelo, son las fechas (antiguedad y edad de alta en el banco) al igual que la edad del cliente.

# Resultados del modelo

El modelo trabaja en 3 partes. La primera parte consiste en predecir el día del próximo gasto, el segundo en predecir el tipo de comercio (y giro) y utilizando regresiones se calcula en la tercera parte la predicción del monto. A partir de esto, el modelo llega a entregar estas predicciones en tabla csv de un solo usuario. Este incluye varios gastos, y predice el comercio (con el giro de comercio), la fecha predicha, y el monto esperado. Todo esto estará en un formato csv despliegable. 

# Limitaciones y Mejoras

Existen algunas limitaciones en nuestro modelo. Por ejemplo, el código no evalúa la precisión del modelo. Aunque se puedan generar valores de MAPE y R^2. También, el modelo es independiente por cada fecha y comercio, por lo que su eficiencia es débil y se ocuparía mejorar antes de ser escalable. Existe un riesgo de Overfitting, dado que el código es entrenado únicamente con datos del pasado y no tiene una capacidad de predecir nuevas tendencias de compras en un cliente. 

También se debe recalcar que no hay un identificador confiable que distinga dos compras en el mismo día por la misma persona en distintos lugares. 
El Identificador IDT propuesto sigue la siguiente estructura:
"idcliente-####"
Donde el #### es el número de compra por el cliente. Es decir inica en 0001 y llegaría a un máximo de 9999 compras (pensada para compras en un año)

Algunas mejoras incluyen superar el efecto de las limitaciones previamente mencionadas. Por ejemplo, se buscaría entrenar un solo modelo global para todos los clientes y uno específico para un solo cliente, para poder utilizar una combinación de ambos para tanto detectar los gastos que se forman continuamente al igual que nuevas tendencias; que pueden ser predecidas utilizando los demás usuarios. Asimismo, se consideraría buscar una forma de incluir una evaluación del modelo con datos reales que se tienen. 

# Conclusiones

El modelo efectivamente retorna una lista de compras esperadas en los futuros 30 días, prediciendo las fechas, el tipo de gasto (y giro) al igual que el monto esperado. Este modelo aplica únicamente para un usuario a la vez y en un desarrollo front-end se puede tener un buscador para visualizar los futuros gastos de cada comprador. También, se requiere de más datos - los cuales el modelo no tenga acceso - para verificar la precisión del modelo.