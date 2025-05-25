# HeyBancoDatathonDAGA
Reto del datathon edición 2025 en el reto de HeyBanco

Dependiendo del giro, se tiene una diferente tasa de interés. 

# Información de la programación:

Se crean nuevos archivos con distintos usos para poder simplificar el reto. Los archivos originales dados por la organización son los archivos de "base_clientes_final.csv", "base_transacciones_final.csv". 


El archivo actividades_empresariales.csv  sirve para cambiar las actividades empresariales de una categoría a un número. Cada valor del número es indicado en el archivo csv.  

El archivo comercios_codificados.csv sirve para númera las categorías de los negocios que se nos han dado y asignarles un número. En esta entrega, son un total de 97 distintos comercios.



# Explicación de Variables

En cualquier variable, si no se tiene un dato, se considera el dato "0". En algunos datos de categoría, se utilizan los valores de 1,2.

El tipo_venta sirve para indicar si la transacción es digital o física.



# Problemas en el reto:

No hay un identificador confiable que distinga dos compras en el mismo día por la misma persona en distintos lugares.  

Esto se resuelve creando un identificador siguiendo l asiguiente estructura:
El Identificador IDT propuesto sigue la siguiente estructura:
"idcliente-####"
Donde el #### es el número de compra por el cliente. Es decir inica en 0001 y llegaría a un máximo de 9999 compras (pensada para compras en un año)