# Implementaci-n-de-una-t-cnica-de-aprendiza
## Descripción del Código y Uso

Este codigo se insíra en la base proporcionada por el proporcionado por el profesor Benjamín Valdés Aguirre. 

El código implementa un modelo de regresión lineal utilizando el algoritmo de descenso de gradiente para ajustar una línea a un conjunto de datos. El objetivo es encontrar los parámetros de la línea (pendiente y ordenada al origen) que mejor se ajusten a los datos y minimicen el error cuadrático medio entre las predicciones del modelo y los valores reales.

El conjunto de datos utilizado se carga desde un archivo CSV llamado 'iris.data', que contiene información sobre características de diferentes tipos de flores iris. Se realiza una transformación en las etiquetas de clase para asignarles valores numéricos.

El código consta de los siguientes pasos:

1. Importar bibliotecas necesarias: Se importan las bibliotecas `math`, `numpy`, `pandas` y `matplotlib.pyplot` para el manejo de cálculos matemáticos, manipulación de datos y visualización de gráficos.

2. Definición de funciones:
   - `plot(x, y, y_pred)`: Esta función estaba destinada a graficar los datos reales y las predicciones del modelo, pero está comentada en esta versión del código.
   - `L2(y, y_pred)`: Calcula el error cuadrático medio (MSE) entre las predicciones del modelo y los valores reales.
   - `gradients2(x, y, y_pred)`: Calcula los gradientes del error con respecto a los parámetros de la línea (pendiente y ordenada al origen).

3. Lectura de datos:
   - Se definen los nombres de las columnas del archivo CSV.
   - Se carga el archivo CSV en un DataFrame de Pandas llamado `df`.
   - Se realiza una transformación de las etiquetas de clase a valores numéricos y se agrega una columna `classint` al DataFrame.

4. Preparación de datos:
   - Se extraen las etiquetas numéricas y se almacenan en un vector `y`.
   - Se eliminan las columnas de clase y clase numérica para obtener la matriz de características `x`.
   - Se agrega una columna de unos al principio de la matriz `x` para representar el término de intercepto.

5. Inicialización de parámetros y configuración del algoritmo:
   - Se inicializan los parámetros de la línea (`m`) como un vector de ceros.
   - Se define la tasa de aprendizaje (`lr`) que controla el tamaño de los pasos de actualización en el descenso de gradiente.
   - Se define el número de pasos de optimización (`steps`), que indica cuántas iteraciones del descenso de gradiente se realizarán.

6. Bucle de optimización:
   - Se inicia un bucle que itera el número de veces definido en `steps`.
   - Se calculan las predicciones del modelo utilizando la matriz `x` y los parámetros `m`.
   - Se calcula el error cuadrático medio (`mse`) entre las predicciones y los valores reales.
   - Se calculan los gradientes del error con respecto a los parámetros utilizando la función `gradients2`.
   - Se actualizan los parámetros `m` utilizando el descenso de gradiente.

7. Impresión de resultados:
   - Se imprimen los valores finales de los parámetros `m` y el término de intercepto.
   - Se imprime el valor final del error cuadrático medio (`mse`).

8. Comentario de función no utilizada:
   - La función `plot(x, y, y_pred)` está comentada ya que no se utiliza en esta versión del código.

Este código se centra en el ajuste de una línea a los datos utilizando regresión lineal mediante descenso de gradiente. Sin embargo, cabe destacar que existen bibliotecas y herramientas que simplifican la implementación de estos algoritmos y proporcionan funcionalidades adicionales para evaluar y visualizar los resultados de manera más eficiente.
