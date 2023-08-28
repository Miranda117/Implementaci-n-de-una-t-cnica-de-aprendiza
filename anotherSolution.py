
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definir una función para graficar los datos y las predicciones
def plot(x, y, y_pred):
    plt.plot(x[:, 0], y, 'o', label="real")  # Graficar datos reales
    plt.plot(x[:, 0], y_pred, label='pred')   # Graficar predicciones
    plt.legend()  # Mostrar leyenda
    plt.show()    # Mostrar gráfico

# Definir una función para calcular el MSE
def L2(y, y_pred):
    n = len(y)
    return np.sum((y - y_pred) ** 2) / n

# Definir una función para calcular los gradientes del error con respecto a los parámetros
def gradients2(x, y, y_pred):
    n = len(y)
    Dm = (-2/n) * np.dot(x.T, (y - y_pred))   # Gradiente respecto a 'm'
    Dc = (-2/n) * np.sum(y - y_pred)          # Gradiente respecto a 'c'
    return Dm, Dc

# Definir los nombres de las columnas en el archivo CSV
columns = ["sepal length","sepal width","petal length","petal width", "class"]

# Leer el archivo CSV y almacenar los datos en un DataFrame
df = pd.read_csv('iris.data', names=columns)

# Mapear los valores de la columna 'class' a valores numéricos
mapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
df['classint'] = df['class'].replace(mapping)

# Extraer la columna 'classint' y convertirla en un vector 'y'
y = df["classint"].to_numpy()

# Eliminar las columnas 'class' y 'classint' para obtener la matriz 'x'
dfSamples = df.drop(["class", "classint"], axis=1)
x = dfSamples.to_numpy()

# Agregar una columna de unos al principio de la matriz 'x' (intercepto)
x = np.column_stack((np.ones(len(x)), x))

# Inicializar los parámetros 'm' como un vector de ceros
m = np.zeros(x.shape[1])

# Definir la tasa de aprendizaje (learning rate)
lr = 0.005

# Definir el número de pasos de optimización (iteraciones)
steps = 10000

y_pred = np.dot(x, m) #multiplicación punto

# Bucle de optimización usando descenso de gradiente
for i in range(steps):
    y_pred = np.dot(x, m)  # Calcular nuevas predicciones
    mse = L2(y, y_pred)    # Calcular el error cuadrático medio
    Dm, Dc = gradients2(x, y, y_pred)  # Calcular gradientes
    m -= Dm * lr  # Actualizar los parámetros 'm' utilizando los gradientes multiplicados por la tasa de aprendizaje

# Imprimir los valores finales de los parámetros 'm' y 'c', así como el error cuadrático medio 'mse'
print(f'm = {m[1:]}')  
print(f'c = {m[0]:.4f}')
print(f'mse = {mse:.4f}')

