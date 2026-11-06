# Regresión Lineal con Gradient Descent

## Objetivo

Implementar un caso práctico de **regresión lineal** usando **Gradient Descent** para predecir el salario de una persona según sus años de experiencia.

Dataset: https://www.kaggle.com/datasets/mohithsairamreddy/salary-data/data 


El dataset utilizado contiene dos columnas a utilizar:

* `Years of Experience`
* `Salary`

---

## 1. Importar librerías

Primero importamos las librerías necesarias para trabajar con datos, cálculos numéricos y gráficos.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

---

## 2. Cargar el dataset

Cargamos el archivo CSV que contiene los datos.

```python
data = pd.read_csv("path.csv")
```

---

## 3. Seleccionar variables

Seleccionamos la variable de entrada `X` y la variable objetivo `y`.

* `X`: años de experiencia
* `y`: salario

```python
X = data["Years of Experience"].values
y = data["Salary"].values
```

---

## 4. Visualizar los datos

Graficamos los datos para observar la relación entre años de experiencia y salario.

```python
plt.scatter(X, y)
plt.xlabel("Años de experiencia")
plt.ylabel("Salario")
plt.title("Relación entre experiencia y salario")
plt.show()
```

---

## 5. Normalizar la variable de entrada

Normalizamos `X` para que Gradient Descent funcione de manera más estable.

```python
X_mean = X.mean()
X_std = X.std()

X_scaled = (X - X_mean) / X_std
```

---

## 6. Inicializar parámetros

La regresión lineal usa la siguiente fórmula:

```text
y_pred = w * X + b
```

Donde:

* `y_pred`: salario predicho
* `X`: años de experiencia
* `w`: peso o pendiente
* `b`: sesgo o intercepto

En Python, inicializamos los parámetros en cero:

```python
w = 0
b = 0
```

---

## 7. Definir hiperparámetros

Definimos el tamaño del paso de aprendizaje, el número de épocas y la cantidad de datos.

```python
learning_rate = 0.01
epochs = 1000
n = len(X_scaled)

loss_history = []
```

---

## 8. Entrenar el modelo con Gradient Descent

En cada época, el modelo calcula una predicción, mide el error y actualiza los parámetros `w` y `b`.

```python
for epoch in range(epochs):
    # Predicción
    y_pred = w * X_scaled + b

    # Error
    error = y_pred - y

    # Mean Squared Error
    mse = np.mean(error ** 2)
    loss_history.append(mse)

    # Gradientes
    dw = (2 / n) * np.sum(error * X_scaled)
    db = (2 / n) * np.sum(error)

    # Actualización de parámetros
    w = w - learning_rate * dw
    b = b - learning_rate * db
```

---

## 9. Mostrar resultados finales

Mostramos los valores finales aprendidos por el modelo.

```python
print("Peso final:", w)
print("Sesgo final:", b)
print("Error final:", loss_history[-1])
```

---

## 10. Graficar la línea de regresión

Graficamos los datos reales junto con la línea aprendida por el modelo.

```python
plt.scatter(X, y, label="Datos reales")

X_line = np.linspace(X.min(), X.max(), 100)
X_line_scaled = (X_line - X_mean) / X_std
y_line = w * X_line_scaled + b

plt.plot(X_line, y_line, label="Línea de regresión")

plt.xlabel("Años de experiencia")
plt.ylabel("Salario")
plt.title("Regresión Lineal con Gradient Descent")
plt.legend()
plt.show()
```

---

## 11. Graficar la disminución del error

Este gráfico muestra si el error disminuye durante el entrenamiento.

```python
plt.plot(loss_history)
plt.xlabel("Épocas")
plt.ylabel("MSE")
plt.title("Disminución del error")
plt.show()
```

---

## 12. Realizar una predicción

Usamos el modelo entrenado para predecir el salario de una persona con 5 años de experiencia.

```python
years = 5

years_scaled = (years - X_mean) / X_std
salary_pred = w * years_scaled + b

print(f"Salario predicho para {years} años de experiencia: {salary_pred:.2f}")
```

---

## Resultado esperado

El modelo debe aprender una relación positiva entre años de experiencia y salario.

Si Gradient Descent funciona correctamente, el error debería disminuir progresivamente durante el entrenamiento.

---

## Nota importante

En este caso práctico no se incluye exploración ni limpieza de datos.

Si aparece un resultado como:

```text
Peso final: nan
Sesgo final: nan
Error final: nan
```
aplicar
```python
data = data.dropna(subset=["column1", "column2"])
```
significa que el dataset puede tener valores nulos, columnas con nombres diferentes o datos no numéricos. Esa revisión debe realizarse antes de entrenar el modelo.

---

