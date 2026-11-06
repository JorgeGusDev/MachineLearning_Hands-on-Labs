# Hands-on-Labs#2: Regresión Logística con Iris Species

## Objetivo

Implementar un caso práctico de **regresión logística** para clasificar especies de flores Iris según sus características.

El dataset contiene información sobre medidas de sépalos y pétalos, junto con la especie correspondiente.

---

## 1. Importar librerías

Primero importamos las librerías necesarias para trabajar con datos, dividir el dataset, escalar variables, entrenar el modelo y evaluar resultados.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

---

## 2. Cargar el dataset

Cargamos el archivo CSV del dataset Iris desde Kaggle.

Dataset: https://www.kaggle.com/datasets/uciml/iris/data

```python
data = pd.read_csv("path.csv")
```

---

## 3. Seleccionar variables

Seleccionamos las variables de entrada `X` y la variable objetivo `y`.

En este dataset:

* `X`: medidas de la flor
* `y`: especie de la flor

```python
X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = data["Species"]
```

---

## 4. Dividir datos en entrenamiento y prueba

Dividimos el dataset en datos de entrenamiento y datos de prueba.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## 5. Escalar las variables

Escalamos las variables para que el modelo trabaje con valores en una escala similar.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 6. Crear y entrenar el modelo

Creamos el modelo de regresión logística y lo entrenamos con los datos de entrenamiento.

```python
model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)
```

---

## 7. Realizar predicciones

Usamos el modelo entrenado para predecir las especies en los datos de prueba.

```python
y_pred = model.predict(X_test_scaled)
```

---

## 8. Evaluar el modelo

Evaluamos el modelo usando cuatro métricas principales de clasificación: **Accuracy**, **Precision**, **Recall** y **F1-score**.

```python
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
```

---

## 9. Predecir una nueva flor

Creamos un nuevo ejemplo con las medidas de una flor y usamos el modelo para predecir su especie.

```python
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
)

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

print("Especie predicha:", prediction[0])
```

---

## Resultado esperado

El modelo debe clasificar las flores Iris en una de las siguientes especies:

* `Iris-setosa`
* `Iris-versicolor`
* `Iris-virginica`

Como el dataset Iris tiene clases bastante diferenciadas, se espera que el modelo obtenga buenos resultados en Accuracy, Precision, Recall y F1-score.


## Statement 
Utilizando el dataset [Heart Disease UCI](https://huggingface.co/datasets/bakhil-aissa/CL?utm_source=chatgpt.com) disponible en Hugging Face, desarrolla un modelo de Regresión Logística para predecir la presencia o ausencia de enfermedades cardíacas a partir de características clínicas como edad, colesterol, presión arterial y frecuencia cardíaca. Realiza la preparación de los datos, entrena y evalúa el modelo mediante Accuracy, Precision, Recall, F1-score y matriz de confusión. Finalmente, analiza los resultados y la capacidad predictiva del modelo.
