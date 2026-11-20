# Caso Práctico usando Python # 2

## Preprocesamiento de los Datos

El preprocesamiento de datos es una etapa crítica en proyectos de Inteligencia Artificial y Machine Learning, ya que permite transformar los datos crudos en un formato adecuado para el modelado.

Las principales fases incluyen:

* Limpieza de datos
* Tratamiento de valores nulos
* Transformación de variables
* Reducción de ruido e inconsistencias
* Escalado y normalización

En este caso práctico se utilizará el dataset **Melbourne Housing Snapshot** de Kaggle.
Dataset: https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot


---

## 1. Importar librerías

Primero importamos las librerías necesarias.

```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
```

---

## 2. Cargar el dataset

Cargamos el archivo CSV desde Kaggle.


```python
df = pd.read_csv("path")
```

Visualizamos las primeras filas.

```python
df.head()
```

---

## 3. Revisar valores nulos

Antes de tratar los valores faltantes, revisamos cuántos valores nulos existen en cada columna.

```python
df.isnull().sum()
```

También podemos ordenar las columnas con más valores nulos.

```python
df.isnull().sum().sort_values(ascending=False)
```

---

# Tratamiento de Datos Nulos

Los valores nulos, también llamados *missing values*, pueden afectar negativamente el rendimiento de los modelos, por lo que deben ser tratados adecuadamente.

---

## 4. Eliminación de datos nulos

Se eliminan filas que contienen valores faltantes.

```python
df = df.dropna()
```

También se puede eliminar valores nulos solo en columnas específicas.

```python
df = df.dropna(subset=["Price"])
```

**Uso:** Cuando la cantidad de datos nulos es pequeña.

---

## 5. Reemplazo con media, mediana o moda

Otra forma de tratar valores nulos es reemplazarlos por valores representativos.

---

### Media

La media puede usarse en variables numéricas cuando los datos son relativamente simétricos.

```python
df["Car"] = df["Car"].fillna(df["Car"].mean())
```

---

### Mediana

La mediana es útil cuando existen valores extremos u outliers.

```python
df["BuildingArea"] = df["BuildingArea"].fillna(df["BuildingArea"].median())
```

---

### Moda

La moda se usa normalmente en variables categóricas, porque representa el valor más frecuente.

```python
df["CouncilArea"] = df["CouncilArea"].fillna(df["CouncilArea"].mode()[0])
```

**Uso:**

* Media → datos numéricos simétricos
* Mediana → datos con outliers
* Moda → variables categóricas

---

## 6. Asignación a una categoría única

En variables categóricas, también se puede reemplazar el valor faltante con una nueva categoría.

```python
df["CouncilArea"] = df["CouncilArea"].fillna("Desconocido")
```

**Uso:** Variables categóricas donde el valor faltante puede representar una categoría válida.

---

## 7. Predicción de valores faltantes por regresión

También se pueden utilizar modelos predictivos para estimar valores ausentes.

En este ejemplo, se intentará predecir valores faltantes de `YearBuilt` usando otras variables numéricas del dataset.

```python
predictors = ["Rooms", "Distance", "Bathroom", "Car", "Landsize", "BuildingArea"]

df_reg = df[predictors + ["YearBuilt"]].copy()

df_reg[predictors] = df_reg[predictors].fillna(df_reg[predictors].median())

known_data = df_reg[df_reg["YearBuilt"].notnull()]
missing_data = df_reg[df_reg["YearBuilt"].isnull()]

model = LinearRegression()

model.fit(
    known_data[predictors],
    known_data["YearBuilt"]
)

predicted_values = model.predict(missing_data[predictors])

df.loc[df["YearBuilt"].isnull(), "YearBuilt"] = predicted_values
```

**Uso:** Cuando los datos tienen relaciones fuertes entre variables.

---

## 8. Verificar nuevamente valores nulos

Después del tratamiento, revisamos nuevamente los valores nulos.

```python
df.isnull().sum().sort_values(ascending=False)
```

---

# Escalado de Características en Machine Learning

El escalado normaliza las variables para que tengan una magnitud similar, evitando que algunas dominen el modelo.

---

## 9. Seleccionar variables numéricas

Seleccionamos algunas variables numéricas del dataset.

```python
features = ["Rooms", "Distance", "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt"]

X = df[features]
```

Antes de escalar, nos aseguramos de que no existan valores nulos en estas variables.

```python
X = X.fillna(X.median())
```

---

## 10. Normalización Min-Max Scaling

Aplicamos Min-Max Scaling para transformar los valores a un rango entre 0 y 1.

```python
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
```

Convertimos el resultado en un DataFrame para visualizarlo mejor.

```python
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

X_scaled_df.head()
```

**Rango:** 0 a 1

---

## Referencia

Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
Capítulo 2: *End-to-End Machine Learning Project*.
