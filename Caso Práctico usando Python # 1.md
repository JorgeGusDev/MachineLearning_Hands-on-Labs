# Caso Práctico usando Python # 1

## 1. Carga del Conjunto de Datos

El primer paso consiste en cargar el conjunto de datos y verificar su estructura general.

```python
import pandas as pd

df = pd.read_csv("dataset.csv")

df.head()
```

**Objetivo:** Familiarizarse con los atributos y registros disponibles.

---


## 2. Análisis Descriptivo

El análisis descriptivo resume estadísticamente las principales características de las variables numéricas.

```python
df.describe()
```

Las métricas obtenidas incluyen:

* Media
* Mediana
* Desviación estándar
* Valor mínimo
* Valor máximo
* Cuartiles

**Objetivo:** Comprender la distribución general de los datos.

---

## 3. Análisis Univariado

El análisis univariado estudia una variable de forma individual.

### Histograma

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df["variable"], kde=True)
plt.show()
```

### Diagrama de Caja (Boxplot)

```python
sns.boxplot(y=df["variable"])
plt.show()
```

**Objetivo:** Identificar la distribución, dispersión y posibles valores atípicos de una variable.

---

## 4. Análisis Bivariado

El análisis bivariado permite estudiar la relación entre dos variables.

### Gráfico de Dispersión

```python
sns.countplot(
    data=df,
    x="variable_x",
    y="variable_y"
)
plt.show()
```

### Matriz de Correlación

```python
corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True
)
plt.show()
```

**Objetivo:** Identificar asociaciones y relaciones entre variables.

---

## 5. Identificación de Anomalías (Outliers)

Los valores atípicos son observaciones que se encuentran alejadas del comportamiento general de los datos.

### Visualización mediante Boxplot

```python
sns.boxplot(y=df["variable"])
plt.show()
```


**Objetivo:** Detectar observaciones que podrían requerir revisión o tratamiento adicional.

---

## 6. Importancia para Inteligencia Artificial y Machine Learning

El análisis exploratorio de datos constituye una etapa fundamental antes de entrenar modelos de Inteligencia Artificial y Machine Learning, ya que permite:

* Comprender la estructura de los datos.
* Detectar anomalías.
* Analizar distribuciones y relaciones entre variables.

Una adecuada preparación de los datos contribuye a obtener modelos más precisos, robustos y confiables.

### Referencia

Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media. Capítulo 2: *End-to-End Machine Learning Project*.

| Gráfico         | Tipo de análisis             |
|-----------------|------------------------------|
| `countplot()`   | Frecuencia de categorías     |
| `histplot()`    | Distribución de una variable |
| `boxplot()`     | Dispersión y outliers        |
| `scatterplot()` | Relación entre dos variables |
| `violinplot()`  | Distribución + densidad + dispersión |
