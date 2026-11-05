
# LAB: Decision Trees for Classification

Dataset

[Multiclass classification of breast cancer histopathology images using multilevel features of deep convolutional neural network | Scientific Reports](https://images.openai.com/static-rsc-4/ONCOYoIZoFbXjhzLYuKkxnwOnlo9Cqu5mWKqeemBWwkhqj8O0sAytIIAa40NZt9BX_aOGxWx6bf8qxWXjmigsTtAJryVGLtcgVTot345Nxua0WcY-6RtJ7FzEK68HWmMVfn7NIL1mId0WK1bCHUkjUgNCj7r70JiM5z0DV7A8tE?purpose=inline)

Breast Cancer Wisconsin Diagnostic Dataset

- 569 registros.
- 30 características numéricas.
- Variable objetivo: `diagnosis` (Benign / Malignant).
- Datos derivados de imágenes de aspiración con aguja fina.

[Dataset en Hugging Face](https://huggingface.co/datasets/scikit-learn/breast-cancer-wisconsin?utm_source=chatgpt.com)

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Datasets at Hugging Face+1

## Código 

1\. Instalar e importar librerías

```
!pip install -q datasets scikit-learn matplotlib seabornimport pandas as pdimport matplotlib.pyplot as pltimport seaborn as snsfrom datasets import load_datasetfrom sklearn.model_selection import train_test_splitfrom sklearn.tree import DecisionTreeClassifier, plot_treefrom sklearn.metrics import (    accuracy_score, classification_report,    confusion_matrix)
```

2\. Cargar dataset desde Hugging Face

```
dataset = load_dataset(    "scikit-learn/breast-cancer-wisconsin")df = dataset["train"].to_pandas()df.head()
```

3\. Preparar los datos

```
# Eliminar columnas que no aportan al modelodf = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")X = df.drop(columns=["diagnosis"])y = df["diagnosis"]X_train, X_test, y_train, y_test = train_test_split(    X, y,    test_size=0.2,    random_state=42,    stratify=y)
```

4\. Entrenar el árbol de decisión

```
model = DecisionTreeClassifier(    criterion="gini",    max_depth=4,    random_state=42)model.fit(X_train, y_train)y_pred = model.predict(X_test)
```

5\. Evaluar el modelo

```
print("Accuracy:", accuracy_score(y_test, y_pred))print(classification_report(y_test, y_pred))sns.heatmap(    confusion_matrix(y_test, y_pred),    annot=True,    fmt="d",    cmap="Blues")plt.xlabel("Predicted")plt.ylabel("Actual")plt.show()
```

6\. Visualizar el árbol

```
plt.figure(figsize=(20, 10))plot_tree(    model,    feature_names=X.columns,    class_names=model.classes_,    filled=True,    rounded=True)plt.show()
```

7\. Analizar la importancia de características

```
importance = pd.Series(    model.feature_importances_,    index=X.columns).sort_values(ascending=False)print(importance.head(10))importance.head(10).plot(kind="barh")plt.gca().invert_yaxis()plt.title("Top 10 Feature Importances")plt.show()
```

## Entregable

- Notebook `.ipynb` con el desarrollo del laboratorio.
- Métricas de evaluación obtenidas.
- Visualización del árbol de decisión.
- Breve análisis de los resultados y del efecto de modificar `max_depth`.

## Pregunta de reflexión

¿Por qué un árbol de decisión con mayor profundidad no necesariamente ofrece mejores predicciones sobre datos nuevos?

Nota académica: el dataset es de diagnóstico histopatológico y se utiliza exclusivamente con fines educativos. El modelo no debe emplearse para realizar diagnósticos médicos reales.

Duración estimada: 60–90 minutos.

El laboratorio permite recorrer el flujo completo: datos → entrenamiento → evaluación → interpretación, sin introducir todavía Random Forest ni otros métodos de ensemble.
