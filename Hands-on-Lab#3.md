# Hands-on-Lab#3: Decision Trees for Classification

## 1. Dataset

**Breast Cancer Wisconsin Diagnostic Dataset**
- 569 records.
- 30 numerical features.
- Target variable: `diagnosis` (Benign / Malignant).
- Features are derived from digitized images of fine needle aspirate (FNA) of breast masses.

**Dataset on Hugging Face:**
[Breast Cancer Wisconsin Diagnostic Dataset](https://huggingface.co/datasets/scikit-learn/breast-cancer-wisconsin)

---

## 2. Implementation

### Step 1. Install and Import Libraries

```python
!pip install -q datasets scikit-learn matplotlib seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
```

### Step 2. Load the Dataset from Hugging Face

```python
dataset = load_dataset(
    "scikit-learn/breast-cancer-wisconsin"
)

df = dataset["train"].to_pandas()

df.head()
```

### Step 3. Data Preparation

```python
# Remove unnecessary columns
df = df.drop(
    columns=["id", "Unnamed: 32"],
    errors="ignore"
)

# Separate features and target variable
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Step 4. Train the Decision Tree

```python
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### Step 5. Evaluate the Model

```python
print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

### Step 6. Visualize the Decision Tree

```python
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True
)

plt.show()
```

### Step 7. Analyze Feature Importance

```python
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance.head(10))

importance.head(10).plot(kind="barh")

plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances")
plt.show()
```
### Step ?. RandomForest (optional)
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=4,
    #min_samples_leaf=3,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:",
      accuracy_score(y_test, y_pred_rf))

print(classification_report(y_test, y_pred_rf))

print(confusion_matrix(
    y_test, y_pred_rf, labels=["B", "M"]
))
```
### Step ?. LightGBM (optional)
```python
# Install LightGBM
!pip install -q lightgbm

from lightgbm import LGBMClassifier

# Train LightGBM
lgbm = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    verbosity=-1,
    random_state=42
)

lgbm.fit(X_train, y_train)

# Make predictions
y_pred_lgbm = lgbm.predict(X_test)

# Evaluate LightGBM
print("LightGBM Accuracy:",
      accuracy_score(y_test, y_pred_lgbm))

print(classification_report(y_test, y_pred_lgbm))

# Compare models
print("Decision Tree Accuracy:",
      accuracy_score(y_test, y_pred))

print("LightGBM Accuracy:",
      accuracy_score(y_test, y_pred_lgbm))

# Compare false negatives (Malignant = M)
fn_dt = ((y_test == "M") & (y_pred == "B")).sum()
fn_lgbm = ((y_test == "M") & (y_pred_lgbm == "B")).sum()

print("Decision Tree FN:", fn_dt)
print("LightGBM FN:", fn_lgbm)

# Confusion matrix
sns.heatmap(
    confusion_matrix(y_test, y_pred_lgbm, labels=["B", "M"]),
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("LightGBM Confusion Matrix")
plt.show()
```
---

## 3. Statement

Utilizando el dataset **Iris**, disponible en Hugging Face (`scikit-learn/iris`), desarrolla un modelo de árbol de decisión para clasificar flores en tres especies: Setosa, Versicolor y Virginica. Realiza la exploración y preparación de los datos, el entrenamiento del modelo y su evaluación mediante métricas apropiadas. Visualiza e interpreta el árbol generado, experimenta con diferentes hiperparámetros y compara los resultados obtenidos. Finalmente, entrega un notebook Jupyter (`.ipynb`) que incluya la implementación, los resultados, las visualizaciones y una breve discusión sobre el rendimiento y la interpretabilidad del modelo.

**Dataset:** https://huggingface.co/datasets/scikit-learn/iris



