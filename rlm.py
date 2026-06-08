# =====================================================
# MODELO DE REGRESIÓN LINEAL MÚLTIPLE
# Proyecto: Gaming Academic Performance
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

# =====================================================
# 1. CARGA DE DATOS
# =====================================================

df = pd.read_csv("Gaming_Academic_Performance.csv")

# Limpieza de registros inconsistentes
df = df[
    (df["grades"] >= 0)
    & (df["grades"] <= 100)
    & (df["addiction_score"] >= 0)
].copy()

# =====================================================
# 2. VARIABLES
# =====================================================

X = df.drop("grades", axis=1)
y = df["grades"]

# =====================================================
# 3. ONE HOT ENCODING
# =====================================================

X = pd.get_dummies(
    X,
    columns=[
        "gender",
        "gaming_genre",
        "stress_level"
    ],
    drop_first=True
)

# =====================================================
# 4. PARTICIÓN 80/20
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# 5. MODELO
# =====================================================

modelo = LinearRegression()

modelo.fit(
    X_train,
    y_train
)

# =====================================================
# 6. PREDICCIONES
# =====================================================

y_pred = modelo.predict(X_test)

# =====================================================
# 7. MÉTRICAS
# =====================================================

r2 = r2_score(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

mape = mean_absolute_percentage_error(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("EVALUACIÓN DEL MODELO")
print("=" * 60)

print(f"R²   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.4f}")

# =====================================================
# 8. VALIDACIÓN CRUZADA K=5
# =====================================================

cv_scores = cross_val_score(
    modelo,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("\n" + "=" * 60)
print("VALIDACIÓN CRUZADA (K=5)")
print("=" * 60)

print("R² por Fold:")

for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score:.4f}")

print(
    f"\nR² Promedio: {cv_scores.mean():.4f}"
)

print(
    f"Desviación Estándar: {cv_scores.std():.4f}"
)

# =====================================================
# 9. COEFICIENTES
# =====================================================

coeficientes = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": modelo.coef_
})

coeficientes = coeficientes.sort_values(
    by="Coeficiente",
    ascending=False
)

print("\n" + "=" * 60)
print("COEFICIENTES DEL MODELO")
print("=" * 60)

print(coeficientes)

# =====================================================
# 10. TOP VARIABLES
# =====================================================

print("\nVariables con mayor impacto positivo:")
print(coeficientes.head(10))

print("\nVariables con mayor impacto negativo:")
print(coeficientes.tail(10))

# =====================================================
# 11. ECUACIÓN DEL MODELO
# =====================================================

print("\n" + "=" * 60)
print("ECUACIÓN DEL MODELO")
print("=" * 60)

print(
    f"Intercepto (β0): "
    f"{modelo.intercept_:.4f}"
)

print(
    "\nUtilice los coeficientes "
    "anteriores para construir "
    "la ecuación completa."
)

# =====================================================
# 12. SUPUESTO DE HOMOCEDASTICIDAD
# =====================================================

residuos = y_test - y_pred

plt.figure(figsize=(8,6))

plt.scatter(
    y_pred,
    residuos
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Valores Predichos")
plt.ylabel("Residuos")

plt.title(
    "Análisis de Residuos"
)

plt.show()

# =====================================================
# 13. REAL VS PREDICHO
# =====================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Valor Real")
plt.ylabel("Valor Predicho")

plt.title(
    "Valores Reales vs Predichos"
)

plt.show()

print("\n" + "=" * 60)
print("SUPUESTOS EVALUADOS")
print("=" * 60)