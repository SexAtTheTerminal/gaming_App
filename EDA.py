# =====================================================
# EDA - Gaming Academic Performance Dataset
# Autor: Gabriel Isaac Cuba García
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# 1. CARGA DEL DATASET
# =====================================================

df = pd.read_csv("Gaming_Academic_Performance.csv")

print("=" * 60)
print("INFORMACIÓN GENERAL")
print("=" * 60)

print("\nDimensiones:")
print(df.shape)

print("\nTipos de datos:")
print(df.dtypes)

print("\nPrimeras filas:")
print(df.head())

# =====================================================
# 2. VALORES NULOS
# =====================================================

print("\n" + "=" * 60)
print("VALORES NULOS")
print("=" * 60)

print(df.isnull().sum())

# =====================================================
# 3. ESTADÍSTICOS DESCRIPTIVOS
# =====================================================

print("\n" + "=" * 60)
print("ESTADÍSTICOS DESCRIPTIVOS")
print("=" * 60)

print(df.describe())

# =====================================================
# 4. DETECCIÓN DE REGISTROS INVÁLIDOS
# =====================================================

print("\n" + "=" * 60)
print("REGISTROS INVÁLIDOS")
print("=" * 60)

grades_mayor_100 = (df["grades"] > 100).sum()
grades_menor_0 = (df["grades"] < 0).sum()
addiction_negativo = (df["addiction_score"] < 0).sum()

print(f"Grades > 100: {grades_mayor_100}")
print(f"Grades < 0: {grades_menor_0}")
print(f"Addiction Score < 0: {addiction_negativo}")

# =====================================================
# 5. LIMPIEZA
# =====================================================

print("\n" + "=" * 60)
print("LIMPIEZA DE DATOS")
print("=" * 60)

df_clean = df[
    (df["grades"] >= 0)
    & (df["grades"] <= 100)
    & (df["addiction_score"] >= 0)
].copy()

print(f"Registros originales: {len(df)}")
print(f"Registros después de limpieza: {len(df_clean)}")
print(f"Registros eliminados: {len(df)-len(df_clean)}")


variables_interes = [
    'grades',
    'study_hours',
    'gaming_hours',
    'attendance',
    'sleep_hours',
    'addiction_score'
]

estadisticas = df_clean[variables_interes].describe().round(2)

print(estadisticas)

# =====================================================
# 6. HISTOGRAMA DE LA VARIABLE OBJETIVO
# =====================================================

plt.figure(figsize=(8,5))
sns.histplot(df_clean["grades"], bins=30)
plt.title("Distribución de Grades")
plt.xlabel("Grades")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# =====================================================
# 7. MATRIZ DE CORRELACIÓN
# =====================================================

numeric_df = df_clean.select_dtypes(include=np.number)

corr = numeric_df.corr()

print("\n" + "=" * 60)
print("CORRELACIÓN CON GRADES")
print("=" * 60)

print(
    corr["grades"]
    .sort_values(ascending=False)
)

# =====================================================
# 8. HEATMAP
# =====================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Matriz de Correlación")
plt.tight_layout()
plt.show()

# =====================================================
# 9. TOP VARIABLES MÁS RELACIONADAS CON GRADES
# =====================================================

top_corr = (
    corr["grades"]
    .drop("grades")
    .abs()
    .sort_values(ascending=False)
)

print("\nVariables más relacionadas con grades:")
print(top_corr)

# =====================================================
# 10. GRÁFICOS DE DISPERSIÓN
# =====================================================

variables_scatter = [
    "study_hours",
    "attendance",
    "addiction_score"
]

for var in variables_scatter:

    plt.figure(figsize=(7,5))

    sns.scatterplot(
        data=df_clean,
        x=var,
        y="grades"
    )

    plt.title(f"Grades vs {var}")

    plt.tight_layout()
    plt.show()

# =====================================================
# 11. BOXPLOTS (OUTLIERS)
# =====================================================

variables_boxplot = [
    "grades",
    "gaming_hours",
    "study_hours",
    "attendance",
    "addiction_score"
]

for var in variables_boxplot:

    plt.figure(figsize=(8,4))

    sns.boxplot(
        x=df_clean[var]
    )

    plt.title(f"Boxplot - {var}")

    plt.tight_layout()
    plt.show()

# =====================================================
# 12. DETECCIÓN DE OUTLIERS POR IQR
# =====================================================

print("\n" + "=" * 60)
print("OUTLIERS (MÉTODO IQR)")
print("=" * 60)

for col in numeric_df.columns:

    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df_clean[
        (df_clean[col] < lower)
        | (df_clean[col] > upper)
    ]

    print(
        f"{col}: {len(outliers)} outliers "
        f"({len(outliers)/len(df_clean)*100:.2f}%)"
    )

# =====================================================
# 13. RESUMEN PARA EL INFORME
# =====================================================

print("\n" + "=" * 60)
print("RESUMEN DEL EDA")
print("=" * 60)

print("""
1. Se verificó la existencia de valores nulos.
2. Se calcularon estadísticos descriptivos.
3. Se identificaron registros inconsistentes.
4. Se eliminaron grades > 100 y addiction_score < 0.
5. Se analizaron correlaciones.
6. Se generaron gráficos de dispersión.
7. Se identificaron outliers mediante boxplots e IQR.
8. El dataset quedó listo para modelado.
""")