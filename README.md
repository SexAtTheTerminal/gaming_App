# Gaming IQ - Academic Performance Predictor

Este proyecto tiene como objetivo analizar y predecir el impacto de los hábitos de juego (gaming) y otros factores de estilo de vida en el rendimiento académico de los estudiantes. El sistema compara dos enfoques de aprendizaje automático para determinar cuál ofrece una mayor precisión en las predicciones.

## 📊 Estructura del Proyecto

*   **`Gaming_Academic_Performance.csv`**: El conjunto de datos principal que contiene variables como horas de estudio, horas de juego, asistencia, nivel de estrés, calidad de sueño y las calificaciones finales (`grades`).
*   **`EDA.py`**: Script de Análisis Exploratorio de Datos. Se encarga de la limpieza de registros inconsistentes, detección de valores atípicos (outliers) mediante el método IQR y visualización de correlaciones.
*   **`rlm.py`**: Implementación de un modelo de **Regresión Lineal Múltiple**. Sirve como modelo base (baseline) para entender las relaciones lineales entre las variables y el rendimiento académico.
*   **`app.py`**: Aplicación web desarrollada en **Flask** que utiliza un modelo de **Gradient Boosting** (XGBooster). Incluye un panel interactivo con métricas de desempeño y un formulario de predicción en tiempo real.
*   **`templates/index.html`**: Interfaz de usuario para la visualización de resultados y predicciones.

## 🤖 Modelos Utilizados

El proyecto está diseñado para comparar los siguientes enfoques:

1.  **Regresión Lineal Múltiple (`rlm.py`)**:
    *   Utiliza `LinearRegression` de scikit-learn.
    *   Realiza codificación de variables categóricas mediante One-Hot Encoding.
    *   Incluye validación cruzada ($K=5$) para asegurar la estabilidad del modelo.
    *   Permite analizar los coeficientes ($\beta$) para interpretar el peso directo de cada variable.

2.  **Gradient Boosting / XGBoost (`app.py`)**:
    *   Utiliza `GradientBoostingRegressor` para capturar relaciones no lineales y complejas en los datos.
    *   Escalamiento de datos con `StandardScaler`.
    *   Proporciona la importancia de las variables (*Feature Importance*) para identificar qué factores afectan más al modelo.

## 🚀 Cómo Ejecutar el Proyecto

### Requisitos Previos
Asegúrate de tener instalado Python 3.x y las dependencias necesarias:
```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn
```

### Ejecución
1.  **Análisis de Datos**: Para ver el análisis estadístico y gráficos:
    ```bash
    python EDA.py
    ```
2.  **Modelo Lineal**: Para evaluar las métricas del modelo de regresión:
    ```bash
    python rlm.py
    ```
3.  **Aplicación Web**: Para iniciar el dashboard interactivo y el predictor:
    ```bash
    python app.py
    ```
    Luego, abre tu navegador en `http://127.0.0.1:5000`.

## 📈 Comparación de Resultados
El proyecto permite contrastar métricas como **R²**, **RMSE** y **MAE** entre ambos modelos. Mientras que la Regresión Lineal ofrece interpretabilidad, el modelo de Gradient Boosting en `app.py` suele ofrecer una mayor capacidad predictiva al manejar mejor las interacciones entre variables.
