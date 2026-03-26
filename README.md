# Trabajo Práctico N°1 – Modelo Predictivo de Tarifas de Uber

## Materia
Aprendizaje Automático I  

## Carrera
Tecnicatura en Inteligencia Artificial – FCEIA  

## Entrega
17/04/2026  

## Estudiantes
- Diez, Laureano 
- Duvia, Uriel  
- Quispe, Rocio  

---

## Descripción

Este proyecto corresponde al Trabajo Práctico N°1 de la asignatura Aprendizaje Automático I.  
El objetivo principal es desarrollar un modelo predictivo de tarifas de Uber, aplicando técnicas de análisis de datos, preprocesamiento y modelos de regresión utilizando la librería scikit-learn.

---

## Dataset

Se utilizó el dataset `uber_fares.csv`, que contiene información sobre viajes realizados con Uber.

### Variables de entrada (features)

- `key`: identificador único del viaje  
- `pickup_datetime`: fecha y hora de inicio  
- `passenger_count`: número de pasajeros  
- `pickup_longitude`, `pickup_latitude`: coordenadas de inicio  
- `dropoff_longitude`, `dropoff_latitude`: coordenadas de destino  

### Variable de salida (target)

- `fare_amount`: tarifa del viaje  

---

## Metodología

### 1. Análisis Exploratorio de Datos (EDA)

- Revisión de valores faltantes y atípicos  
- Visualización mediante histogramas y gráficos de dispersión  
- Análisis de correlación entre variables  

### 2. Preprocesamiento

- Limpieza de datos  
- Eliminación o tratamiento de valores faltantes  
- Preparación de variables para el modelado  
- División del dataset en conjuntos de entrenamiento y prueba  

### 3. Modelado

Se implementaron modelos de regresión utilizando scikit-learn:

- Regresión Lineal (`LinearRegression`)  
- Métodos basados en gradiente (`SGDRegressor`)  
- Técnicas de regularización:
  - Ridge  
  - Lasso  
  - Elastic Net  

### 4. Evaluación

Se utilizaron métricas para evaluar el desempeño de los modelos:

- Error Cuadrático Medio (MSE)  
- Raíz del Error Cuadrático Medio (RMSE)  
- Error Absoluto Medio (MAE)  
- Coeficiente de determinación (R²)  

Además, se realizaron:

- Análisis de residuos  
- Comparación entre modelos  

### 5. Optimización

- Ajuste de hiperparámetros  
- Evaluación del impacto de la regularización  
- Comparación de resultados para seleccionar el mejor modelo  

---

## Conclusiones

Se analizaron los resultados obtenidos a partir de distintos modelos de regresión, evaluando su capacidad predictiva y su comportamiento frente a diferentes configuraciones.  
El trabajo permite comprender la relación entre las variables del dataset y la tarifa del viaje, así como la importancia del preprocesamiento y la selección del modelo adecuado.

---

## Estructura del repositorio

- `TP-regresion-AA1.ipynb`: notebook con el desarrollo completo  
- `uber_fares.csv`: dataset utilizado  
- `README.md`: descripción del proyecto  
