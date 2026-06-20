# Sistema de Inferencia Meteorológica - MLOps

Este directorio contiene el entorno Dockerizado para la predicción de lluvia en Australia utilizando una Red Neuronal (TensorFlow/Keras) y un pipeline de preprocesamiento estructurado.

## Archivos incluidos
* `inferencia.py`: Script principal de ejecución.
* `requirements.txt`: Dependencias de Python.
* `*.pkl` y `*.keras`: Artefactos serializados del preprocesamiento y el modelo entrenado.
* `Dockerfile`: Instrucciones de construcción de la imagen.

## Instrucciones de Uso

### 1. Construir la imagen de Docker
Abre una terminal en esta misma carpeta y ejecuta el siguiente comando para construir la imagen. Le asignaremos el nombre `prediccion-clima`:

```bash
docker build -t prediccion-clima .