FROM python:3.11-slim

# Configurar el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el requirements primero (Buena práctica para aprovechar el caché de Docker)
COPY requirements.txt ./

# Instalar dependencias sin guardar caché temporal para mantener la imagen liviana
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos (inferencia.py, los .pkl y el .keras)
COPY . .

# Verificaciones de seguridad: El contenedor no se construirá si falta algún artefacto vital
RUN test -f imputador_categorico.pkl || (echo "ERROR: imputador_categorico.pkl no encontrado" && exit 1)
RUN test -f imputadores_knn.pkl || (echo "ERROR: imputadores_knn.pkl no encontrado" && exit 1)
RUN test -f escalador_robusto.pkl || (echo "ERROR: escalador_robusto.pkl no encontrado" && exit 1)
RUN test -f columnas_entrenamiento.pkl || (echo "ERROR: columnas_entrenamiento.pkl no encontrado" && exit 1)
RUN test -f modelo_red_neuronal.h5 || (echo "ERROR: modelo_red_neuronal.h5 no encontrado" && exit 1)

# Usamos ENTRYPOINT para que el contenedor reciba dinámicamente el archivo CSV como argumento
ENTRYPOINT ["python", "inferencia.py"]