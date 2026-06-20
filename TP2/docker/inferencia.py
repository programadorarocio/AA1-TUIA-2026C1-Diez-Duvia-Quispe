import joblib
import pandas as pd
import numpy as np
import logging
import sys
from tensorflow.keras.models import load_model
import os

# Configuración del Logger (Estilo Producción)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

def aplicar_ingenieria_variables(df):
    """Aplica las transformaciones trigonométricas y gradientes climáticos"""
    df_clean = df.copy()
    if 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        dia_del_año = df_clean['Date'].dt.dayofyear
        df_clean['Date_sin'] = np.sin(2 * np.pi * dia_del_año / 366.0)
        df_clean['Date_cos'] = np.cos(2 * np.pi * dia_del_año / 366.0)
    
    if all(c in df_clean.columns for c in ['MaxTemp', 'MinTemp', 'Humidity3pm', 'Humidity9am', 'Pressure3pm', 'Pressure9am']):
        df_clean['TempRange'] = df_clean['MaxTemp'] - df_clean['MinTemp']
        df_clean['HumDiff'] = df_clean['Humidity3pm'] - df_clean['Humidity9am']
        df_clean['PressureDiff'] = df_clean['Pressure3pm'] - df_clean['Pressure9am']
    
    return df_clean

def main():
    try:
        print("\n" + "="*60)
        logger.info("INICIANDO PROCESO DE INFERENCIA METEOROLÓGICA")
        print("="*60)

        # 1. LEER INPUT DESDE ARGUMENTO
        if len(sys.argv) < 2:
            logger.error("No se recibió el archivo CSV como argumento.")
            logger.error("Uso correcto: python inferencia.py <ruta_archivo.csv>")
            sys.exit(1)
        
        input_csv = sys.argv[1]
        logger.info(f"Leyendo datos crudos desde: {input_csv}")
        df_input = pd.read_csv(input_csv)
        logger.info(f"Volumen de datos: {df_input.shape[0]} observaciones registradas.")

        # Limpieza inicial de variables objetivo si vinieran en el CSV de prueba
        cols_drop = [c for c in ['RainTomorrow', 'RainfallTomorrow', 'Unnamed: 0'] if c in df_input.columns]
        if cols_drop:
            df_input = df_input.drop(columns=cols_drop)

        # 2. CARGAR ARTEFACTOS (Módulos independientes)
        logger.info("Cargando artefactos de Machine Learning (Imputadores, Scaler, Modelo)...")
        imp_cat = joblib.load('imputador_categorico.pkl')
        dict_knn = joblib.load('imputadores_knn.pkl')
        knn_por_ciudad, knn_global = dict_knn['por_ciudad'], dict_knn['global']
        scaler = joblib.load('escalador_robusto.pkl')
        columnas_entrenamiento = joblib.load('columnas_entrenamiento.pkl')
        
        # Ocultar warnings molestos de TensorFlow en consola
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        modelo_nn = load_model('modelo_red_neuronal.h5')

        # 3. PREPROCESAMIENTO EN CALIENTE
        logger.info("Ejecutando pipeline de transformación de datos...")
        
        df_procesado = aplicar_ingenieria_variables(df_input)
        df_procesado = df_procesado.drop(columns=['Date'], errors='ignore')

        cols_num = df_procesado.select_dtypes(include=[np.number]).columns.tolist()
        cols_cat = df_procesado.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

        # Imputación Moda
        df_procesado[cols_cat] = imp_cat.transform(df_procesado[cols_cat])

        # Escalado
        df_num_scaled = pd.DataFrame(scaler.transform(df_procesado[cols_num]), columns=cols_num, index=df_procesado.index)
        df_num_scaled['Location'] = df_procesado['Location'].copy()

        # Imputación KNN Agrupada
        for city in df_num_scaled['Location'].unique():
            mask = df_num_scaled['Location'] == city
            if mask.sum() > 0:
                city_data = df_num_scaled.loc[mask, cols_num].copy()
                cols_100_nan = city_data.columns[city_data.isna().all()]
                if len(cols_100_nan) > 0:
                    city_data[cols_100_nan] = 0.0
                
                if city in knn_por_ciudad:
                    df_num_scaled.loc[mask, cols_num] = knn_por_ciudad[city].transform(city_data)
                else:
                    df_num_scaled.loc[mask, cols_num] = knn_global.transform(city_data)

        df_procesado[cols_num] = df_num_scaled.drop(columns=['Location'])

        # OHE y Alineación Estructural
        df_final = pd.get_dummies(df_procesado, columns=cols_cat, drop_first=True)
        for col in columnas_entrenamiento:
            if col not in df_final.columns:
                df_final[col] = 0
        df_final = df_final[columnas_entrenamiento].astype('float32')

        # 4. PREDICCIÓN
        logger.info("Calculando probabilidades con Red Neuronal Base...")
        probabilidades_lluvia = modelo_nn.predict(df_final, verbose=0).flatten()
        
        # Aplicamos el umbral estándar (0.5). Si en tu TP justificaste 0.21, cambialo aquí.
        predicciones = (probabilidades_lluvia >= 0.5).astype(int)

        # 5. GENERAR OUTPUT
        logger.info("Estructurando archivo de salida...")
        prediction_labels = ['Andá al parque tranquilo' if pred == 0 else 'Llevá paraguas' for pred in predicciones]
        
        # Calculamos confianza (qué tan seguro está el modelo de su propia respuesta)
        confianza = [prob if pred == 1 else (1 - prob) for prob, pred in zip(probabilidades_lluvia, predicciones)]

        output_df = pd.DataFrame({
            'Location': df_input['Location'] if 'Location' in df_input.columns else 'Desconocida',
            'Predicción': prediction_labels,
            'Probabilidad_Lluvia': np.round(probabilidades_lluvia, 4),
            'Confianza_Modelo': np.round(confianza, 4)
        })

        output_path = './predicciones_clima.csv'
        output_df.to_csv(output_path, index=False)
        logger.info(f"Éxito: Archivo guardado correctamente en {output_path}")

        # 6. RESUMEN ESTADÍSTICO EN CONSOLA
        print("\n" + "="*60)
        print("RESUMEN DE PREDICCIONES")
        print("="*60)
        print(f"Total de días evaluados: {len(predicciones)}")
        print(f"  - Días Despejados (No lloverá): {(predicciones == 0).sum()} ({(predicciones == 0).sum()/len(predicciones)*100:.1f}%)")
        print(f"  - Alertas de Tormenta (Sí lloverá): {(predicciones == 1).sum()} ({(predicciones == 1).sum()/len(predicciones)*100:.1f}%)")
        print(f"\nNivel de confianza promedio del algoritmo: {np.mean(confianza):.2%}")
        
        print("\nMuestra de resultados generados:")
        print(output_df.head())

        print("\n" + "="*60)
        logger.info("Inferencia finalizada. Sistema apagado.")
        print("="*60 + "\n")

    except FileNotFoundError as e:
        logger.error(f"Error Crítico: No se encontró un archivo vital - {e}")
        logger.error("Verifique que los archivos .pkl, .keras y el CSV de entrada estén en la misma carpeta o el volumen de Docker esté bien montado.")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error inesperado durante la ejecución: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()