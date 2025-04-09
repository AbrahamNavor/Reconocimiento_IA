# Sistema de Recomendación con IA

Este proyecto implementa un sistema de recomendación utilizando técnicas de filtrado colaborativo, filtrado por contenido y algoritmos de aprendizaje automático como KNN, K-Means y Autoencoders.

## Estructura del Proyecto

RecomendadorIA/
├── data/
│   ├── dataset.csv  # Dataset (ej. MovieLens)
├── models/
│   ├── collaborative_filtering.py
│   ├── content_based_filtering.py
│   ├── knn_model.py
│   ├── k_means_model.py
│   ├── autoencoder_model.py
├── evaluation/
│   ├── metrics.py
├── interface/
│   ├── app.py       # Interfaz de usuario (Flask/Streamlit)
│   ├── templates/   # (Si usas Flask)
│   │   ├── index.html
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
├── utils/
│   ├── data_loader.py
│   ├── preprocessing.py
├── README.md
├── requirements.txt


## Instalación

1.  Clonar el repositorio.
2.  Crear un entorno virtual: `python3 -m venv venv`
3.  Activar el entorno virtual: `source venv/bin/activate` (o `venv\Scripts\activate` en Windows)
4.  Instalar las dependencias: `pip install -r requirements.txt`

## Uso

1.  Colocar el dataset en la carpeta `data/`.
2.  Ejecutar la interfaz de usuario: `streamlit run interface/app.py`

## Modelos Implementados

*   **Filtrado Colaborativo:** Recomienda ítems basados en las preferencias de usuarios similares.
*   **Filtrado por Contenido:** Recomienda ítems similares en contenido a los que el usuario ha mostrado interés.
*   **KNN:** Utiliza el algoritmo de los k-vecinos más cercanos para clasificar y recomendar ítems.
*   **K-Means:** Agrupa usuarios o ítems en clusters para realizar recomendaciones basadas en el cluster al que pertenecen.
*   **Autoencoder:** Utiliza una red neuronal autoasociativa para aprender representaciones latentes de los datos y realizar recomendaciones.

## Métricas de Evaluación

*   **RMSE:** Error cuadrático medio.
*   **Precisión:** Proporción de recomendaciones relevantes entre las realizadas.
*   **Recall:** Proporción de recomendaciones relevantes encontradas entre todas las posibles.