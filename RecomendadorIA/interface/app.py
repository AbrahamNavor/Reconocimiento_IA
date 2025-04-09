import streamlit as st
import pandas as pd
import numpy as np
import gc
from models.collaborative_filtering import CollaborativeFiltering
from models.content_based_filtering import ContentBasedFiltering
from models.knn_model import KNNModel
from models.k_means_model import KMeansModel
from models.autoencoder_model import AutoencoderTrainer
from utils.data_loader import load_movies, load_ratings, load_links, load_tags

# Configuración de la página
st.set_page_config(page_title="Sistema de Recomendación con IA", layout="wide")

# Título de la app
st.title('Sistema de Recomendación con IA')

# Función para cargar datos con caché
@st.cache_data
def load_data():
    movies_df = load_movies()
    ratings_df = load_ratings()
    links_df = load_links()
    tags_df = load_tags()
    
    # Si el dataset es muy grande, podemos limitarlo para pruebas
    # ratings_df = ratings_df.sample(n=100000, random_state=42) if len(ratings_df) > 100000 else ratings_df
    
    # Unir solo los datasets necesarios para reducir tamaño
    df = pd.merge(ratings_df, movies_df, on='movieId')
    
    return df, movies_df, ratings_df, links_df, tags_df

# Carga de datos solo cuando sea necesario
with st.spinner('Cargando datos...'):
    df, movies_df, ratings_df, links_df, tags_df = load_data()
    st.success(f"Datos cargados: {len(df)} filas, {df.shape[1]} columnas")

# Sidebar para selección de modelo
model_selection = st.sidebar.selectbox(
    'Selecciona el modelo de recomendación:',
    ('Filtrado Colaborativo', 'Filtrado por Contenido', 'KNN', 'K-Means', 'Autoencoder')
)

# Inicialización bajo demanda de modelos
if model_selection == 'Filtrado Colaborativo':
    with st.spinner('Inicializando modelo de filtrado colaborativo...'):
        cf_model = CollaborativeFiltering(df)
        # Entrenar solo si se necesita una predicción
        train_model = st.checkbox('Entrenar modelo', value=False)
        
        user_id = st.number_input('Ingresa el ID del usuario:', min_value=1, max_value=df['userId'].max(), value=1)
        movie_id = st.number_input('Ingresa el ID de la película:', min_value=1, max_value=df['movieId'].max(), value=1)
        
        if st.button('Predecir Rating'):
            if train_model:
                with st.spinner('Entrenando modelo...'):
                    cf_model.create_user_similarity_matrix()
            
            rating = cf_model.predict_rating(user_id, movie_id)
            st.write(f'Rating predicho para el usuario {user_id} y la película {movie_id}: {rating}')
            
            # Liberar memoria
            del cf_model
            gc.collect()

elif model_selection == 'Filtrado por Contenido':
    with st.spinner('Inicializando modelo de filtrado por contenido...'):
        cb_model = ContentBasedFiltering(df)
        
        # UI para recomendaciones
        if 'movieId' in df.columns:
            movie_options = movies_df[['movieId', 'title']].drop_duplicates()
            selected_movie = st.selectbox(
                'Selecciona una película:',
                options=movie_options['title'].tolist()
            )
            movie_id = movie_options[movie_options['title'] == selected_movie]['movieId'].values[0]
            movie_index = df[df['movieId'] == movie_id].index[0] if len(df[df['movieId'] == movie_id]) > 0 else 0
        else:
            movie_index = st.number_input('Ingresa el índice de la película:', min_value=0, max_value=len(df)-1, value=0)
            
        num_recommendations = st.slider('Número de recomendaciones:', min_value=1, max_value=20, value=5)
        
        if st.button('Recomendar Películas'):
            with st.spinner('Calculando recomendaciones...'):
                recommendations = cb_model.recommend_items(movie_index, top_n=num_recommendations)
                
                st.subheader(f'Películas similares a "{movies_df.loc[movies_df["movieId"] == movie_id, "title"].values[0] if "movieId" in df.columns else movie_index}"')
                
                # Mostrar recomendaciones
                for idx, score in recommendations:
                    if 'title' in df.columns:
                        title = df.iloc[idx]['title']
                    elif 'title' in movies_df.columns and 'movieId' in df.columns:
                        movie_id = df.iloc[idx]['movieId']
                        title = movies_df.loc[movies_df['movieId'] == movie_id, 'title'].values[0] if len(movies_df.loc[movies_df['movieId'] == movie_id]) > 0 else f"Película {movie_id}"
                    else:
                        title = f"Ítem {idx}"
                        
                    st.write(f"• {title} (Similitud: {score:.4f})")
            
            # Liberar memoria
            del cb_model
            gc.collect()

elif model_selection == 'KNN':
    with st.spinner('Inicializando modelo KNN...'):
        # Configuración del modelo KNN
        features = st.multiselect(
            'Selecciona características para el modelo:',
            options=[col for col in df.columns if df[col].dtype in [np.int64, np.float64]],
            default=['rating', 'movieId'] if 'rating' in df.columns and 'movieId' in df.columns else None
        )
        
        target = st.selectbox(
            'Selecciona la variable objetivo:',
            options=[col for col in df.columns if col not in features],
            index=0
        )
        
        if st.button('Entrenar Modelo KNN'):
            if not features or not target:
                st.error('Por favor selecciona características y variable objetivo válidas.')
            else:
                with st.spinner('Entrenando modelo KNN...'):
                    knn_model = KNNModel(df, features=features, target=target)
                    knn_X_test, knn_y_test = knn_model.train()
                    accuracy = knn_model.evaluate(knn_X_test, knn_y_test)
                    
                    st.success(f'Entrenamiento completado!')
                    st.metric('Precisión del modelo KNN', f"{accuracy:.4f}")
                
                # Liberar memoria
                del knn_model
                gc.collect()

elif model_selection == 'K-Means':
    with st.spinner('Inicializando modelo K-Means...'):
        # Configuración del modelo K-Means
        features = st.multiselect(
            'Selecciona características para el modelo:',
            options=[col for col in df.columns if df[col].dtype in [np.int64, np.float64]],
            default=['rating', 'movieId'] if 'rating' in df.columns and 'movieId' in df.columns else None
        )
        
        n_clusters = st.slider('Número de clusters:', min_value=2, max_value=10, value=3)
        
        if st.button('Entrenar Modelo K-Means'):
            if not features:
                st.error('Por favor selecciona características válidas.')
            else:
                with st.spinner('Entrenando modelo K-Means...'):
                    kmeans_model = KMeansModel(df, features=features, n_clusters=n_clusters)
                    kmeans_clusters = kmeans_model.train()
                    
                    st.success(f'Entrenamiento completado!')
                    
                    # Mostrar distribución de clusters
                    cluster_counts = np.bincount(kmeans_clusters)
                    for i, count in enumerate(cluster_counts):
                        st.write(f"Cluster {i}: {count} elementos")
                
                # Liberar memoria
                del kmeans_model
                gc.collect()

elif model_selection == 'Autoencoder':
    with st.spinner('Inicializando modelo Autoencoder...'):
        # Configuración del modelo Autoencoder
        features = st.multiselect(
            'Selecciona características para el modelo:',
            options=[col for col in df.columns if df[col].dtype in [np.int64, np.float64]],
            default=['rating', 'movieId'] if 'rating' in df.columns and 'movieId' in df.columns else None
        )
        
        encoding_dim = st.slider('Dimensión del encoding:', min_value=2, max_value=64, value=32)
        
        if st.button('Entrenar Modelo Autoencoder'):
            if not features:
                st.error('Por favor selecciona características válidas.')
            else:
                with st.spinner('Entrenando modelo Autoencoder (esto puede tomar tiempo)...'):
                    ae_model = AutoencoderTrainer(df, features=features, encoding_dim=encoding_dim)
                    
                    try:
                        ae_model.train()
                        st.success(f'Entrenamiento completado!')
                        
                        # Reconstruir datos y mostrar error
                        reconstructed = ae_model.reconstruct_data()
                        mse = ae_model.evaluate()
                        st.metric('Error Cuadrático Medio (MSE)', f"{mse:.6f}")
                        
                        st.write("Muestra de datos reconstruidos:")
                        st.dataframe(pd.DataFrame(reconstructed[:5], columns=features))
                        
                    except Exception as e:
                        st.error(f"Error durante el entrenamiento: {str(e)}")
                
                # Liberar memoria
                del ae_model
                gc.collect()

# Información adicional
with st.sidebar.expander("Información del Dataset"):
    st.write(f"Total de películas: {len(movies_df)}")
    st.write(f"Total de ratings: {len(ratings_df)}")
    st.write(f"Usuarios únicos: {ratings_df['userId'].nunique()}")
    st.write(f"Rating promedio: {ratings_df['rating'].mean():.2f}")

# Nota sobre el tamaño del dataset (especialmente útil para datasets grandes)
total_records = len(df)
if total_records > 10000:
    st.sidebar.warning(f"El dataset es grande ({total_records:,} registros). Si experimentas problemas de memoria, considera usar un subset de datos más pequeño.")