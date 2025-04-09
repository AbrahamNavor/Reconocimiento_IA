import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gc

class CollaborativeFiltering:
    def __init__(self, dataframe):
        self.df = dataframe
        self.user_similarity_matrix = None
        # Verificar que las columnas necesarias existen
        required_columns = ['userId', 'movieId', 'rating']
        for col in required_columns:
            if col not in self.df.columns:
                print(f"Advertencia: La columna '{col}' no existe en el DataFrame.")
        
        print("Columnas disponibles en el DataFrame:", self.df.columns.tolist())

    def create_user_similarity_matrix(self):
        """
        Crea una matriz de similitud de usuarios basada en ratings
        """
        # Crear matriz pivote de usuarios vs. películas
        user_movie_matrix = self.df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
        
        # Calcular similitud entre usuarios
        self.user_similarity_matrix = cosine_similarity(user_movie_matrix)
        
        # Crear DataFrame para fácil acceso
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity_matrix,
            index=user_movie_matrix.index,
            columns=user_movie_matrix.index
        )
        
        return self.user_similarity_matrix

    def predict_rating(self, user_id, item_id):
        """
        Predice el rating que un usuario daría a un item
        """
        # Si no tenemos matriz de similitud, crearla
        if self.user_similarity_matrix is None:
            self.create_user_similarity_matrix()
        
        # Verificar si el usuario existe en nuestro dataset
        if user_id not in self.user_similarity_df.index:
            print(f"Usuario {user_id} no encontrado en el dataset.")
            return 0
        
        # Encontrar usuario más similar que ya haya clasificado el item
        similar_users = self.user_similarity_df[user_id].sort_values(ascending=False).index
        
        for similar_user in similar_users:
            # Verificar si el usuario similar ha calificado este item
            user_ratings = self.df[(self.df['userId'] == similar_user) & (self.df['movieId'] == item_id)]
            if not user_ratings.empty:
                rating = user_ratings['rating'].values[0]
                return rating
        
        # Si no encontramos ningún usuario similar que haya calificado, devolver promedio
        return self.df['rating'].mean()