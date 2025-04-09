import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
import gc

class ContentBasedFiltering:
    def __init__(self, dataframe):
        self.df = dataframe
        self.tfidf_matrix = None
        self.item_similarity_matrix = None
        # Imprimir columnas disponibles para diagnóstico
        print("Columnas disponibles en el DataFrame:", self.df.columns.tolist())

    def create_tfidf_matrix(self, description_column=None):
        # Intenta detectar automáticamente una columna adecuada si no se especifica
        if description_column is None:
            possible_columns = ['description', 'overview', 'title', 'genres', 'plot', 'synopsis', 'summary']
            for col in possible_columns:
                if col in self.df.columns:
                    description_column = col
                    print(f"Usando la columna '{col}' para análisis de texto.")
                    break
            else:
                # Si no encuentra ninguna columna adecuada, usa la primera columna de texto
                text_columns = self.df.select_dtypes(include=['object']).columns
                if len(text_columns) > 0:
                    description_column = text_columns[0]
                    print(f"No se encontró columna de descripción específica. Usando '{description_column}' como alternativa.")
                else:
                    raise ValueError("No se encontró ninguna columna de texto adecuada para TF-IDF.")
        
        # Verificar si la columna existe
        if description_column not in self.df.columns:
            raise KeyError(f"La columna '{description_column}' no existe en el DataFrame.")
            
        # Crear matriz TF-IDF
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.df[description_column].fillna(''))
        return self.tfidf_matrix

    def create_item_similarity_matrix(self, batch_size=1000):
        """
        Crea la matriz de similitud en lotes para conservar memoria
        """
        if self.tfidf_matrix is None:
            self.create_tfidf_matrix()
        
        n_samples = self.tfidf_matrix.shape[0]
        
        # Para conjuntos de datos muy grandes, ni siquiera almacenamos la matriz completa
        if n_samples > 10000:
            print(f"El conjunto de datos es muy grande ({n_samples} elementos). No se almacenará la matriz completa.")
            print("Se calcularán similitudes bajo demanda.")
            return None
        
        # Para conjuntos medianos, calculamos por lotes
        print(f"Calculando matriz de similitud para {n_samples} elementos en lotes de {batch_size}...")
        
        # Crear una matriz vacía para almacenar los resultados
        self.item_similarity_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)  # Usar float32 en lugar de float64
        
        # Calcular la similitud por lotes
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch = self.tfidf_matrix[i:end]
            
            # Calcular similitud entre este lote y todos los elementos
            similarities = linear_kernel(batch, self.tfidf_matrix)
            
            # Almacenar en la matriz final
            self.item_similarity_matrix[i:end, :] = similarities
            
            # Informar progreso
            print(f"Procesado lote {i//batch_size + 1}/{(n_samples-1)//batch_size + 1}")
            
            # Forzar liberación de memoria
            del similarities
            gc.collect()
            
        return self.item_similarity_matrix

    def recommend_items(self, item_id, top_n=10):
        """
        Recomienda elementos similares al elemento especificado
        """
        # Asegurarse de que tfidf_matrix esté inicializado
        if self.tfidf_matrix is None:
            try:
                self.create_tfidf_matrix()
            except Exception as e:
                print(f"Error al crear matriz TF-IDF: {e}")
                # Intentar crear una matriz simple con títulos si están disponibles
                if 'title' in self.df.columns:
                    print("Intentando crear matriz con títulos como fallback...")
                    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
                    self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.df['title'].fillna(''))
                elif 'genres' in self.df.columns:
                    print("Intentando crear matriz con géneros como fallback...")
                    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
                    self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.df['genres'].fillna(''))
                else:
                    # Si no tenemos columnas de texto, crear una matriz basada en características numéricas
                    print("No se encontraron columnas de texto. Usando características numéricas...")
                    num_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
                    if not num_features:
                        raise ValueError("No se encontraron columnas numéricas ni de texto para crear una matriz de similitud.")
                    # Normalizar características numéricas
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    self.tfidf_matrix = csr_matrix(scaler.fit_transform(self.df[num_features].fillna(0)))
        
        # Comprobar de nuevo, por si la creación falló
        if self.tfidf_matrix is None:
            raise ValueError("No se pudo crear la matriz TF-IDF. Verifica tus datos.")
        
        # Verificar que el item_id está dentro del rango
        n_samples = self.tfidf_matrix.shape[0]
        if item_id < 0 or item_id >= n_samples:
            raise ValueError(f"El item_id {item_id} está fuera de rango. Debe estar entre 0 y {n_samples-1}.")
        
        # Para datasets enormes o cuando la matriz de similitud no está calculada
        if self.item_similarity_matrix is None:
            # Calcular similitudes solo para este ítem específico
            item_vector = self.tfidf_matrix[item_id:item_id+1]
            item_similarities = linear_kernel(item_vector, self.tfidf_matrix)[0]
        else:
            item_similarities = self.item_similarity_matrix[item_id]
        
        # Obtener los top N ítems más similares (excluyendo el propio ítem)
        similar_indices = np.argsort(item_similarities)[::-1]
        similar_items = [idx for idx in similar_indices if idx != item_id][:top_n]
        
        # Devolver índices y puntuaciones
        scores = [item_similarities[idx] for idx in similar_items]
        return list(zip(similar_items, scores))