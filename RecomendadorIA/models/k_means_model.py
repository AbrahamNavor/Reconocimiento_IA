from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

class KMeansModel:
    def __init__(self, dataframe, features, n_clusters=3):
        self.df = dataframe
        self.features = features
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()

    def preprocess(self):
        # Escalar las características
        X = self.df[self.features]
        self.scaled_features = self.scaler.fit_transform(X)
        return self.scaled_features

    def train(self):
        self.preprocess()
        self.model.fit(self.scaled_features)
        return self.model.labels_

    def get_clusters(self):
        return self.model.labels_