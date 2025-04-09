from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

class KNNModel:
    def __init__(self, dataframe, features, target):
        self.df = dataframe
        self.features = features
        self.target = target
        self.model = KNeighborsClassifier(n_neighbors=5)  # Puedes ajustar el número de vecinos

    def train(self):
        X = self.df[self.features]
        y = self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        return X_test, y_test

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy