import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df, features, target, test_size=0.2, random_state=42):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test