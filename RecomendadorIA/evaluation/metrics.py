from sklearn.metrics import mean_squared_error, precision_score, recall_score
import numpy as np

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

def calculate_recall(y_true, y_pred):
    return recall_score(y_true, y_pred)