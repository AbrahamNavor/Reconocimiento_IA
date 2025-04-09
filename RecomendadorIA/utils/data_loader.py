import pandas as pd

def load_movies():
    return pd.read_csv('data/movies.csv')

def load_ratings():
    return pd.read_csv('data/ratings.csv')

def load_links():
    return pd.read_csv('data/links.csv')

def load_tags():
    return pd.read_csv('data/tags.csv')