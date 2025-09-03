# notebook_script.py - A linear script version of the notebook for quick runs
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')
movies['genres'] = movies['genres'].fillna('')
movies['genres_clean'] = movies['genres'].str.replace('|', ' ')
vectorizer = TfidfVectorizer(stop_words='english')
genre_tfidf = vectorizer.fit_transform(movies['genres_clean'])
content_sim = cosine_similarity(genre_tfidf, genre_tfidf)
user_item = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
svd = TruncatedSVD(n_components=50, random_state=42)
item_factors = svd.fit_transform(user_item.T)
print('Ready. Use functions from the notebook to get recommendations.')
