import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Paths to .csvs
movies_path = os.path.join(os.path.dirname(__file__), 'ml-32m', 'movies.csv')
ratings_path = os.path.join(os.path.dirname(__file__), 'ml-32m', 'ratings.csv')

# Load movies and ratings data
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# Limit to 40000 movies due to perfomance issues
top_movies = ratings.groupby('movieId').size().nlargest(40000).index
movies = movies[movies['movieId'].isin(top_movies)]

# Movies + rankings merge
movie_ratings = pd.merge(ratings, movies, on='movieId')
movie_ratings['genres'] = movie_ratings['genres'].str.replace('|', ' ')
movies['genres'] = movies['genres'].str.replace('|', ' ')

# Content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
movie_similarity = cosine_similarity(tfidf_matrix)


def recommend_movies_by_genre(movie_name, num_recommendations=5):
    movie_idx = movies[movies['title'] == movie_name].index[0]
    similar_movies = np.argsort(-movie_similarity[movie_idx])
    similar_movie_indices = similar_movies[:num_recommendations]
    return movies.iloc[similar_movie_indices][['title', 'genres']]


# Collaborative filtering
movie_ratings['userId'] = movie_ratings['userId'].astype('category')
movie_ratings['movieId'] = movie_ratings['movieId'].astype('category')
user_movie_sparse = csr_matrix(
    (movie_ratings['rating'], (movie_ratings['userId'].cat.codes, movie_ratings['movieId'].cat.codes))
)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
model_knn.fit(user_movie_sparse)

movie_ids = movie_ratings['movieId'].cat.categories


def recommend_movies(user_id, num_recommendations=5):
    user_idx = user_id - 1  # Adjust for zero-based indexing
    distances, indices = model_knn.kneighbors(user_movie_sparse[user_idx], n_neighbors=10)
    similar_users_indices = indices.flatten()
    similar_users_ratings = user_movie_sparse[similar_users_indices].mean(axis=0).A1
    user_ratings = user_movie_sparse[user_idx].toarray().flatten()
    unrated_movies_idx = np.where(user_ratings == 0)[0]
    if len(unrated_movies_idx) == 0:
        return pd.DataFrame()  # Return empty if no recommendations
    recommended_movies_idx = np.argsort(-similar_users_ratings[unrated_movies_idx])[:num_recommendations]
    recommended_movie_ids = movie_ids[unrated_movies_idx[recommended_movies_idx]]
    return movies[movies['movieId'].isin(recommended_movie_ids)][['title', 'genres']]


# Recommendation based on the user's rated movies list
def recommend_from_user_list(user_rated_movies, user_movie_sparse, movie_ids, movies_df, model_knn, num_recommendations=5):
    # Ensure movieId is a categorical column
    if not pd.api.types.is_categorical_dtype(movies_df['movieId']):
        movies_df['movieId'] = movies_df['movieId'].astype('category')

    # Indices of rated movies
    rated_movie_indices = movies_df[movies_df['title'].isin(user_rated_movies)]['movieId'].cat.codes

    # Empty if empty
    if len(rated_movie_indices) == 0:
        return pd.DataFrame()

    # Avarage rantings from similar users
    similar_users_ratings = np.zeros(user_movie_sparse.shape[1])

    for movie_idx in rated_movie_indices:
        movie_column = user_movie_sparse[:, movie_idx].toarray().flatten()

        # Input shape for kneighbors
        movie_column_reshaped = movie_column.reshape(1, -1)

        # K-nearest neighbors who rated the same movies
        distances, indices = model_knn.kneighbors(movie_column_reshaped, n_neighbors=10)
        similar_users_indices = indices.flatten()

        # Mean rating from similar users
        user_ratings = user_movie_sparse[similar_users_indices].mean(axis=0).A1

        # Add to overall score
        similar_users_ratings += user_ratings

    # Remove reviewed movies
    user_ratings = user_movie_sparse[:, rated_movie_indices].mean(axis=0).A1
    unrated_movies_idx = np.where(user_ratings == 0)[0]

    # If nothing - return empty
    if len(unrated_movies_idx) == 0:
        return pd.DataFrame()

    # Sort movies by predicted ratings
    recommended_movies_idx = np.argsort(-similar_users_ratings[unrated_movies_idx])[:num_recommendations]

    # Get movie IDs and titles
    recommended_movie_ids = movie_ids[unrated_movies_idx[recommended_movies_idx]]

    return movies_df[movies_df['movieId'].isin(recommended_movie_ids)][['title', 'genres']]

