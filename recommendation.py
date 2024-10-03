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

# Limit to 60000 movies due to perfomance issues
top_movies = ratings.groupby('movieId').size().nlargest(60000).index
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
    # Ensure the movie exists within the top 60,000 movies
    if movie_name not in movies['title'].values:
        print(f"Movie '{movie_name}' not found in the top 60,000 movies.")
        return pd.DataFrame()  # Return an empty DataFrame if the movie is not found

    movie_idx = movies[movies['title'] == movie_name].index[0]

    # Check if movie_idx is within bounds (less than the size of the movie_similarity matrix)
    if movie_idx >= len(movie_similarity):
        print(f"Movie '{movie_name}' is outside the top 60,000 movies for recommendations.")
        return pd.DataFrame()

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


def recommend_from_user_list(user_rated_movies, user_movie_sparse, movie_ids, movies_df, model_knn,
                             num_recommendations=5):
    # Convert movie titles to movie IDs
    rated_movie_ids = movies_df[movies_df['title'].isin(user_rated_movies)]['movieId']

    # Ensure the movie IDs are only from the top 60,000 movies used in training
    top_movie_ids = movie_ids  # This corresponds to the top 60,000 movie IDs
    valid_movie_ids = rated_movie_ids[rated_movie_ids.isin(top_movie_ids)]

    # Convert to sparse matrix column indices (since we use the sparse matrix)
    valid_movie_indices = np.searchsorted(top_movie_ids, valid_movie_ids)

    if len(valid_movie_indices) == 0:
        print("No valid rated movies found for recommendation.")
        return pd.DataFrame()

    # Get the columns corresponding to these movies from the user-movie sparse matrix
    movie_column = user_movie_sparse[:, valid_movie_indices].mean(axis=1)

    # Reshape to pass to KNN (should have the same number of features as model_knn was trained on)
    movie_column_reshaped = np.asarray(movie_column).reshape(1, -1)[:, :60000]  # Convert to numpy array, limit to 40,000 features

    # Get similar users based on the rated movies
    distances, indices = model_knn.kneighbors(movie_column_reshaped, n_neighbors=10)

    # Aggregate ratings from similar users
    similar_users_ratings = user_movie_sparse[indices.flatten()].mean(axis=0).A1

    # Sort movies by predicted rating and recommend the top ones
    unrated_movie_indices = np.where(user_movie_sparse[:, valid_movie_indices].sum(axis=1) == 0)[0]
    unrated_movie_indices = unrated_movie_indices[unrated_movie_indices < 60000]  # Ensure they are within bounds
    recommended_movie_indices = np.argsort(-similar_users_ratings[unrated_movie_indices])[:num_recommendations]

    recommended_movie_ids = movie_ids[recommended_movie_indices]
    return movies_df[movies_df['movieId'].isin(recommended_movie_ids)][['title', 'genres']]




