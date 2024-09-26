import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Script location
current_directory = os.path.dirname(__file__)

# Paths to .csvs
movies_path = os.path.join(current_directory, 'ml-32m', 'movies.csv')
ratings_path = os.path.join(current_directory, 'ml-32m', 'ratings.csv')

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# Checking if data loaded correctly
print(movies.head())
print(ratings.head())

# Number of unique movies and users
print(f"Unique Movies: {movies['movieId'].nunique()}")
print(f"Unique Users: {ratings['userId'].nunique()}")

# Limit to 40000 movies by ratings
top_movies = ratings.groupby('movieId').size().nlargest(40000).index
movies = movies[movies['movieId'].isin(top_movies)]

# Movies + rankings merge
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Checking missing data
print(movie_ratings.isnull().sum())

# Changing genres output to make them a single string
movie_ratings['genres'] = movie_ratings['genres'].str.replace('|', ' ')
movies['genres'] = movies['genres'].str.replace('|', ' ')

# Content-based filtering

# TF-IDF matrix for the genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Similarity between movies based on genres
movie_similarity = cosine_similarity(tfidf_matrix)


def recommend_movies_by_genre(movie_name, movie_sim, movies_, num_recommendations=5):
    movie_idx = movies_[movies_['title'] == movie_name].index[0]
    similar_movies = np.argsort(-movie_sim[movie_idx])  # Sort by similarity
    similar_movie_indices = similar_movies[:num_recommendations]
    return movies_.iloc[similar_movie_indices]


# Collaborative filtering (Nearest Neighbors and sparse matrix)

# userId and movieId to get integer indices
movie_ratings['userId'] = movie_ratings['userId'].astype('category')
movie_ratings['movieId'] = movie_ratings['movieId'].astype('category')

# Sparse matrix with user indices and movie indices
user_movie_sparse = csr_matrix(
    (movie_ratings['rating'],
     (movie_ratings['userId'].cat.codes, movie_ratings['movieId'].cat.codes))
)

# NearestNeighbors for similar users
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
model_knn.fit(user_movie_sparse)


# Collaborative filtering
def recommend_movies(user_id, model, user_movie_mat, movie_, movies_df, num_recommendations=5):
    user_idx = user_id - 1  # Adjust for zero-based indexing

    # K-nearest neighbors for the user
    distances, indices = model.kneighbors(user_movie_mat[user_idx], n_neighbors=10)

    # Get the most similar users
    similar_users_indices = indices.flatten()

    # Aggregate ratings of similar users
    similar_users_ratings = user_movie_mat[similar_users_indices].mean(axis=0).A1  # 1D array

    # User's own ratings
    user_ratings = user_movie_mat[user_idx].toarray().flatten()

    # Movies that are yet to be rated
    unrated_movies_idx = np.where(user_ratings == 0)[0]

    # Empty dataframe if 0 unrated movies
    if len(unrated_movies_idx) == 0:
        print("User watched all the movies. No recommendations available.")
        return pd.DataFrame()

    # Sort movies by predicted ratings and recommend top ones
    recommended_movies_idx = np.argsort(-similar_users_ratings[unrated_movies_idx])[:num_recommendations]

    # Get the movie IDs of recommended movies
    recommended_movie_ids = movie_[unrated_movies_idx[recommended_movies_idx]]

    return movies_df[movies_df['movieId'].isin(recommended_movie_ids)][['title', 'genres']]


# Model Evaluation

def evaluate_model(user_movie_mat, model):
    # Select a small set of users to evaluate
    test_users = np.random.choice(user_movie_mat.shape[0], 1000, replace=False)

    # Predictions and actual ratings for evaluation
    predicted_ratings = []
    actual_ratings = []

    for user_idx in test_users:
        distances, indices = model.kneighbors(user_movie_mat[user_idx], n_neighbors=10)
        similar_users_indices = indices.flatten()

        # Combined ratings of similar users
        similar_users_ratings = user_movie_mat[similar_users_indices].mean(axis=0).A1

        # User's actual ratings
        user_ratings = user_movie_mat[user_idx].toarray().flatten()

        # Get the indices of rated movies
        rated_movies_idx = np.where(user_ratings > 0)[0]

        if len(rated_movies_idx) > 0:
            # Predicted and actual ratings for the user's rated movies
            predicted_ratings.extend(similar_users_ratings[rated_movies_idx])
            actual_ratings.extend(user_ratings[rated_movies_idx])

    # Calculate RMSE (root-mean square deviation)
    rmse = sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    print(f"Model RMSE: {rmse}")
    return rmse


# Combining recommendation systems

def hybrid_recommendation():
    # Ask the user to select the recommendation type
    rec_type = input(
        "Which recommendation approach do you want? (content-based / collaborative-based): ").strip().lower()

    if rec_type == "content-based":
        # Ask for movie name and number of recommendations
        movie_name = input("Enter the movie title: ").strip()
        num_recommendations = int(input("Enter the number of recommendations: "))
        recommendations = recommend_movies_by_genre(movie_name, movie_similarity, movies, num_recommendations)
        print(f"\nTop {num_recommendations} content-based recommendations based on '{movie_name}':")
        print(recommendations[['title', 'genres']])

    elif rec_type == "collaborative-based":
        # Ask for user ID and number of recommendations
        user_id = int(input("Enter the user ID: "))
        num_recommendations = int(input("Enter the number of recommendations: "))
        recommendations = recommend_movies(user_id, model_knn, user_movie_sparse, movie_ids, movies,
                                           num_recommendations)
        print(f"\nTop {num_recommendations} collaborative-based recommendations for user ID {user_id}:")
        print(recommendations[['title', 'genres']])

    else:
        print("Invalid recommendation type. Please choose 'content-based' or 'collaborative-based'.")


# Main body

# Get the actual movieId values back from categorical codes
movie_ids = movie_ratings['movieId'].cat.categories

# Evaluate the collaborative filtering model
evaluate_model(user_movie_sparse, model_knn)

# Run
hybrid_recommendation()
