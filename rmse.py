import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from recommendation import recommend_from_user_list, recommend_movies, movies, user_movie_sparse, movie_ids, model_knn

ratings = pd.read_csv('ml-32m/ratings.csv')
movies = pd.read_csv('ml-32m/movies.csv')

# Merge ratings with movie titles
movie_ratings = pd.merge(ratings, movies, on='movieId')

def calculate_rmse(predicted_ratings, actual_ratings):
    return np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))

# Predicting ratings using recommend_from_user_list
def predict_ratings_user_list(user_rated_movies, actual_ratings, user_movie_sparse, movie_ids, model_knn, movies_df):
    recommendations = recommend_from_user_list(
        user_rated_movies, user_movie_sparse, movie_ids, movies_df, model_knn, num_recommendations=len(user_rated_movies)
    )

    if recommendations.empty:
        return []

    predicted_ratings = []
    for movie in actual_ratings.index:
        if movie in recommendations['title'].values:
            predicted_ratings.append(4.0)
        else:
            predicted_ratings.append(2.0)

    return predicted_ratings

# Predicting ratings using collaborative filtering
def predict_ratings_collaborative(user_id, test_movies):
    recommendations = recommend_movies(user_id, num_recommendations=len(test_movies))

    if recommendations.empty:
        return []

    predicted_ratings = []
    for movie in test_movies['title'].values:
        if movie in recommendations['title'].values:
            predicted_ratings.append(4.0)
        else:
            predicted_ratings.append(2.0)

    return predicted_ratings

# Evaluating RMSE for multiple users
def evaluate_rmse_multiple_users(user_ids, movie_ratings, movies_df, user_movie_sparse, movie_ids, model_knn):
    rmse_user_list_all = []
    rmse_collab_all = []

    for user_id in user_ids:
        print(f"Evaluating RMSE for User ID: {user_id}")

        # Get the user's rated movies
        user_rated_movies = movie_ratings[movie_ratings['userId'] == user_id]

        # If user has less than 5 ratings, skip (to ensure enough data for train/test split)
        if len(user_rated_movies) < 5:
            print(f"Skipping User ID {user_id} - not enough ratings.")
            continue

        # Split into training (80%) and testing (20%) sets
        train_ratings = user_rated_movies.sample(frac=0.8, random_state=42)
        test_ratings = user_rated_movies.drop(train_ratings.index)

        # Extract the movie titles from the ratings DataFrame
        train_movies = train_ratings['title'].values
        test_movies = test_ratings[['title', 'rating']]

        # Predict ratings using the user's movie list (content/user-based)
        predicted_user_ratings = predict_ratings_user_list(
            train_movies, test_ratings.set_index('title')['rating'], user_movie_sparse, movie_ids, model_knn, movies_df
        )

        # Predict ratings using collaborative filtering
        predicted_collab_ratings = predict_ratings_collaborative(user_id, test_movies)

        # Actual ratings
        actual_ratings = test_ratings['rating'].values

        # Calculate RMSE for both methods
        if predicted_user_ratings:
            rmse_user_list = calculate_rmse(predicted_user_ratings, actual_ratings)
            rmse_user_list_all.append(rmse_user_list)
        if predicted_collab_ratings:
            rmse_collab = calculate_rmse(predicted_collab_ratings, actual_ratings)
            rmse_collab_all.append(rmse_collab)

    # Calculate average RMSE for all users
    avg_rmse_user_list = np.mean(rmse_user_list_all) if rmse_user_list_all else None
    avg_rmse_collab = np.mean(rmse_collab_all) if rmse_collab_all else None

    print(f"\nAverage RMSE for User List Recommendation (Content/User-based): {avg_rmse_user_list:.4f}")
    print(f"Average RMSE for Collaborative Filtering Recommendation: {avg_rmse_collab:.4f}")

if __name__ == "__main__":
    user_ids = movie_ratings['userId'].unique()[:2000]

    evaluate_rmse_multiple_users(
        user_ids=user_ids,             # List of user IDs to test
        movie_ratings=movie_ratings,   # Movie + ratings merged DataFrame
        movies_df=movies,              # Movie metadata
        user_movie_sparse=user_movie_sparse,  # User-movie sparse matrix
        movie_ids=movie_ids,           # Movie IDs for indexing
        model_knn=model_knn            # Trained NearestNeighbors model
    )
