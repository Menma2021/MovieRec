import os
import pandas as pd

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

# Most rated movies
most_rated = ratings.groupby('movieId').size().sort_values(ascending=False).head(10)
print(movies[movies['movieId'].isin(most_rated.index)])

# Movies + rankings merge
movie_ratings = pd.merge(ratings, movies, on='movieId')
print(movie_ratings.head())

# Checking missing data
print(movie_ratings.isnull().sum())

# Changing genres output to make them a single string
movie_ratings['genres'] = movie_ratings['genres'].str.replace('|', ' ')
