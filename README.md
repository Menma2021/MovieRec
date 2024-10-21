# Movie Recommendation System
## Overview
This project is a Movie Recommendation System using Tkinter for the GUI and the k-Nearest Neighbors (k-NN) algorithm for generating personalized recommendations. The application allows users to input their movie preferences and receive content-based/collaborative filtering or user-based recommendations. The movie dataset is sourced from the MovieLens dataset

## Features
- Search movie by title: autocomplete search functionality for selecting movies
- Rate movies: users can rate movies on a scale of 1 to 5
- Add/Remove movies from rating list: easily manage a personal movie list
- Content-based recommendations: suggests movies based on genre similarities
- User-based recommendations: recommends movies by analyzing a user's movie list using k-NN
- Collaborative filtering: movie suggestions based on user similarity using the k-NN model
- Recommendation display: shows up to 5 movies for each recommendation type
  
## Installation
### Prerequisites
- Python 3.6+ (used 3.12)
- Tkinter for UI
- Pandas for data manipulation
- NumPy for numerical operations
- scikit-learn for machine learning algorithms (k-NN)
- MovieLens dataset (movies.csv and ratings.csv files) (provided in the project repository)

### Steps to Set Up
1. Clone the repository:
  ```bash
  git clone https://github.com/Menma2021/MovieRec.git
  cd MovieRec
  ```
2. Install necessary libraries (or use environment in a repository)
  ```bash
  pip install pandas numpy scikit-learn
  ```
3. Ensure the MovieLens dataset CSV files (movies.csv, ratings.csv) are placed in the correct directory (ml-32m).

4. Run the application

## Usage
- Search for a movie:
  1. Use the search bar to look for a movie by title
  2. Select a movie from the dropdown autocomplete results

- Rate movies:
  After selecting a movie, rate it using the slider (1-5)

- Manage movie list:
  1. Use the "Add to List" button to add a rated movie to your list
  2. Use the "Remove from List" button to remove a movie from your list
  3. Click "View My List" to see the list of movies you've rated
   
- Get recommendations:
  1. Content-based: recommends movies similar in genre to the one you've selected
  2. User-based (Based on List): recommends movies by analyzing your list of rated movies
  3. Collaborative filtering: enter a User ID to get recommendations based on that user's movie-watching history

- Recommendation Output:
  Recommended movies are displayed in a text box showing the title and genre of each suggestion

## Future Improvements
- Improve UI, probably make it web-based
- Add more sophisticated recommendation algorithms (e.g., hybrid models)
- Expand the dataset to include more metadata for better recommendations, as well as improve/optimise data collection to improve speed

## Contribution
Contributions are welcome! If you find any issues or want to propose improvements, feel free to fork the repository, make changes, and submit a pull request

## License
This project is licensed under the MIT License

## Acknowledgements
- MovieLens Dataset: provided by GroupLens for movie data and ratings
- scikit-learn: For machine learning algorithms and implementation of k-NN
- Tkinter: For building the graphical user interface (GUI)
