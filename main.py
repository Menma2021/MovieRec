import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.neighbors import NearestNeighbors
from recommendation import recommend_movies_by_genre, recommend_movies, recommend_from_user_list, movies, \
    user_movie_sparse, movie_ids


class MovieRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommendation System")
        self.root.geometry("600x600")

        # Sparse matrix and movie IDs
        self.user_movie_sparse = user_movie_sparse
        self.movie_ids = movie_ids
        self.movies = movies
        self.model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
        self.model_knn.fit(self.user_movie_sparse)

        self.create_widgets()
        self.movie_list = {}

    def create_widgets(self):
        # Search bar
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(self.root, textvariable=self.search_var, width=40)
        self.search_entry.grid(row=0, column=0, padx=10, pady=10)
        self.search_button = ttk.Button(self.root, text="Search", command=self.search_movie)
        self.search_button.grid(row=0, column=1, padx=10, pady=10)

        # Autocomplete
        self.movie_listbox = tk.Listbox(self.root, height=5)
        self.movie_listbox.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        self.movie_listbox.bind("<<ListboxSelect>>", self.on_movie_select)

        # Rating slider
        self.rating_var = tk.IntVar(value=3)
        self.rating_slider = ttk.Scale(self.root, from_=1, to=5, orient="horizontal", variable=self.rating_var)
        self.rating_slider.grid(row=2, column=0, padx=10, pady=10)
        self.rating_label = ttk.Label(self.root, text="Rating (1-5)")
        self.rating_label.grid(row=2, column=1, padx=10, pady=10)

        # Add to list button
        self.add_button = ttk.Button(self.root, text="Add to List", command=self.add_movie_to_list)
        self.add_button.grid(row=3, column=0, padx=10, pady=10)

        # Remove from list button
        self.remove_button = ttk.Button(self.root, text="Remove from List", command=self.remove_movie_from_list)
        self.remove_button.grid(row=3, column=1, padx=10, pady=10)

        # View list button
        self.view_list_button = ttk.Button(self.root, text="View My List", command=self.view_movie_list)
        self.view_list_button.grid(row=4, column=0, padx=10, pady=10)

        # Recommendation button
        self.rec_button = ttk.Button(self.root, text="Recommend (Content)", command=self.recommend_content_based)
        self.rec_button.grid(row=5, column=0, padx=10, pady=10)

        self.rec_user_button = ttk.Button(self.root, text="Recommend (Based on List)",
                                          command=self.recommend_user_based)
        self.rec_user_button.grid(row=5, column=1, padx=10, pady=10)

        # Collaborative filtering
        self.user_id_var = tk.StringVar()
        self.user_id_entry = ttk.Entry(self.root, textvariable=self.user_id_var, width=10)
        self.user_id_entry.grid(row=6, column=0, padx=10, pady=10)
        self.rec_collab_button = ttk.Button(self.root, text="Recommend (Collaborative)",
                                            command=self.recommend_collaborative_based)
        self.rec_collab_button.grid(row=6, column=1, padx=10, pady=10)

        # Recommendation output
        self.output_box = tk.Text(self.root, height=10, width=70)
        self.output_box.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

    def search_movie(self):
        # Autocomplete
        search_term = self.search_var.get().lower()
        matches = movies[movies['title'].str.lower().str.contains(search_term)]['title'].head(10)
        self.movie_listbox.delete(0, tk.END)
        for movie in matches:
            self.movie_listbox.insert(tk.END, movie)

    def on_movie_select(self, event):
        # Movie selected from listbox
        selected_movie = self.movie_listbox.get(tk.ACTIVE)
        self.search_var.set(selected_movie)

    def add_movie_to_list(self):
        # Adding movie and rating to the user's list
        movie_name = self.search_var.get()
        rating = self.rating_var.get()
        if movie_name:
            self.movie_list[movie_name] = rating
            messagebox.showinfo("Success", f"Added '{movie_name}' with rating {rating}")
        else:
            messagebox.showwarning("Warning", "No movie selected")

    def remove_movie_from_list(self):
        # Removing movie from the user's list
        movie_name = self.search_var.get()
        if movie_name in self.movie_list:
            del self.movie_list[movie_name]
            messagebox.showinfo("Success", f"Removed '{movie_name}' from your list")
        else:
            messagebox.showwarning("Warning", "Movie not in your list")

    def view_movie_list(self):
        # Viewing user's movie list
        self.output_box.delete(1.0, tk.END)
        if not self.movie_list:
            self.output_box.insert(tk.END, "Your movie list is empty.")
        else:
            self.output_box.insert(tk.END, "Your movie list:\n")
            for movie, rating in self.movie_list.items():
                self.output_box.insert(tk.END, f"{movie}: {rating}/5\n")

    def recommend_content_based(self):
        # Content-based filtering
        movie_name = self.search_var.get()
        if movie_name:
            recommendations = recommend_movies_by_genre(movie_name)
            self.show_recommendations(recommendations)
        else:
            messagebox.showwarning("Warning", "No movie selected")

    def recommend_user_based(self):
        # Get user-rated movies
        user_rated_movies = list(self.movie_list.keys())

        # Call the recommendation function with model_knn
        recommendations = recommend_from_user_list(
            user_rated_movies,
            self.user_movie_sparse,
            movie_ids,
            self.movies,
            self.model_knn,
            num_recommendations=5
        )

        # Display recommendations
        if not recommendations.empty:
            self.show_recommendations(recommendations)
        else:
            self.display_message("No recommendations available.")

    def recommend_collaborative_based(self):
        # ID-specific collaborative filtering
        user_id = self.user_id_var.get()
        if user_id.isdigit():
            recommendations = recommend_movies(int(user_id))
            self.show_recommendations(recommendations)
        else:
            messagebox.showwarning("Warning", "Enter a valid User ID")

    def show_recommendations(self, recommendations):
        self.output_box.delete(1.0, tk.END)
        if recommendations.empty:
            self.output_box.insert(tk.END, "No recommendations found.")
        else:
            self.output_box.insert(tk.END, "Recommended Movies:\n")
            for index, row in recommendations.iterrows():
                self.output_box.insert(tk.END, f"{row['title']} - {row['genres']}\n")


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommenderApp(root)
    root.mainloop()
