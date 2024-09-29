# main_app.py
import tkinter as tk
from tkinter import ttk, messagebox
from recommendation import recommend_movies_by_genre, recommend_movies, movies


class MovieRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommendation System")
        self.root.geometry("600x400")

        # UI components
        self.create_widgets()

        # User's movie list and ratings
        self.movie_list = {}

    def create_widgets(self):
        # Search bar
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(self.root, textvariable=self.search_var, width=40)
        self.search_entry.grid(row=0, column=0, padx=10, pady=10)
        self.search_button = ttk.Button(self.root, text="Search", command=self.search_movie)
        self.search_button.grid(row=0, column=1, padx=10, pady=10)

        # Autocomplete dropdown
        self.movie_listbox = tk.Listbox(self.root, height=5)
        self.movie_listbox.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        self.movie_listbox.bind("<<ListboxSelect>>", self.on_movie_select)

        # Slider
        self.rating_var = tk.IntVar(value=3)
        self.rating_slider = ttk.Scale(self.root, from_=1, to=5, orient="horizontal", variable=self.rating_var)
        self.rating_slider.grid(row=2, column=0, padx=10, pady=10)
        self.rating_label = ttk.Label(self.root, text="Rating (1-5)")
        self.rating_label.grid(row=2, column=1, padx=10, pady=10)

        # Adding to list
        self.add_button = ttk.Button(self.root, text="Add to List", command=self.add_movie_to_list)
        self.add_button.grid(row=3, column=0, padx=10, pady=10)

        # View list
        self.view_list_button = ttk.Button(self.root, text="View My List", command=self.view_movie_list)
        self.view_list_button.grid(row=3, column=1, padx=10, pady=10)

        # Recommendation buttons
        self.rec_button = ttk.Button(self.root, text="Recommend (Content)", command=self.recommend_content_based)
        self.rec_button.grid(row=4, column=0, padx=10, pady=10)
        self.rec_collab_button = ttk.Button(self.root, text="Recommend (Collaborative)",
                                            command=self.recommend_collaborative_based)
        self.rec_collab_button.grid(row=4, column=1, padx=10, pady=10)

        # Output
        self.output_box = tk.Text(self.root, height=10, width=70)
        self.output_box.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

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

    def view_movie_list(self):
        # Show the user's movie list
        movie_list_str = "\n".join([f"{movie}: {rating}" for movie, rating in self.movie_list.items()])
        self.output_box.delete(1.0, tk.END)
        self.output_box.insert(tk.END, "My Movie List:\n")
        self.output_box.insert(tk.END, movie_list_str)

    def recommend_content_based(self):
        # Recommend based on a selected movie's genre
        movie_name = self.search_var.get()
        if movie_name:
            recommendations = recommend_movies_by_genre(movie_name)
            self.output_box.delete(1.0, tk.END)
            self.output_box.insert(tk.END, f"Recommendations based on '{movie_name}':\n")
            for _, row in recommendations.iterrows():
                self.output_box.insert(tk.END, f"{row['title']} - {row['genres']}\n")
        else:
            messagebox.showwarning("Warning", "No movie selected")

    def recommend_collaborative_based(self):
        # Recommend movies for a user (ID = 1 for now)
        recommendations = recommend_movies(user_id=1)
        self.output_box.delete(1.0, tk.END)
        self.output_box.insert(tk.END, "Collaborative Recommendations:\n")
        for _, row in recommendations.iterrows():
            self.output_box.insert(tk.END, f"{row['title']} - {row['genres']}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommenderApp(root)
    root.mainloop()
