import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import Counter
import random

# Load dataset
movies = pd.read_csv("data/movies.csv")
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# 🎨 Genre Frequency Chart
genre_list = []
for genre in movies['genres']:
    genre_list.extend(genre.split())

genre_counts = Counter(genre_list)

# 🎨 Random bright colors
random.seed(42)
colors = ['#%06X' % random.randint(0x777777, 0xFFFFFF) for _ in genre_counts]

# Plot bar chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(genre_counts.keys(), genre_counts.values(), color=colors)
plt.xticks(rotation=45)
plt.title("🎬 Genre Frequency in Dataset")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.tight_layout()

# Show chart in Streamlit
st.subheader("📊 Genre Frequency Bar Chart")
st.pyplot(fig)

# TF-IDF and Cosine Similarity
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Recommendation function
def get_recommendations(title):
    try:
        idx = movies[movies['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        movie_indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[movie_indices].tolist()
    except IndexError:
        return []

# Streamlit UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Choose a movie:", movies['title'].tolist())

if st.button("Recommend"):
    recommendations = get_recommendations(selected_movie)
    if recommendations:
        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.write(f"- {movie}")
    else:
        st.warning("Movie not found in database.")
