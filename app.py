# FINAL FIXED VERSION

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("data/movies.csv")
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# 🎯 Genre frequency chart
import matplotlib.pyplot as plt
from collections import Counter

# Count genre frequency
genre_list = []
for genre in movies['genres']:
    genre_list.extend(genre.split())

genre_counts = Counter(genre_list)

# Plot using matplotlib
fig, ax = plt.subplots()
ax.bar(genre_counts.keys(), genre_counts.values(), color='skyblue')
plt.xticks(rotation=45)
plt.title("Genre Frequency")

# Show in Streamlit
st.subheader("🎯 Genre Frequency in Dataset")
st.pyplot(fig)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])

cosine_sim = cosine_similarity(tfidf_matrix)

def get_recommendations(title):
    try:
        idx = movies[movies['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        movie_indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[movie_indices].tolist()
    except IndexError:
        return []

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
