import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV from 'data' folder
movies = pd.read_csv("data/movies.csv", quotechar='"', encoding='utf-8', engine='python')
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# Extract year from title
movies['year'] = movies['title'].apply(lambda x: re.findall(r'\((\d{4})\)', x))
movies['year'] = movies['year'].apply(lambda x: int(x[0]) if x else None)

# TF-IDF + Cosine Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation Function
def get_recommendations(title, genre_filter=None, from_year=None, to_year=None):
    # Match partial title
    matches = [t for t in indices.index if title.lower() in t.lower()]
    if not matches:
        return ["Movie not found."]
    
    idx = indices[matches[0]]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
    recs = []
    for i, score in sim_scores:
        movie = movies.iloc[i]
        if genre_filter and genre_filter.lower() not in movie['genres'].lower():
            continue
        if from_year and to_year:
            if movie['year'] is None or not (from_year <= movie['year'] <= to_year):
                continue
        recs.append(movie['title'])
        if len(recs) == 5:
            break
    return recs

# Streamlit Interface
st.title("🎬 Movie Recommendation System")
st.write("Get personalized movie recommendations based on genre and release year!")

# Movie input
movie_input = st.text_input("Enter a movie title (partial is okay):")

# Genre dropdown
genre_options = sorted(set(g for row in movies['genres'] for g in row.split()))
genre_filter = st.selectbox("Filter by genre (optional):", [""] + genre_options)

# Year range dropdown
year_list = sorted(movies['year'].dropna().unique())
from_year = st.selectbox("From Year (optional):", [""] + [str(y) for y in year_list])
to_year = st.selectbox("To Year (optional):", [""] + [str(y) for y in year_list])

# Convert year input to int
from_y = float(from_year) if from_year else None
to_y = float(to_year) if to_year else None

# Button
if st.button("Get Recommendations"):
    if movie_input:
        results = get_recommendations(
            movie_input,
            genre_filter if genre_filter else None,
            from_y,
            to_y
        )
        if results:
            for i, title in enumerate(results, 1):
                st.write(f"{i}. {title}")
        else:
            st.write("No matching recommendations found.")
    else:
        st.warning("Please enter a movie name.")
