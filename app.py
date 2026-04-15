import streamlit as st
import pickle
import pandas as pd
import requests

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
.main-title {
    text-align: center;
    color: #E50914;
    font-size: 55px;
    font-weight: bold;
}
.sub-text {
    text-align: center;
    font-size: 20px;
    color: #bbbbbb;
}
.stButton>button {
    background-color: #E50914;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
.movie-card {
    background-color: #181818;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

movies = pickle.load(open('movies.pkl','rb'))   
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

def fetch_poster(movie_name):
    api_key = "d469d013fcefa9da1cf7213d4896dbd0"
    
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_name}"
        data = requests.get(url).json()

        if len(data['results']) == 0:
            return "https://via.placeholder.com/500x750?text=No+Image"

        movie_id = data['results'][0]['id']

        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
        data = requests.get(url).json()

        if data.get('poster_path'):
            return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"
    except:
        return "https://via.placeholder.com/500x750?text=Error"

def recommend(movie):
    movie = movie.lower()

    if movie not in movies['title'].str.lower().values:
        return ["Movie not found"], ["https://via.placeholder.com/500x750"]*5

    idx = movies[movies['title'].str.lower() == movie].index[0]
    similarity_scores = similarity[idx]

    movies_list = sorted(list(enumerate(similarity_scores)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]

    names, posters = [], []

    for i in movies_list:
        title = movies.iloc[i[0]].title
        names.append(title)
        posters.append(fetch_poster(title))

    return names, posters

def recommend_by_genre(genre):
    genre = genre.lower()

    filtered = movies[movies['genres'].str.contains(genre, case=False, na=False)]

    if filtered.empty:
        return ["No movies found"], ["https://via.placeholder.com/500x750"]*5

    names, posters = [], []

    for i in filtered.head(5).index:
        title = movies.iloc[i].title
        names.append(title)
        posters.append(fetch_poster(title))

    return names, posters

st.markdown('<div class="main-title">🎬 Movie Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Find movies you’ll love instantly 🍿</div>', unsafe_allow_html=True)

st.write("")

col1, col2, col3 = st.columns([1,2,1])

with col2:
    selected_movie = st.selectbox(
        "Search Movie (Dropdown)",
        movies['title'].values
    )

    search_query = st.text_input("Or type genre (e.g. action, comedy, romance)")

st.caption("💡 Tip: Use dropdown for movies or type genre for category search")

st.write("")

col1, col2, col3 = st.columns([1,1,1])

with col2:
    recommend_btn = st.button("🎯 Recommend")

if recommend_btn:
    with st.spinner("Fetching recommendations... 🍿"):

        if search_query.strip() != "":
            names, posters = recommend_by_genre(search_query)
        else:
            names, posters = recommend(selected_movie)

        st.markdown("---")
        st.subheader("✨ Recommended for you")

        cols = st.columns(5)

        for i in range(len(names)):
            with cols[i]:
                st.image(posters[i])
                st.markdown(f"<div class='movie-card'><b>{names[i]}</b></div>", unsafe_allow_html=True)