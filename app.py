import streamlit as st
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}

/* Title */
.title {
    text-align: center;
    color: #E50914;
    font-size: 50px;
    font-weight: bold;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #bbbbbb;
}

/* Button */
.stButton>button {
    background-color: #E50914;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

/* Movie card */
.movie-card {
    background-color: #181818;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

movies = pickle.load(open('movies.pkl', 'rb'))

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=ae5d858b83a2085e55c3904041bb9aad&language=en-US"
    data = requests.get(url).json()

    if data.get('poster_path') is None:
        return "https://via.placeholder.com/500x750?text=No+Image"

    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

def recommend(movie):
    if movie not in movies['title'].values:
        return [], []

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]

    names = []
    posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        names.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))

    return names, posters

st.markdown('<div class="title">🎬 Movie Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find movies you’ll love instantly 🍿</div>', unsafe_allow_html=True)

st.write("")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    selected_movie = st.selectbox("Search Movie", movies['title'].values)

st.write("")

col1, col2, col3 = st.columns([1,1,1])
with col2:
    clicked = st.button("🎯 Recommend")

if clicked:
    with st.spinner("Fetching recommendations... 🍿"):
        names, posters = recommend(selected_movie)

        if len(names) == 0:
            st.warning("Movie not found!")
        else:
            st.markdown("---")
            st.subheader("✨ Recommended for you")

            cols = st.columns(5)

            for i in range(5):
                with cols[i]:
                    st.image(posters[i])
                    st.markdown(
                        f"<div class='movie-card'><b>{names[i]}</b></div>",
                        unsafe_allow_html=True
                    )