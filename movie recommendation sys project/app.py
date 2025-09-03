import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import requests

st.set_page_config(page_title='Movie Recommender', layout='wide')

@st.cache_data
def load_data():
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    movies['genres'] = movies['genres'].fillna('')
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ')
    return movies, ratings

@st.cache_data
def build_models(movies, ratings, n_components=50):
    vectorizer = TfidfVectorizer(stop_words='english')
    genre_tfidf = vectorizer.fit_transform(movies['genres_clean'])
    content_sim = cosine_similarity(genre_tfidf, genre_tfidf)

    user_item = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    item_factors = svd.fit_transform(user_item.T)

    item_movieIds = user_item.columns.tolist()
    movieid_to_colidx = {mid: i for i, mid in enumerate(item_movieIds)}
    item_sim_latent = cosine_similarity(item_factors)

    return vectorizer, genre_tfidf, content_sim, item_sim_latent, item_movieIds, movieid_to_colidx


movies, ratings = load_data()
vectorizer, genre_tfidf, content_sim, item_sim_latent, item_movieIds, movieid_to_colidx = build_models(movies, ratings)
movie_idx = pd.Series(movies.index, index=movies['movieId'])

st.title('🎬 Movie Recommendation System')
st.sidebar.header('Preferences')

method = st.sidebar.selectbox('Recommendation method', ['Hybrid (CF + Content)', 'Content-based', 'Collaborative (latent)'])
fav_input = st.sidebar.text_input('Enter favourite movies (separate multiple with ; )')
fav_list = [x.strip() for x in fav_input.split(';') if x.strip()]

all_genres = sorted({g for sub in movies['genres'].str.split('|').dropna() for g in sub if g})
sel_genres = st.sidebar.multiselect('Preferred genres (optional)', all_genres)
num_rec = st.sidebar.slider('Number of recommendations', 1, 20, 5)

# Optional TMDb
st.sidebar.markdown('---')
use_tmdb = st.sidebar.checkbox('Show movie posters (requires TMDb API key)')
tmdb_key = ''
if use_tmdb:
    tmdb_key = st.sidebar.text_input('TMDb API Key (optional)', type='password')

# resolve favourites to movieIds (first match)
fav_movieIds = []
for title in fav_list:
    matches = movies[movies['title'].str.contains(title, case=False, regex=False)]
    if not matches.empty:
        fav_movieIds.append(int(matches.iloc[0]['movieId']))


def recommend_by_genres_app(genres_list, top_n=5):
    if not genres_list:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])
    q = ' '.join(genres_list)
    qvec = vectorizer.transform([q])
    sim = cosine_similarity(qvec, genre_tfidf).flatten()
    idx = sim.argsort()[-top_n:][::-1]
    return movies.iloc[idx][['movieId', 'title', 'genres']]


def recommend_hybrid_app(fav_ids, top_n=5):
    scores = np.zeros(item_sim_latent.shape[0])
    for mid in fav_ids:
        if mid in item_movieIds:
            scores += item_sim_latent[item_movieIds.index(mid)]
    for mid in fav_ids:
        if mid in item_movieIds:
            scores[item_movieIds.index(mid)] = -np.inf
    top_cols = np.argsort(scores)[-top_n:][::-1]
    recs = [item_movieIds[c] for c in top_cols]
    return movies[movies['movieId'].isin(recs)][['movieId', 'title', 'genres']]


# Poster fetcher
def fetch_poster(title, year=None):
    if not (use_tmdb and tmdb_key):
        return None
    try:
        params = {'api_key': tmdb_key, 'query': title, 'include_adult': False}
        r = requests.get('https://api.themoviedb.org/3/search/movie', params=params, timeout=5)
        data = r.json()
        if data.get('results'):
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return 'https://image.tmdb.org/t/p/w200' + poster_path
    except Exception:
        return None
    return None


st.write('### Your favourites (parsed)')
if fav_movieIds:
    st.write(movies[movies['movieId'].isin(fav_movieIds)][['title', 'genres']])
else:
    st.write('No favourite movies matched. Use titles like: Toy Story, The Matrix, etc.')


if st.button('Get Recommendations'):
    if method == 'Content-based':
        if fav_movieIds:
            mid = fav_movieIds[0]
            if mid in movie_idx:
                idx = movie_idx[mid]
                sim_scores = list(enumerate(content_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                top_indices = [i for i, score in sim_scores[1:num_rec+1]]
                recs = movies.iloc[top_indices][['title', 'genres']]
            else:
                recs = pd.DataFrame()
        elif sel_genres:
            recs = recommend_by_genres_app(sel_genres, top_n=num_rec)
        else:
            recs = pd.DataFrame()

    elif method == 'Collaborative (latent)':
        if fav_movieIds:
            recs = recommend_hybrid_app(fav_movieIds, top_n=num_rec)
        else:
            st.warning('Provide at least one favourite movie for collaborative recommendations')
            recs = pd.DataFrame()

    else:  # Hybrid
        rec_cf = recommend_hybrid_app(fav_movieIds, top_n=num_rec*2) if fav_movieIds else pd.DataFrame()
        rec_content = recommend_by_genres_app(sel_genres, top_n=num_rec*2) if sel_genres else pd.DataFrame()
        if not rec_cf.empty and not rec_content.empty:
            merged = pd.concat([rec_cf, rec_content]).drop_duplicates('movieId')
            recs = merged.head(num_rec)[['title', 'genres']]
        elif not rec_cf.empty:
            recs = rec_cf.head(num_rec)[['title', 'genres']]
        elif not rec_content.empty:
            recs = rec_content.head(num_rec)[['title', 'genres']]
        else:
            recs = pd.DataFrame()

    if recs.empty:
        st.write('No recommendations found. Try different inputs.')
    else:
        st.write('### Top recommendations')

        if use_tmdb and tmdb_key:
            for _, row in recs.reset_index(drop=True).iterrows():
                poster = fetch_poster(row['title'])
                c1, c2 = st.columns([1, 4])
                if poster:
                    with c1:
                        st.image(poster, width=120)
                with c2:
                    st.markdown(
                        f"**{row['title']}**  \n"
                        f"Genres: {row['genres']}"
                    )
                st.markdown('---')
        else:
            st.table(recs.reset_index(drop=True))
