import streamlit as st
import pandas as pd
import requests
import zipfile
import io

MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

@st.cache_resource
def load_data():
    st.write("Downloading MovieLens 100k dataset...")
    try:
        response = requests.get(MOVIELENS_URL)
        response.raise_for_status()
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        with zip_file.open('ml-100k/u.data') as f:
            ratings_df = pd.read_csv(f, sep='\t', header=None,
                                     names=['user_id', 'movie_id', 'rating', 'timestamp'])
        with zip_file.open('ml-100k/u.item') as f:
            movies_df = pd.read_csv(f, sep='|', header=None, encoding='latin-1',
                                    names=['movie_id', 'movie_title', 'release_date', 'video_release_date',
                                           'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                           'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                           'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                           'Thriller', 'War', 'Western'])
            movies_df = movies_df[['movie_id', 'movie_title']]
        st.success("Dataset loaded successfully!")
        return ratings_df, movies_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

def create_rating_matrix(ratings_df):
    rating_matrix = ratings_df.pivot_table(index='user_id', columns='movie_id', values='rating')
    return rating_matrix
