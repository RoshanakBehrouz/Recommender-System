import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_loading import load_data, create_rating_matrix
from wmf_model import WeightedMatrixFactorization

LATENT_FEATURES = 2
W0 = 0.01
REGULARIZATION_LAMBDA = 0.1
ITERATIONS = 15
TOP_N_RECOMMENDATIONS = 10
NEW_USER_NUM_RATINGS = 10

plt.style.use('seaborn-v0_8-darkgrid')

st.set_page_config(page_title="Movie Recommender (WMF)", layout="wide")
st.title("🎬 Movie Recommender System (Weighted Matrix Factorization)")

st.markdown("""
This application demonstrates a movie recommender system using Weighted Matrix Factorization (WMF)
with the MovieLens 100k dataset. It utilizes the Weighted Alternating Least Squares (WALS)
optimization algorithm.
""")

ratings_df, movies_df = load_data()
all_user_ids = ratings_df['user_id'].unique()
all_movie_ids = ratings_df['movie_id'].unique()
user_to_idx = {user_id: idx for idx, user_id in enumerate(sorted(all_user_ids))}
movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(sorted(all_movie_ids))}
num_users = len(user_to_idx)
num_movies = len(movie_to_idx)

st.sidebar.header("Model Parameters")
st.sidebar.write(f"Number of users: {num_users}")
st.sidebar.write(f"Number of movies: {num_movies}")
st.sidebar.write(f"Latent features (K): {LATENT_FEATURES}")
st.sidebar.write(f"Weight for unobserved (w0): {W0}")
st.sidebar.write(f"Regularization (lambda): {REGULARIZATION_LAMBDA}")
st.sidebar.write(f"Iterations: {ITERATIONS}")

train_df, val_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

@st.cache_resource
def train_model(train_data, val_data, n_users, n_movies, u_map, m_map):
    st.write("Initializing and training WMF model...")
    model = WeightedMatrixFactorization(n_users, n_movies, LATENT_FEATURES, W0, REGULARIZATION_LAMBDA)
    model.fit(train_data, val_data, u_map, m_map, ITERATIONS)
    st.success("Model training complete!")
    return model

model = train_model(train_df, val_df, num_users, num_movies, user_to_idx, movie_to_idx)

st.header("Learning Curve (RMSE over Iterations)")
if model.train_rmse_history and model.val_rmse_history:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, ITERATIONS + 1), model.train_rmse_history, label='Train RMSE', marker='o')
    ax.plot(range(1, ITERATIONS + 1), model.val_rmse_history, label='Validation RMSE', marker='x')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")
    ax.set_title("WMF Training Progress")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.info("Learning curve data not available yet. Train the model first.")

st.header("Get Movie Recommendations")
tab1, tab2 = st.tabs(["Existing User", "New User (Cold Start)"])

with tab1:
    st.subheader("Recommendations for Existing Users")
    st.markdown("Enter a User ID from the dataset (e.g., 1 to 943) to get recommendations.")
    available_user_ids = sorted(ratings_df['user_id'].unique().tolist())
    selected_user_id = st.selectbox(
        "Select an Existing User ID:",
        options=available_user_ids,
        index=available_user_ids.index(1) if 1 in available_user_ids else 0,
        key="existing_user_select"
    )
    if st.button("Get Recommendations for Existing User"):
        if selected_user_id:
            with st.spinner(f"Generating recommendations for User ID {selected_user_id}..."):
                full_rating_matrix = create_rating_matrix(ratings_df)
                recommendations = model.recommend_movies(selected_user_id, user_to_idx, movie_to_idx, movies_df, full_rating_matrix, TOP_N_RECOMMENDATIONS)
                if not recommendations.empty:
                    st.success(f"Top {TOP_N_RECOMMENDATIONS} Recommendations for User ID {selected_user_id}:")
                    st.dataframe(recommendations[['movie_title', 'predicted_rating']].round(2).reset_index(drop=True))
                else:
                    st.info(f"Could not generate recommendations for User ID {selected_user_id}. They might have rated all movies or no valid movies to recommend.")
        else:
            st.warning("Please enter a valid User ID.")

with tab2:
    st.subheader("Recommendations for New Users (Cold Start)")
    st.markdown(f"""
    To get recommendations for a new user, please rate at least {NEW_USER_NUM_RATINGS} movies below.
    Your ratings will be used to infer your preferences.
    """)
    top_movies = ratings_df['movie_id'].value_counts().head(NEW_USER_NUM_RATINGS * 2).index.tolist()
    sample_movies_for_new_user = movies_df[movies_df['movie_id'].isin(top_movies)].sample(n=NEW_USER_NUM_RATINGS, random_state=42)
    new_user_ratings_input = {}
    st.write("Please rate the following movies (1-5 stars):")
    for index, row in sample_movies_for_new_user.iterrows():
        movie_id = row['movie_id']
        movie_title = row['movie_title']
        rating = st.slider(f"**{movie_title}**", 1, 5, 3, key=f"new_user_rating_{movie_id}")
        new_user_ratings_input[movie_id] = rating
    if st.button("Get Recommendations for New User"):
        if len(new_user_ratings_input) >= NEW_USER_NUM_RATINGS:
            with st.spinner("Inferring preferences and generating recommendations..."):
                new_user_embedding = model.infer_new_user_embedding(new_user_ratings_input, movie_to_idx)
                if new_user_embedding is not None:
                    rated_movie_ids_list = list(new_user_ratings_input.keys())
                    recommendations = model.recommend_for_new_user(new_user_embedding, movie_to_idx, movies_df, rated_movie_ids_list, TOP_N_RECOMMENDATIONS)
                    if not recommendations.empty:
                        st.success(f"Top {TOP_N_RECOMMENDATIONS} Recommendations for You:")
                        st.dataframe(recommendations[['movie_title', 'predicted_rating']].round(2).reset_index(drop=True))
                    else:
                        st.info("Could not generate recommendations based on your ratings.")
                else:
                    st.error("Failed to infer new user embedding. Please try again.")
        else:
            st.warning(f"Please rate at least {NEW_USER_NUM_RATINGS} movies to get recommendations.")
