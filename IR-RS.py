import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set Matplotlib style for better plots
plt.style.use('seaborn-v0_8-darkgrid')

# --- Configuration ---
# Define the URL for the MovieLens 100k dataset
MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
# Latent features (K) for matrix factorization
LATENT_FEATURES = 2
# Weight for unobserved ratings (w0 in the formula)
W0 = 0.01
# Regularization parameter (lambda) to prevent overfitting
REGULARIZATION_LAMBDA = 0.1
# Number of WALS iterations
ITERATIONS = 15
# Number of top recommendations to show
TOP_N_RECOMMENDATIONS = 10
# Number of movies to ask new users to rate
NEW_USER_NUM_RATINGS = 10

# --- Data Loading and Preprocessing ---

@st.cache_resource
def load_data():
    """
    Downloads and loads the MovieLens 100k dataset.
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Ratings data.
            - pd.DataFrame: Movies data.
    """
    st.write("Downloading MovieLens 100k dataset...")
    try:
        response = requests.get(MOVIELENS_URL)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))

        # Read u.data (ratings)
        with zip_file.open('ml-100k/u.data') as f:
            ratings_df = pd.read_csv(f, sep='\t', header=None,
                                     names=['user_id', 'movie_id', 'rating', 'timestamp'])

        # Read u.item (movie titles)
        with zip_file.open('ml-100k/u.item') as f:
            movies_df = pd.read_csv(f, sep='|', header=None, encoding='latin-1',
                                    names=['movie_id', 'movie_title', 'release_date', 'video_release_date',
                                           'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                           'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                           'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                           'Thriller', 'War', 'Western'])
            # Select only movie_id and movie_title
            movies_df = movies_df[['movie_id', 'movie_title']]

        st.success("Dataset loaded successfully!")
        return ratings_df, movies_df
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading dataset: {e}")
        st.stop()
    except zipfile.BadZipFile:
        st.error("Downloaded file is not a valid zip file.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        st.stop()


def create_rating_matrix(ratings_df):
    """
    Creates a user-item rating matrix.
    Args:
        ratings_df (pd.DataFrame): DataFrame containing user_id, movie_id, rating.
    Returns:
        pd.DataFrame: Pivot table of ratings.
    """
    # Create a pivot table for user-item ratings
    # Fill NaN with 0 for unrated movies for matrix operations, but remember they are unobserved
    rating_matrix = ratings_df.pivot_table(index='user_id', columns='movie_id', values='rating')
    return rating_matrix

# --- Weighted Matrix Factorization (WMF) Model ---

class WeightedMatrixFactorization:
    """
    Implements Weighted Matrix Factorization (WMF) using Weighted Alternating Least Squares (WALS).
    The objective function used is:
    sum_{(i,j) in Obs} (r_ij - u_i^T v_j)^2 + w_0 * sum_{(i,j) in Nobs} (u_i^T v_j)^2
    where r_ij is the rating, u_i is user embedding, v_j is item embedding,
    Obs are observed ratings, Nobs are non-observed ratings, and w_0 is a weight for non-observed.
    Regularization (lambda * I) is added to the inverse matrix for stability.
    """
    def __init__(self, num_users, num_movies, latent_features, w0, reg_lambda):
        self.num_users = num_users
        self.num_movies = num_movies
        self.latent_features = latent_features
        self.w0 = w0
        self.reg_lambda = reg_lambda

        # Initialize user and item embeddings randomly
        # Scale by sqrt(1/latent_features) to keep initial magnitudes reasonable
        self.user_factors = np.random.rand(self.num_users, self.latent_features) * np.sqrt(1/self.latent_features)
        self.item_factors = np.random.rand(self.num_movies, self.latent_features) * np.sqrt(1/self.latent_features)

        self.train_rmse_history = []
        self.val_rmse_history = []

    def _calculate_rmse(self, ratings_df, user_map, movie_map):
        """
        Calculates the Root Mean Squared Error (RMSE) for given ratings.
        Args:
            ratings_df (pd.DataFrame): DataFrame with 'user_id', 'movie_id', 'rating'.
            user_map (dict): Mapping from original user_id to internal index.
            movie_map (dict): Mapping from original movie_id to internal index.
        Returns:
            float: RMSE value.
        """
        predictions = []
        actuals = []
        for _, row in ratings_df.iterrows():
            user_idx = user_map.get(row['user_id'])
            movie_idx = movie_map.get(row['movie_id'])
            if user_idx is not None and movie_idx is not None:
                predicted_rating = np.dot(self.user_factors[user_idx], self.item_factors[movie_idx])
                predictions.append(predicted_rating)
                actuals.append(row['rating'])
        if not actuals:
            return float('inf') # Return infinity if no actuals to prevent division by zero
        return np.sqrt(mean_squared_error(actuals, predictions))

    def fit(self, train_df, val_df, user_map, movie_map, iterations=ITERATIONS):
        """
        Trains the WMF model using WALS.
        Args:
            train_df (pd.DataFrame): Training ratings.
            val_df (pd.DataFrame): Validation ratings.
            user_map (dict): Mapping from original user_id to internal index.
            movie_map (dict): Mapping from original movie_id to internal index.
            iterations (int): Number of WALS iterations.
        """
        st.write(f"Training WMF model for {iterations} iterations...")

        # Create inverse mappings for internal indices to original IDs
        self.idx_to_user_id = {v: k for k, v in user_map.items()}
        self.idx_to_movie_id = {v: k for k, v in movie_map.items()}

        # Create sparse rating matrices for efficient lookup of observed ratings
        # This will store (user_idx, movie_idx): rating
        self.train_ratings_dict = {}
        for _, row in train_df.iterrows():
            user_idx = user_map[row['user_id']]
            movie_idx = movie_map[row['movie_id']]
            if user_idx not in self.train_ratings_dict:
                self.train_ratings_dict[user_idx] = {}
            self.train_ratings_dict[user_idx][movie_idx] = row['rating']

        # Precompute I * lambda for regularization
        identity_matrix_k = np.eye(self.latent_features) * self.reg_lambda

        for iteration in range(iterations):
            st.write(f"Iteration {iteration + 1}/{iterations}...")

            # 1. Update User Factors (U)
            # Precompute V^T V for efficiency
            VtV = np.dot(self.item_factors.T, self.item_factors)

            for u_idx in range(self.num_users):
                # Get movies rated by current user in training set
                observed_movies_for_user = self.train_ratings_dict.get(u_idx, {})
                rated_movie_indices = list(observed_movies_for_user.keys())

                if not rated_movie_indices: # If user has no ratings in training set, skip update
                    continue

                # Sum for observed terms: sum_{j in Obs_i} v_j v_j^T
                sum_vj_vjT_obs = np.zeros((self.latent_features, self.latent_features))
                # Sum for B_i: sum_{j in Obs_i} r_ij v_j
                sum_rij_vj_obs = np.zeros(self.latent_features)

                for m_idx in rated_movie_indices:
                    vj = self.item_factors[m_idx]
                    rij = observed_movies_for_user[m_idx]
                    sum_vj_vjT_obs += np.outer(vj, vj) # vj * vj^T
                    sum_rij_vj_obs += rij * vj

                # A_i = (1 - w0) * sum_{j in Obs_i} v_j v_j^T + w0 * V^T V + lambda * I
                A_i = (1 - self.w0) * sum_vj_vjT_obs + self.w0 * VtV + identity_matrix_k
                B_i = sum_rij_vj_obs

                # Solve for u_i: A_i * u_i = B_i => u_i = A_i_inv * B_i
                try:
                    self.user_factors[u_idx] = np.linalg.solve(A_i, B_i)
                except np.linalg.LinAlgError:
                    st.warning(f"Singular matrix encountered for user {u_idx}. Skipping update.")
                    continue

            # 2. Update Item Factors (V)
            # Precompute U^T U for efficiency
            UtU = np.dot(self.user_factors.T, self.user_factors)

            # Transpose the train_ratings_dict for efficient item updates
            # This will store (movie_idx, user_idx): rating
            train_ratings_by_movie = {}
            for u_idx, movies_rated in self.train_ratings_dict.items():
                for m_idx, rating in movies_rated.items():
                    if m_idx not in train_ratings_by_movie:
                        train_ratings_by_movie[m_idx] = {}
                    train_ratings_by_movie[m_idx][u_idx] = rating

            for m_idx in range(self.num_movies):
                # Get users who rated current movie in training set
                observed_users_for_movie = train_ratings_by_movie.get(m_idx, {})
                rating_user_indices = list(observed_users_for_movie.keys())

                if not rating_user_indices: # If movie has no ratings in training set, skip update
                    continue

                # Sum for observed terms: sum_{i in Obs_j} u_i u_i^T
                sum_ui_uiT_obs = np.zeros((self.latent_features, self.latent_features))
                # Sum for B_j: sum_{i in Obs_j} r_ij u_i
                sum_rij_ui_obs = np.zeros(self.latent_features)

                for u_idx in rating_user_indices:
                    ui = self.user_factors[u_idx]
                    rij = observed_users_for_movie[u_idx]
                    sum_ui_uiT_obs += np.outer(ui, ui) # ui * ui^T
                    sum_rij_ui_obs += rij * ui

                # A_j = (1 - w0) * sum_{i in Obs_j} u_i u_i^T + w0 * U^T U + lambda * I
                A_j = (1 - self.w0) * sum_ui_uiT_obs + self.w0 * UtU + identity_matrix_k
                B_j = sum_rij_ui_obs

                # Solve for v_j: A_j * v_j = B_j => v_j = A_j_inv * B_j
                try:
                    self.item_factors[m_idx] = np.linalg.solve(A_j, B_j)
                except np.linalg.LinAlgError:
                    st.warning(f"Singular matrix encountered for movie {m_idx}. Skipping update.")
                    continue

            # Evaluate RMSE after each iteration
            train_rmse = self._calculate_rmse(train_df, user_map, movie_map)
            val_rmse = self._calculate_rmse(val_df, user_map, movie_map)
            self.train_rmse_history.append(train_rmse)
            self.val_rmse_history.append(val_rmse)
            st.write(f"  Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")

    def predict_rating(self, user_idx, movie_idx):
        """
        Predicts the rating for a given user and movie.
        Args:
            user_idx (int): Internal index of the user.
            movie_idx (int): Internal index of the movie.
        Returns:
            float: Predicted rating.
        """
        if user_idx < 0 or user_idx >= self.num_users or \
           movie_idx < 0 or movie_idx >= self.num_movies:
            return None # Invalid indices

        return np.dot(self.user_factors[user_idx], self.item_factors[movie_idx])

    def recommend_movies(self, user_id, user_map, movie_map, movies_df, rating_matrix, top_n=TOP_N_RECOMMENDATIONS):
        """
        Generates movie recommendations for an existing user.
        Args:
            user_id (int): Original user ID.
            user_map (dict): Mapping from original user_id to internal index.
            movie_map (dict): Mapping from original movie_id to internal index.
            movies_df (pd.DataFrame): DataFrame with movie titles.
            rating_matrix (pd.DataFrame): User-item rating matrix.
            top_n (int): Number of top recommendations to return.
        Returns:
            pd.DataFrame: Top N recommended movies with predicted ratings.
        """
        user_idx = user_map.get(user_id)
        if user_idx is None:
            return pd.DataFrame() # User not found

        # Get movies already rated by the user
        rated_movies = rating_matrix.loc[user_id].dropna().index.tolist()
        rated_movie_indices = [movie_map[mid] for mid in rated_movies if mid in movie_map]

        predictions = []
        for movie_orig_id, movie_idx in movie_map.items():
            if movie_orig_id not in rated_movies: # Only recommend unrated movies
                predicted_rating = self.predict_rating(user_idx, movie_idx)
                predictions.append({'movie_id': movie_orig_id, 'predicted_rating': predicted_rating})

        predictions_df = pd.DataFrame(predictions)
        if predictions_df.empty:
            return pd.DataFrame()

        # Merge with movie titles and sort by predicted rating
        recommended_movies = predictions_df.merge(movies_df, on='movie_id')
        recommended_movies = recommended_movies.sort_values(by='predicted_rating', ascending=False)

        return recommended_movies.head(top_n)

    def infer_new_user_embedding(self, new_user_ratings, movie_map):
        """
        Infers an embedding for a new user based on their initial ratings.
        Args:
            new_user_ratings (dict): Dictionary of {movie_id: rating} for the new user.
            movie_map (dict): Mapping from original movie_id to internal index.
        Returns:
            np.array: Inferred user embedding.
        """
        st.write("Inferring new user embedding...")
        # Precompute V^T V for efficiency
        VtV = np.dot(self.item_factors.T, self.item_factors)
        identity_matrix_k = np.eye(self.latent_features) * self.reg_lambda

        sum_vj_vjT_obs = np.zeros((self.latent_features, self.latent_features))
        sum_rij_vj_obs = np.zeros(self.latent_features)

        rated_movie_indices = []
        for movie_orig_id, rating in new_user_ratings.items():
            movie_idx = movie_map.get(movie_orig_id)
            if movie_idx is not None:
                vj = self.item_factors[movie_idx]
                sum_vj_vjT_obs += np.outer(vj, vj)
                sum_rij_vj_obs += rating * vj
                rated_movie_indices.append(movie_idx)

        if not rated_movie_indices:
            st.warning("No valid ratings provided for new user. Cannot infer embedding.")
            return None

        # A_new_user = (1 - w0) * sum_{j in Obs_new_user} v_j v_j^T + w0 * V^T V + lambda * I
        A_new_user = (1 - self.w0) * sum_vj_vjT_obs + self.w0 * VtV + identity_matrix_k
        B_new_user = sum_rij_vj_obs

        try:
            new_user_embedding = np.linalg.solve(A_new_user, B_new_user)
            st.success("New user embedding inferred.")
            return new_user_embedding
        except np.linalg.LinAlgError:
            st.error("Singular matrix encountered while inferring new user embedding. Please try different ratings.")
            return None

    def recommend_for_new_user(self, new_user_embedding, movie_map, movies_df, rated_movie_ids, top_n=TOP_N_RECOMMENDATIONS):
        """
        Generates movie recommendations for a new user based on their inferred embedding.
        Args:
            new_user_embedding (np.array): Inferred embedding for the new user.
            movie_map (dict): Mapping from original movie_id to internal index.
            movies_df (pd.DataFrame): DataFrame with movie titles.
            rated_movie_ids (list): List of movie IDs already rated by the new user.
            top_n (int): Number of top recommendations to return.
        Returns:
            pd.DataFrame: Top N recommended movies with predicted ratings.
        """
        if new_user_embedding is None:
            return pd.DataFrame()

        predictions = []
        for movie_orig_id, movie_idx in movie_map.items():
            if movie_orig_id not in rated_movie_ids: # Only recommend unrated movies
                predicted_rating = np.dot(new_user_embedding, self.item_factors[movie_idx])
                predictions.append({'movie_id': movie_orig_id, 'predicted_rating': predicted_rating})

        predictions_df = pd.DataFrame(predictions)
        if predictions_df.empty:
            return pd.DataFrame()

        # Merge with movie titles and sort by predicted rating
        recommended_movies = predictions_df.merge(movies_df, on='movie_id')
        recommended_movies = recommended_movies.sort_values(by='predicted_rating', ascending=False)

        return recommended_movies.head(top_n)


# --- Streamlit Application ---

st.set_page_config(page_title="Movie Recommender (WMF)", layout="wide")

st.title("🎬 Movie Recommender System (Weighted Matrix Factorization)")
st.markdown("""
This application demonstrates a movie recommender system using Weighted Matrix Factorization (WMF)
with the MovieLens 100k dataset. It utilizes the Weighted Alternating Least Squares (WALS)
optimization algorithm.
""")

# Load data
ratings_df, movies_df = load_data()

# Prepare data for model training
all_user_ids = ratings_df['user_id'].unique()
all_movie_ids = ratings_df['movie_id'].unique()

# Create mappings from original IDs to contiguous internal indices
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

# Split data into training and validation sets
train_df, val_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

@st.cache_resource
def train_model(train_data, val_data, n_users, n_movies, u_map, m_map):
    """
    Trains the WMF model and returns the trained model and RMSE history.
    """
    st.write("Initializing and training WMF model...")
    model = WeightedMatrixFactorization(n_users, n_movies, LATENT_FEATURES, W0, REGULARIZATION_LAMBDA)
    model.fit(train_data, val_data, u_map, m_map, ITERATIONS)
    st.success("Model training complete!")
    return model

# Train the model (this will be cached)
model = train_model(train_df, val_df, num_users, num_movies, user_to_idx, movie_to_idx)

# --- Display Learning Curve ---
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

# --- Recommendation Section ---
st.header("Get Movie Recommendations")

tab1, tab2 = st.tabs(["Existing User", "New User (Cold Start)"])

with tab1:
    st.subheader("Recommendations for Existing Users")
    st.markdown("Enter a User ID from the dataset (e.g., 1 to 943) to get recommendations.")
    
    # Get a list of available user IDs for the selectbox
    available_user_ids = sorted(ratings_df['user_id'].unique().tolist())
    
    selected_user_id = st.selectbox(
        "Select an Existing User ID:",
        options=available_user_ids,
        index=available_user_ids.index(1) if 1 in available_user_ids else 0, # Default to user 1
        key="existing_user_select"
    )

    if st.button("Get Recommendations for Existing User"):
        if selected_user_id:
            with st.spinner(f"Generating recommendations for User ID {selected_user_id}..."):
                # Create the full rating matrix for checking rated movies
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

    # Get a sample of popular movies for new user to rate
    # Get movies with most ratings and pick a diverse set
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



