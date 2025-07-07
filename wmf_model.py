import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_squared_error

class WeightedMatrixFactorization:
    def __init__(self, num_users, num_movies, latent_features, w0, reg_lambda):
        self.num_users = num_users
        self.num_movies = num_movies
        self.latent_features = latent_features
        self.w0 = w0
        self.reg_lambda = reg_lambda
        self.user_factors = np.random.rand(self.num_users, self.latent_features) * np.sqrt(1/self.latent_features)
        self.item_factors = np.random.rand(self.num_movies, self.latent_features) * np.sqrt(1/self.latent_features)
        self.train_rmse_history = []
        self.val_rmse_history = []

    def _calculate_rmse(self, ratings_df, user_map, movie_map):
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
            return float('inf')
        return np.sqrt(mean_squared_error(actuals, predictions))

    def fit(self, train_df, val_df, user_map, movie_map, iterations):
        st.write(f"Training WMF model for {iterations} iterations...")
        self.idx_to_user_id = {v: k for k, v in user_map.items()}
        self.idx_to_movie_id = {v: k for k, v in movie_map.items()}
        self.train_ratings_dict = {}
        for _, row in train_df.iterrows():
            user_idx = user_map[row['user_id']]
            movie_idx = movie_map[row['movie_id']]
            if user_idx not in self.train_ratings_dict:
                self.train_ratings_dict[user_idx] = {}
            self.train_ratings_dict[user_idx][movie_idx] = row['rating']
        identity_matrix_k = np.eye(self.latent_features) * self.reg_lambda
        for iteration in range(iterations):
            st.write(f"Iteration {iteration + 1}/{iterations}...")
            VtV = np.dot(self.item_factors.T, self.item_factors)
            for u_idx in range(self.num_users):
                observed_movies_for_user = self.train_ratings_dict.get(u_idx, {})
                rated_movie_indices = list(observed_movies_for_user.keys())
                if not rated_movie_indices:
                    continue
                sum_vj_vjT_obs = np.zeros((self.latent_features, self.latent_features))
                sum_rij_vj_obs = np.zeros(self.latent_features)
                for m_idx in rated_movie_indices:
                    vj = self.item_factors[m_idx]
                    rij = observed_movies_for_user[m_idx]
                    sum_vj_vjT_obs += np.outer(vj, vj)
                    sum_rij_vj_obs += rij * vj
                A_i = (1 - self.w0) * sum_vj_vjT_obs + self.w0 * VtV + identity_matrix_k
                B_i = sum_rij_vj_obs
                try:
                    self.user_factors[u_idx] = np.linalg.solve(A_i, B_i)
                except np.linalg.LinAlgError:
                    st.warning(f"Singular matrix encountered for user {u_idx}. Skipping update.")
                    continue
            UtU = np.dot(self.user_factors.T, self.user_factors)
            train_ratings_by_movie = {}
            for u_idx, movies_rated in self.train_ratings_dict.items():
                for m_idx, rating in movies_rated.items():
                    if m_idx not in train_ratings_by_movie:
                        train_ratings_by_movie[m_idx] = {}
                    train_ratings_by_movie[m_idx][u_idx] = rating
            for m_idx in range(self.num_movies):
                observed_users_for_movie = train_ratings_by_movie.get(m_idx, {})
                rating_user_indices = list(observed_users_for_movie.keys())
                if not rating_user_indices:
                    continue
                sum_ui_uiT_obs = np.zeros((self.latent_features, self.latent_features))
                sum_rij_ui_obs = np.zeros(self.latent_features)
                for u_idx in rating_user_indices:
                    ui = self.user_factors[u_idx]
                    rij = observed_users_for_movie[u_idx]
                    sum_ui_uiT_obs += np.outer(ui, ui)
                    sum_rij_ui_obs += rij * ui
                A_j = (1 - self.w0) * sum_ui_uiT_obs + self.w0 * UtU + identity_matrix_k
                B_j = sum_rij_ui_obs
                try:
                    self.item_factors[m_idx] = np.linalg.solve(A_j, B_j)
                except np.linalg.LinAlgError:
                    st.warning(f"Singular matrix encountered for movie {m_idx}. Skipping update.")
                    continue
            train_rmse = self._calculate_rmse(train_df, user_map, movie_map)
            val_rmse = self._calculate_rmse(val_df, user_map, movie_map)
            self.train_rmse_history.append(train_rmse)
            self.val_rmse_history.append(val_rmse)
            st.write(f"  Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")

    def predict_rating(self, user_idx, movie_idx):
        if user_idx < 0 or user_idx >= self.num_users or \
           movie_idx < 0 or movie_idx >= self.num_movies:
            return None
        return np.dot(self.user_factors[user_idx], self.item_factors[movie_idx])

    def recommend_movies(self, user_id, user_map, movie_map, movies_df, rating_matrix, top_n):
        user_idx = user_map.get(user_id)
        if user_idx is None:
            return pd.DataFrame()
        rated_movies = rating_matrix.loc[user_id].dropna().index.tolist()
        predictions = []
        for movie_orig_id, movie_idx in movie_map.items():
            if movie_orig_id not in rated_movies:
                predicted_rating = self.predict_rating(user_idx, movie_idx)
                predictions.append({'movie_id': movie_orig_id, 'predicted_rating': predicted_rating})
        predictions_df = pd.DataFrame(predictions)
        if predictions_df.empty:
            return pd.DataFrame()
        recommended_movies = predictions_df.merge(movies_df, on='movie_id')
        recommended_movies = recommended_movies.sort_values(by='predicted_rating', ascending=False)
        return recommended_movies.head(top_n)

    def infer_new_user_embedding(self, new_user_ratings, movie_map):
        st.write("Inferring new user embedding...")
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
        A_new_user = (1 - self.w0) * sum_vj_vjT_obs + self.w0 * VtV + identity_matrix_k
        B_new_user = sum_rij_vj_obs
        try:
            new_user_embedding = np.linalg.solve(A_new_user, B_new_user)
            st.success("New user embedding inferred.")
            return new_user_embedding
        except np.linalg.LinAlgError:
            st.error("Singular matrix encountered while inferring new user embedding. Please try different ratings.")
            return None

    def recommend_for_new_user(self, new_user_embedding, movie_map, movies_df, rated_movie_ids, top_n):
        if new_user_embedding is None:
            return pd.DataFrame()
        predictions = []
        for movie_orig_id, movie_idx in movie_map.items():
            if movie_orig_id not in rated_movie_ids:
                predicted_rating = np.dot(new_user_embedding, self.item_factors[movie_idx])
                predictions.append({'movie_id': movie_orig_id, 'predicted_rating': predicted_rating})
        predictions_df = pd.DataFrame(predictions)
        if predictions_df.empty:
            return pd.DataFrame()
        recommended_movies = predictions_df.merge(movies_df, on='movie_id')
        recommended_movies = recommended_movies.sort_values(by='predicted_rating', ascending=False)
        return recommended_movies.head(top_n)
