import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender
from data.loader import build_user_product_matrix


class UserBasedCF(BaseRecommender):
    """
    User based collaborative filtering recommender.
    For a given user, finds the most similar users based on their purchase history. New products are recommended from these similar users
    """

    def fit(self, interactions_df, normalization="l2"):
        (
            self.user_product_matrix,
            self.user_ids,
            self.product_ids,
        ) = build_user_product_matrix(interactions_df, normalization)
        self.user_similarity = cosine_similarity(self.user_product_matrix)

    def recommend(self, user_id, n=10, k_neighbours=10):
        if not hasattr(self, "user_similarity"):
            raise RuntimeError("Model must be fitted before recommending")

        if user_id not in self.user_ids:
            raise ValueError(f"user id : {user_id} is not present during model fitting")

        user_idx = self.user_ids.get_loc(user_id)
        similarity = self.user_similarity[user_idx].copy()
        # Every user will have high similarity with themselves. So we need to mask it
        similarity[user_idx] = 0

        top_similar_users_idx = np.argsort(similarity)[-k_neighbours:]
        weights = similarity[top_similar_users_idx]

        similar_users_matrix = self.user_product_matrix[top_similar_users_idx]

        product_scores = similar_users_matrix.T @ weights

        # We need to mask products that user has already seen
        seen_items = self.user_product_matrix[user_idx].indices
        product_scores[seen_items] = 0

        top_product_idxs = np.argsort(product_scores)[-n:][::-1]
        top_product_ids = self.product_ids[top_product_idxs]
        top_product_scores = product_scores[top_product_idxs]

        return pd.DataFrame(
            {
                "customer_unique_id": user_id,
                "product_id": top_product_ids,
                "product_score": top_product_scores,
            }
        )

    def recommend_batch(self, user_ids, n=10, k_neighbours=10):
        result = [self.recommend(user_id, n, k_neighbours) for user_id in user_ids]
        return pd.concat(result, ignore_index=True)


class ProductBasedCF(BaseRecommender):
    """
    Product based collaborative filtering recommender.
    For a given user, finds the most similar products to the products that user has already used
    """

    def fit(self, interactions_df, normalization="l2"):
        (
            self.user_product_matrix,
            self.user_ids,
            self.product_ids,
        ) = build_user_product_matrix(interactions_df, normalization)
        self.product_similarity = cosine_similarity(self.user_product_matrix.T)

    def recommend(self, user_id, n=10, k_neighbours=10):
        if not hasattr(self, "product_similarity"):
            raise RuntimeError("Model must be fitted before recommending")

        if user_id not in self.user_ids:
            raise ValueError(f"user id : {user_id} is not present during model fitting")

        user_idx = self.user_ids.get_loc(user_id)
        # past purchases
        purchased_products_idx = self.user_product_matrix[user_idx].indices
        similarity = self.product_similarity[purchased_products_idx].copy()
        # Every product will have high similarity with themselves. So we need to mask it
        row = np.arange(len(purchased_products_idx))
        similarity[row, purchased_products_idx] = 0

        top_similar_products_idx = np.argsort(similarity)[:, -k_neighbours:]
        weights = similarity[
            row[:, None], top_similar_products_idx
        ].flatten()  # n_purchased*k_neighbours
        top_similar_products_idx = (
            top_similar_products_idx.flatten()
        )  # n_purchased*k_neighbours

        product_scores = np.bincount(
            top_similar_products_idx, weights=weights, minlength=len(self.product_ids)
        )  # n_products*

        # We need to mask products that user has already seen
        product_scores[purchased_products_idx] = 0

        top_product_idxs = np.argsort(product_scores)[-n:][::-1]
        top_product_ids = self.product_ids[top_product_idxs]
        top_product_scores = product_scores[top_product_idxs]

        return pd.DataFrame(
            {
                "customer_unique_id": user_id,
                "product_id": top_product_ids,
                "product_score": top_product_scores,
            }
        )

    def recommend_batch(self, user_ids, n=10, k_neighbours=10):
        result = [self.recommend(user_id, n, k_neighbours) for user_id in user_ids]
        return pd.concat(result, ignore_index=True)
