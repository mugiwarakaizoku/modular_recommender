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

        top_similar_users_idx = np.argpartition(similarity, -k_neighbours)[
            -k_neighbours:
        ]
        weights = similarity[top_similar_users_idx]

        similar_users_matrix = self.user_product_matrix[top_similar_users_idx]

        product_scores = similar_users_matrix.T @ weights

        # We need to mask products that user has already seen
        seen_items = self.user_product_matrix[user_idx].indices
        n = max(1, min(n, len(product_scores) - len(seen_items)))
        product_scores[seen_items] = 0

        top_product_idxs = np.argpartition(product_scores, -n)[-n:]
        top_product_idxs = top_product_idxs[
            np.argsort(product_scores[top_product_idxs])[::-1]
        ]

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

        top_similar_products_idx = np.argpartition(similarity, -k_neighbours, axis=1)[
            :, -k_neighbours:
        ]
        weights = similarity[
            row[:, None], top_similar_products_idx
        ].flatten()  # n_purchased*k_neighbours
        top_similar_products_idx = (
            top_similar_products_idx.flatten()
        )  # n_purchased*k_neighbours

        product_scores = np.bincount(
            top_similar_products_idx, weights=weights, minlength=len(self.product_ids)
        )  # n_products*
        n = max(1, min(n, len(product_scores) - len(purchased_products_idx)))

        # We need to mask products that user has already seen
        product_scores[purchased_products_idx] = 0

        top_product_idxs = np.argpartition(product_scores, -n)[-n:]
        top_product_idxs = top_product_idxs[
            np.argsort(product_scores[top_product_idxs])[::-1]
        ]

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


class MatrixFactorizerCF(BaseRecommender):
    """
    Matrix Factorization based Collaborative filter.
    For a given user-product interaction matrix, finds the latent factors of users and products
    """

    def fit(
        self,
        interactions_df,
        normalization="l2",
        embedding_dim=40,
        n_iter=10000,
        sgd_sample_size=10000,
        lr=0.01,
    ):
        (
            self.user_product_matrix,
            self.user_ids,
            self.product_ids,
        ) = build_user_product_matrix(interactions_df, normalization)
        n_users = len(self.user_ids)
        n_products = len(self.product_ids)

        scale = 0.01
        self.user_embedding_matrix = np.random.normal(
            0, scale, (n_users, embedding_dim)
        )
        self.product_embedding_matrix = np.random.normal(
            0, scale, (embedding_dim, n_products)
        )

        user_product_coo = self.user_product_matrix.tocoo()

        # user bias
        csr_data = self.user_product_matrix.data
        csr_ind_ptr = self.user_product_matrix.indptr

        self.global_mean = csr_data.mean() if csr_data.size > 0 else 0.0

        row_sum = np.add.reduceat(csr_data, csr_ind_ptr[:-1])
        row_cnt = np.diff(csr_ind_ptr)
        user_means = np.divide(
            row_sum,
            row_cnt,
            out=np.full(n_users, self.global_mean, dtype=csr_data.dtype),
            where=row_cnt != 0,
        )
        self.user_bias = user_means - self.global_mean

        # product bias
        coo_data = user_product_coo.data
        cols = user_product_coo.col
        col_sum = np.bincount(cols, weights=coo_data, minlength=n_products)
        col_cnt = np.bincount(cols, minlength=n_products)

        product_means = np.divide(
            col_sum,
            col_cnt,
            out=np.full(n_products, self.global_mean, dtype=coo_data.dtype),
            where=col_cnt != 0,
        )
        self.product_bias = product_means - self.global_mean

        for i in range(n_iter):
            actual_sample_size = min(sgd_sample_size, len(user_product_coo.data))
            idx = np.random.choice(
                len(user_product_coo.data), actual_sample_size, replace=False
            )

            sampled_user_ids = user_product_coo.row[idx]  # sgd_sampled_size*1
            sampled_product_ids = user_product_coo.col[idx]  # sgd_sampled_size*1
            sampled_actuals = user_product_coo.data[idx]  # sgd_sampled_size*1

            u_vecs = self.user_embedding_matrix[
                sampled_user_ids
            ]  # sgd_sample_size*embedding_dim
            p_vecs = self.product_embedding_matrix[
                :, sampled_product_ids
            ].T  # sgd_sample_size*embedding_dim

            pred = (
                np.sum(u_vecs * p_vecs, axis=1)
                + self.user_bias[sampled_user_ids]
                + self.product_bias[sampled_product_ids]
                + self.global_mean
            )  # sgd_sample_size
            err = sampled_actuals - pred

            user_updates = lr * (
                err[:, None] * p_vecs
            )  # sgd_sample_size * embedding_dim
            product_updates = lr * (err[:, None] * u_vecs)

            # handles updates for repeated users correctly
            np.add.at(self.user_embedding_matrix, sampled_user_ids, user_updates)

            tmp_prod_embedding = self.product_embedding_matrix.T
            np.add.at(tmp_prod_embedding, sampled_product_ids, product_updates)

            # bias updates
            np.add.at(self.user_bias, sampled_user_ids, lr * err)
            np.add.at(self.product_bias, sampled_product_ids, lr * err)

    def predict(self, user_ids, product_ids):
        user_idxs = self.user_ids.get_indexer(user_ids)
        product_idxs = self.product_ids.get_indexer(product_ids)

        if (user_idxs == -1).any():
            missing = np.array(user_ids)[user_idxs == -1].tolist()
            raise ValueError(f"Unknown user_ids: {missing}")
        if (product_idxs == -1).any():
            missing = np.array(product_ids)[product_idxs == -1].tolist()
            raise ValueError(f"Unknown product_ids: {missing}")

        user_embedding = self.user_embedding_matrix[user_idxs]  # n_user*embedd
        product_embedding = self.product_embedding_matrix[
            :, product_idxs
        ].T  # n_product*embedd

        scores = (
            np.sum(user_embedding * product_embedding, axis=1)
            + self.global_mean
            + self.user_bias[user_idxs]
            + self.product_bias[product_idxs]
        )

        return scores

    def recommend(self, user_id, n=10):
        if not hasattr(self, "user_embedding_matrix"):
            raise RuntimeError("Model must be fitted before recommending")

        if user_id not in self.user_ids:
            raise ValueError(f"user id : {user_id} is not present during model fitting")

        user_idx = self.user_ids.get_loc(user_id)
        purchased_products_idx = self.user_product_matrix[user_idx].indices

        user_embedding = self.user_embedding_matrix[user_idx]  # 1*embedding_dim
        predictions = (
            (user_embedding @ self.product_embedding_matrix).flatten()
            + self.global_mean
            + self.user_bias[user_idx]
            + self.product_bias
        )  # n_products
        predictions[purchased_products_idx] = -np.inf

        top_product_idxs = np.argpartition(predictions, -n)[-n:]
        top_product_idxs = top_product_idxs[
            np.argsort(predictions[top_product_idxs])[::-1]
        ]

        top_product_ids = self.product_ids[top_product_idxs]
        top_product_scores = predictions[top_product_idxs]

        return pd.DataFrame(
            {
                "customer_unique_id": user_id,
                "product_id": top_product_ids,
                "product_score": top_product_scores,
            }
        )

    def recommend_batch(self, user_ids, n=10):
        result = [self.recommend(user_id, n) for user_id in user_ids]
        return pd.concat(result, ignore_index=True)
