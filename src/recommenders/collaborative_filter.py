import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender
from data.loader import build_user_product_matrix
from tqdm import tqdm
from scipy.special import expit


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


class SGDMatrixFactorizerCF(BaseRecommender):
    """
    Matrix Factorization based Collaborative filter.
    For a given user-product interaction matrix, finds the latent factors of users and products
    Weights are updated using Stochastic Gradient Descent
    """

    def __init__(
        self,
        normalization="l2",
        embedding_dim=20,
        n_iter=10000,
        sgd_sample_size=1000,
        lr=0.01,
        reg=0.01,
    ):
        self.embedding_dim = embedding_dim
        self.n_iter = n_iter
        self.sgd_sample_size = sgd_sample_size
        self.lr = lr
        self.reg = reg
        self.normalization = normalization

    def fit(self, interactions_df):
        (
            self.user_product_matrix,
            self.user_ids,
            self.product_ids,
        ) = build_user_product_matrix(interactions_df, self.normalization)

        n_users = len(self.user_ids)
        n_products = len(self.product_ids)

        scale = 0.01
        self.user_embedding_matrix = np.random.normal(
            0, scale, (n_users, self.embedding_dim)
        )
        self.product_embedding_matrix = np.random.normal(
            0, scale, (n_products, self.embedding_dim)
        )

        user_product_coo = self.user_product_matrix.tocoo()

        # user bias
        csr_data = self.user_product_matrix.data
        csr_ind_ptr = self.user_product_matrix.indptr

        self.global_mean = csr_data.mean() if csr_data.size > 0 else 0.0

        row_sum = np.asarray(self.user_product_matrix.sum(axis=1)).flatten()
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

        for i in tqdm(range(self.n_iter)):
            actual_sample_size = min(self.sgd_sample_size, len(user_product_coo.data))
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
                sampled_product_ids
            ]  # sgd_sample_size*embedding_dim

            pred = (
                np.sum(u_vecs * p_vecs, axis=1)
                + self.user_bias[sampled_user_ids]
                + self.product_bias[sampled_product_ids]
                + self.global_mean
            )  # sgd_sample_size
            err = sampled_actuals - pred

            user_updates = self.lr * (
                err[:, None] * p_vecs - self.reg * u_vecs
            )  # sgd_sample_size * embedding_dim

            product_updates = self.lr * (err[:, None] * u_vecs - self.reg * p_vecs)

            # handles updates for repeated users correctly
            np.add.at(self.user_embedding_matrix, sampled_user_ids, user_updates)
            np.add.at(
                self.product_embedding_matrix, sampled_product_ids, product_updates
            )

            # bias updates
            np.add.at(
                self.user_bias,
                sampled_user_ids,
                self.lr * (err - self.reg * self.user_bias[sampled_user_ids]),
            )
            np.add.at(
                self.product_bias,
                sampled_product_ids,
                self.lr * (err - self.reg * self.product_bias[sampled_product_ids]),
            )

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
            product_idxs
        ]  # n_product*embedd

        scores = (
            user_embedding @ product_embedding.T
            + self.global_mean
            + self.user_bias[user_idxs][:, None]
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

        predictions = self.predict([user_id], self.product_ids).flatten()
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


class ALSMatrixFactorizerCF(BaseRecommender):
    """
    Matrix Factorization based Collaborative filter.
    For a given user-product interaction matrix, finds the latent factors of users and products
    Weights are updated using Alternate Least Squares
    """

    def __init__(
        self,
        normalization="l2",
        embedding_dim=20,
        n_iter=100,
        reg=0.01,
        confidence_scale=20,
    ):
        self.embedding_dim = embedding_dim
        self.n_iter = n_iter
        self.reg = reg
        self.normalization = normalization
        self.confidence_scale = confidence_scale

    def fit(self, interactions_df):
        (
            self.user_product_matrix,
            self.user_ids,
            self.product_ids,
        ) = build_user_product_matrix(interactions_df, self.normalization)

        n_users = len(self.user_ids)
        n_products = len(self.product_ids)

        scale = 0.01
        self.user_embedding_matrix = np.random.normal(
            0, scale, (n_users, self.embedding_dim)
        )  # n_users*embedd

        self.product_embedding_matrix = np.random.normal(
            0, scale, (n_products, self.embedding_dim)
        )  # n_products*embedd

        user_product_csc = self.user_product_matrix.tocsc()

        regularization_matrix = self.reg * np.eye(self.embedding_dim)  # embedd*embedd
        for _ in tqdm(range(self.n_iter)):
            user_T_user = (
                self.user_embedding_matrix.T @ self.user_embedding_matrix
            )  # embedd*embedd
            product_T_product = (
                self.product_embedding_matrix.T @ self.product_embedding_matrix
            )  # embedd*embedd

            for u in range(n_users):
                row = self.user_product_matrix[u]

                # writing the whole equation like Ax=B
                A = product_T_product.copy()  # embedd*embedd
                B = np.zeros(self.embedding_dim)
                for j, r_uj in zip(row.indices, row.data):
                    cu = 1 + self.confidence_scale * r_uj
                    pj = self.product_embedding_matrix[j]  # embedd
                    A += (cu - 1) * np.outer(pj, pj)  # embedd*embedd
                    B += cu * pj  # embedd

                A += regularization_matrix  # embedd*embedd

                # solving using cholesky
                L = np.linalg.cholesky(A)
                y = np.linalg.solve(L, B)
                x = np.linalg.solve(L.T, y)  # embedd

                self.user_embedding_matrix[u] = x

            for p in range(n_products):
                col = user_product_csc[:, p]

                # writing the whole equation like Ax=B
                A = user_T_user.copy()  # embedd*embedd
                B = np.zeros(self.embedding_dim)
                for i, r_ip in zip(col.indices, col.data):
                    cu = 1 + self.confidence_scale * r_ip
                    ui = self.user_embedding_matrix[i]  # embedd
                    A += (cu - 1) * np.outer(ui, ui)  # embedd*embedd
                    B += cu * ui  # embedd

                A += regularization_matrix  # embedd*embedd

                # solving using cholesky
                L = np.linalg.cholesky(A)
                y = np.linalg.solve(L, B)
                x = np.linalg.solve(L.T, y)  # embedd

                self.product_embedding_matrix[p] = x

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
            product_idxs
        ]  # n_product*embedd

        scores = user_embedding @ product_embedding.T

        return scores

    def recommend(self, user_id, n=10):
        if not hasattr(self, "user_embedding_matrix"):
            raise RuntimeError("Model must be fitted before recommending")

        if user_id not in self.user_ids:
            raise ValueError(f"user id : {user_id} is not present during model fitting")

        user_idx = self.user_ids.get_loc(user_id)
        purchased_products_idx = self.user_product_matrix[user_idx].indices

        predictions = self.predict([user_id], self.product_ids).flatten()
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


class BPRMatrixFactorizerCF(BaseRecommender):
    """
    Matrix Factorization based Collaborative filter.
    For a given user-product interaction matrix, finds the latent factors of users and products
    Weights are updated using Bayesian Personalized Ranking Optimization
    """

    def __init__(
        self,
        normalization="l2",
        embedding_dim=20,
        n_iter=100,
        reg=0.01,
        user_batch_size=64,
        alpha=0.8,
        lr=0.01,
    ):
        self.embedding_dim = embedding_dim
        self.n_iter = n_iter
        self.reg = reg
        self.normalization = normalization
        self.user_batch_size = user_batch_size
        self.alpha = alpha
        self.lr = lr

    def _sample_batch(self, user_product_matrix, batch_size, alpha):
        num_users, num_items = user_product_matrix.shape
        user_items = user_product_matrix.indices
        user_values = user_product_matrix.data
        indptrs = user_product_matrix.indptr

        # sample users
        users = np.random.randint(0, num_users, size=batch_size)

        starts = indptrs[users]
        ends = indptrs[users + 1]
        lengths = ends - starts

        # remove users with no interactions
        valid = lengths > 0
        starts = starts[valid]
        ends = ends[valid]
        lengths = lengths[valid]
        users = users[valid]
        if len(users) == 0:
            return self._sample_batch(user_product_matrix, batch_size, alpha)

        # positive sampling
        random_offsets = (np.random.rand(batch_size) * lengths).astype(int)
        pos_indices = starts + random_offsets
        pos_items = user_items[pos_indices]
        pos_scores = user_values[pos_indices]

        # decide zero sample vs weak positive
        use_weak = np.random.rand(len(users)) < alpha

        # neg sampling
        neg_items = np.random.randint(0, num_items, size=batch_size)
        mask = ~use_weak

        while np.any(mask):
            u = users[mask]
            ni = neg_items[mask]

            start = indptrs[u]
            end = indptrs[u + 1]

            is_positive = np.array(
                [ni_i in user_items[s:e] for ni_i, s, e in zip(ni, start, end)]
            )

            # resample for invalid users
            neg_items[mask] = np.where(
                is_positive, np.random.randint(0, num_items, size=len(ni)), ni
            )

            # update mask
            mask_indices = np.where(mask)[0]
            mask[mask_indices] = is_positive

        # weak positives sampling
        weak_items = np.full(batch_size, -1, dtype=int)
        weak_indices = np.where(use_weak)[0]

        for i in weak_indices:
            s, e = starts[i], ends[i]
            curr_user_items = user_items[s:e]
            curr_user_scores = user_values[s:e]

            # find weaker items
            weak_mask = curr_user_scores < pos_scores[i]
            if np.any(weak_mask):
                candidates = curr_user_items[weak_mask]

                # pick a weak item randomly
                weak_items[i] = np.random.choice(candidates)
            else:
                use_weak[i] = False

        # combine final negative items
        final_neg_items = np.where(use_weak, weak_items, neg_items)

        return users, pos_items, final_neg_items

    def fit(self, interactions_df):
        (
            self.user_product_matrix,
            self.user_ids,
            self.product_ids,
        ) = build_user_product_matrix(interactions_df, self.normalization)

        n_users = len(self.user_ids)
        n_products = len(self.product_ids)

        scale = 0.01
        self.user_embedding_matrix = np.random.normal(
            0, scale, (n_users, self.embedding_dim)
        )  # n_users*embedd

        self.product_embedding_matrix = np.random.normal(
            0, scale, (n_products, self.embedding_dim)
        )  # n_products*embedd

        for _ in tqdm(range(self.n_iter)):
            users, pos_items, neg_items = self._sample_batch(
                self.user_product_matrix, self.user_batch_size, self.alpha
            )

            u_vec = self.user_embedding_matrix[users]  # batch_size*embedd
            p_vec = self.product_embedding_matrix[pos_items]  # batch_size*embedd
            n_vec = self.product_embedding_matrix[neg_items]  # batch_size*embedd

            x_u_i = np.sum(u_vec * p_vec, axis=1)
            x_u_j = np.sum(u_vec * n_vec, axis=1)
            x_u_i_j = x_u_i - x_u_j
            sigmoid_val = expit(-x_u_i_j)[:, np.newaxis]

            # gradient updates
            u_update = ((p_vec - n_vec) * sigmoid_val - self.reg * u_vec) * self.lr
            p_update = self.lr * (sigmoid_val * u_vec - self.reg * p_vec)
            n_update = self.lr * (-sigmoid_val * u_vec - self.reg * n_vec)

            np.add.at(self.user_embedding_matrix, users, u_update)
            np.add.at(self.product_embedding_matrix, pos_items, p_update)
            np.add.at(self.product_embedding_matrix, neg_items, n_update)

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
            product_idxs
        ]  # n_product*embedd

        scores = user_embedding @ product_embedding.T

        return scores

    def recommend(self, user_id, n=10):
        if not hasattr(self, "user_embedding_matrix"):
            raise RuntimeError("Model must be fitted before recommending")

        if user_id not in self.user_ids:
            raise ValueError(f"user id : {user_id} is not present during model fitting")

        user_idx = self.user_ids.get_loc(user_id)
        purchased_products_idx = self.user_product_matrix[user_idx].indices

        predictions = self.predict([user_id], self.product_ids).flatten()
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
