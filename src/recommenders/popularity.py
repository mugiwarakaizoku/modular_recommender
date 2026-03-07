from .base import BaseRecommender
import pandas as pd


class PopularityRecommender(BaseRecommender):
    """
    Recommends popular items. Serves as a baseline
    """

    def fit(self, interactions_df):
        self.interactions = interactions_df
        self.top_items = (
            interactions_df.groupby("product_id")["customer_unique_id"]
            .count()
            .sort_values(ascending=False)
            .index.tolist()
        )

    def recommend(self, user_id, n=10):
        if not hasattr(self, "top_items"):
            raise RuntimeError("Model must be fitted before recommending")
        seen_items = set(
            self.interactions[self.interactions["customer_unique_id"] == user_id][
                "product_id"
            ]
        )
        recomm = [item for item in self.top_items if item not in seen_items][:n]
        recomm_df = pd.DataFrame(recomm, columns=["product_id"])
        recomm_df = recomm_df.merge(
            self.interactions[
                ["product_id", "product_category_name", "product_category_name_english"]
            ].drop_duplicates(),
            on="product_id",
            how="left",
        )
        return recomm_df

    def recommend_batch(self, user_ids, n=10):
        all_recomms = []
        for user_id in user_ids:
            recomm_df = self.recommend(user, n)
            recomm_df["customer_unique_id"] = user
            all_recomms.append(recomm_df)

        return pd.concat(all_recomms, ignore_index=True)
