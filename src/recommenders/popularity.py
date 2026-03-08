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

        self.seen_items = interactions_df.groupby("customer_unique_id")[
            "product_id"
        ].apply(set)

    def recommend(self, user_id, n=10):
        if not hasattr(self, "top_items"):
            raise RuntimeError("Model must be fitted before recommending")
        seen_items = self.seen_items.get(user_id, set())
        recomm = []
        for item in self.top_items:
            if item not in seen_items:
                recomm.append(item)
            if len(recomm) == n:
                break
        return pd.DataFrame({"customer_unique_id": user_id, "product_id": recomm})

    def recommend_batch(self, user_ids, n=10):
        result = [self.recommend(user_id, n) for user_id in user_ids]
        return pd.concat(result, ignore_index=True)
