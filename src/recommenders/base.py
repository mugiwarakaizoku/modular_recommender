class BaseRecommender:
    def fit(self, interactions_df):
        pass

    def recommend(self, user_id, n=10):
        pass
