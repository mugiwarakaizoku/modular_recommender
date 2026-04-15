from pathlib import Path
from data.loader import load_interactions, build_user_product_matrix
from recommenders.popularity import PopularityRecommender
from recommenders.collaborative_filter import (
    UserBasedCF,
    ProductBasedCF,
    SGDMatrixFactorizerCF,
    ALSMatrixFactorizerCF,
    BPRMatrixFactorizerCF,
)
import random
from evaluation.metrics import precision_k, recall_k, ndcg_k
from utils.utils import test_train_split_per_user
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_dir = PROJECT_ROOT / "data" / "raw"

interactions = load_interactions(data_dir, min_interactions=3, review_threshold=4)
print(f"Length of Interactions: {len(interactions)}")

product_info = interactions[
    ["product_id", "product_category_name", "product_category_name_english"]
].drop_duplicates()

train, test = test_train_split_per_user(interactions)
test_users = list(test["customer_unique_id"].unique())
actuals = test[["customer_unique_id", "product_id"]]


models_dict = {
    "Popular Recommender": PopularityRecommender(),
    "Product Based CF": ProductBasedCF(),
    "User Based CF": UserBasedCF(),
    "SGD Matrix Factorization": SGDMatrixFactorizerCF(
        sgd_sample_size=400, n_iter=10000, embedding_dim=20
    ),
    "ALS Matrix Factorization": ALSMatrixFactorizerCF(n_iter=50, embedding_dim=150),
    "BPR Matrix Factorization": BPRMatrixFactorizerCF(n_iter=10000, embedding_dim=50),
}

model_comparison = []
for model_name, model in models_dict.items():
    model.fit(train)
    recomms = model.recommend_batch(user_ids=test_users)
    recomms = recomms.merge(product_info, on="product_id", how="left")

    recomm_df = recomms[["customer_unique_id", "product_id"]]

    precision = precision_k(actuals, recomm_df)

    recall = recall_k(actuals, recomm_df)

    ndcg = ndcg_k(actuals, recomm_df)

    model_comparison.append(
        {
            "model_name": model_name,
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
        }
    )

print(pd.DataFrame(model_comparison))
