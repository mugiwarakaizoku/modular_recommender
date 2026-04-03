from pathlib import Path
from data.loader import load_interactions, build_user_product_matrix
from recommenders.popularity import PopularityRecommender
from recommenders.collaborative_filter import (
    UserBasedCF,
    ProductBasedCF,
    MatrixFactorizerCF,
)
import random
from evaluation.metrics import precision_k, recall_k, ndcg_k
from utils.utils import test_train_split_per_user

PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_dir = PROJECT_ROOT / "data" / "raw"

interactions = load_interactions(data_dir, min_interactions=5)
product_info = interactions[
    ["product_id", "product_category_name", "product_category_name_english"]
].drop_duplicates()

train, test = test_train_split_per_user(interactions)
test_users = list(test["customer_unique_id"].unique())

model = MatrixFactorizerCF()
model.fit(train)

recomms = model.recommend_batch(user_ids=test_users)
recomms = recomms.merge(product_info, on="product_id", how="left")
print("Recommendations: \n", recomms.head(20))

recomm_df = recomms[["customer_unique_id", "product_id"]]
actuals = test[["customer_unique_id", "product_id"]]

precision = precision_k(actuals, recomm_df)
print("Precision: ", precision)

recall = recall_k(actuals, recomm_df)
print("Recall: ", recall)

ndcg = ndcg_k(actuals, recomm_df)
print("nDCG: ", ndcg)
