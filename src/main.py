from pathlib import Path
from data.loader import load_interactions
from recommenders.popularity import PopularityRecommender

PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_dir = PROJECT_ROOT / "data" / "raw"

interactions = load_interactions(data_dir)

model = PopularityRecommender()
model.fit(interactions)

user_id = "c8460e4251689ba205045f3ea17884a1"
recomms = model.recommend(user_id=user_id)
print("Recommendations for user: ", user_id)
print(recomms)
