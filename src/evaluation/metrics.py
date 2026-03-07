import pandas as pd


def precision_k(
    actual_df: pd.DataFrame, recommended_df: pd.DataFrame, k: int = 10
) -> float:
    """
    Computes precision@k

    Parameters:
    actual_df:  Ground truth customer, product interactions
    recommended_df: Model recommendations
    k: Number of top recommendations to evaluate per user

    Returns: Mean precision@k across all users
    """
    actual_items = actual_df.groupby("customer_unique_id")["product_id"].apply(set)
    recommended_items = recommended_df.groupby("customer_unique_id")[
        "product_id"
    ].apply(list)

    precisions = []
    for user, recomm in recommended_items.items():
        top_k_recomm = set(recomm[:k])
        relevant_items = actual_items.get(user, set())
        len_matches = len(top_k_recomm & relevant_items)
        precision = len_matches / k
        precisions.append(precision)

    if len(precisions) == 0:
        return 0
    return sum(precisions) / len(precisions)
