import pandas as pd
import math


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


def recall_k(
    actual_df: pd.DataFrame, recommended_df: pd.DataFrame, k: int = 10
) -> float:
    """
    Computes recall@k

    Parameters:
    actual_df:  Ground truth customer, product interactions
    recommended_df: Model recommendations
    k: Number of top recommendations to evaluate per user

    Returns: Mean recall@k across all users
    """
    actual_items = actual_df.groupby("customer_unique_id")["product_id"].apply(set)
    recommended_items = recommended_df.groupby("customer_unique_id")[
        "product_id"
    ].apply(list)

    recalls = []
    for user, items in actual_items.items():
        if len(items) == 0:
            continue
        recomm_items = recommended_items.get(user, [])
        top_k_recomm = set(recomm_items[:k])
        len_matches = len(top_k_recomm & items)
        recall = len_matches / len(items)
        recalls.append(recall)

    if len(recalls) == 0:
        return 0
    return sum(recalls) / len(recalls)


def ndcg_k(actual_df: pd.DataFrame, recommended_df: pd.DataFrame, k: int = 10) -> float:
    """
    Computes nDCG@k

    Parameters:
    actual_df:  Ground truth customer, product interactions
    recommended_df: Model recommendations
    k: Number of top recommendations to evaluate per user

    Returns: Mean nDCG@k across all users
    """
    actual_items = actual_df.groupby("customer_unique_id")["product_id"].apply(set)
    recommended_items = recommended_df.groupby("customer_unique_id")[
        "product_id"
    ].apply(list)

    ndcgs = []
    for user, recom_item in recommended_items.items():
        relevant_items = actual_items.get(user, set())
        if len(relevant_items) == 0:
            continue
        top_k_recomm = recom_item[:k]

        dcg = 0
        for idx, item in enumerate(top_k_recomm):
            flag = 1 if item in relevant_items else 0
            dcg += flag / math.log2(idx + 2)

        ideal_matches = min(len(relevant_items), k)
        idcg = sum(1 / math.log2(idx + 2) for idx in range(ideal_matches))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    if len(ndcgs) == 0:
        return 0
    return sum(ndcgs) / len(ndcgs)
