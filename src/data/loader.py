import pandas as pd
from pathlib import Path


def load_interactions(
    data_dir: str,
    signal: str = "purchase",
    review_threshold: int = 4,
    min_interactions: int = 1,
) -> pd.DataFrame:
    """
    Builds a user-items interaction dataframe

    Parameters:
    data_dir: Path to raw CSV files
    signal: Definition of a positive user-item interaction. One of:
            "purchase" - all delivered items.
            "positive" - delivered purhcases with review_score >= review_threshold
    review_threshold: Minimum review score of delivered items
    min_interactions: Minimum number of interactions a user must have

    Returns: A pandas dataframe with columns:
                customer_unique_id
                product_id
                interaction_count
                product_category_name
                product_category_name_english
            One row per (customer_unique_id,product_id) pair
    """
    data_dir = Path(data_dir)
    df_orders = pd.read_csv(data_dir / "olist_orders_dataset.csv")
    df_items = pd.read_csv(data_dir / "olist_order_items_dataset.csv")
    df_customers = pd.read_csv(data_dir / "olist_customers_dataset.csv")
    df_products = pd.read_csv(data_dir / "olist_products_dataset.csv")
    df_prodcut_category = pd.read_csv(
        data_dir / "product_category_name_translation.csv"
    )

    df_delivered_orders = df_orders[df_orders["order_status"] == "delivered"][
        ["order_id", "customer_id"]
    ]

    df = df_items[["order_id", "product_id"]].merge(
        df_delivered_orders, on="order_id", how="inner"
    )

    df = df.merge(
        df_customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="inner",
    )

    if signal == "positive":
        df_reviews = pd.read_csv(data_dir / "olist_order_reviews_dataset.csv")
        df_reviews = (
            df_reviews.groupby("order_id")["review_score"].max().reset_index(drop=True)
        )
        df = df.merge(df_reviews, on="order_id", how="left")
        df = df[df["review_score"] >= review_threshold]

    interactions = (
        df.groupby(["customer_unique_id", "product_id"])
        .size()
        .reset_index(name="interaction_count")
    )
    if min_interactions > 1:
        user_cnts = interactions.groupby("customer_unique_id")[
            "interaction_count"
        ].sum()
        active_usrs = user_cnts[user_cnts >= min_interactions].index
        interactions = interactions[
            interactions["customer_unique_id"].isin(active_usrs)
        ]

    interactions = interactions.reset_index(drop=True).sort_values(
        by="interaction_count", ascending=False
    )
    interactions = interactions.merge(
        df_products[["product_id", "product_category_name"]],
        on="product_id",
        how="left",
    ).merge(df_prodcut_category, on="product_category_name", how="left")
    return interactions


def build_user_product_matrix(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Converts Interactions dataframe into a dense user-product matrix
    Rows represent users, columns represent products, values represent interaction counts

    Parameters:
    Interactions: Output of load_interactions()

    Returns: A dense user-product matrix
    """
    matrix = interactions.pivot_table(
        index="customer_unique_id",
        columns="product_id",
        values="interaction_count",
        fill_value=0,
    )
    matrix.columns.name = None
    return matrix
