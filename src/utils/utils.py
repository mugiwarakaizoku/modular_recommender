import pandas as pd


def test_train_split_per_user(df, test_ratio=0.2):
    """
    Splits interaction data into train and test sets on a per-user basis.
    Parameters:
    ----------
    df : pd.DataFrame
        Input interactions dataframe containing at least 'customer_unique_id'
    test_ratio : float, optional (default=0.2)
        Fraction of interactions per user to include in test set
    Returns:
    -------
    train_df : pd.DataFrame
        Training interactions
    test_df : pd.DataFrame
        Testing interactions
    """
    train, test = [], []

    for user, group in df.groupby("customer_unique_id"):
        if len(group) < 2:
            train.append(group)
            continue
        items = group.sample(frac=1, random_state=42)
        split = int(len(items) * (1 - test_ratio))
        train.append(items.iloc[:split])
        test.append(items.iloc[split:])
    return pd.concat(train), pd.concat(test)
