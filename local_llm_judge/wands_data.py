import pandas as pd


def pairwise_df(n):
    try:
        products = pd.read_csv('data/WANDS/dataset/product.csv', delimiter='\t')
        queries = pd.read_csv('data/WANDS/dataset/query.csv', delimiter='\t')
        labels = pd.read_csv('data/WANDS/dataset/label.csv', delimiter='\t')
    except FileNotFoundError:
        msg = ("Please download the WANDS dataset from https://github.com/wayfair/WANDS/" +
               "and place it in the data folder")
        raise FileNotFoundError(msg)
    labels.loc[labels['label'] == 'Exact', 'grade'] = 2
    labels.loc[labels['label'] == 'Partial', 'grade'] = 1
    labels.loc[labels['label'] == 'Irrelevant', 'grade'] = 0
    labels = labels.merge(queries, how='left', on='query_id')
    labels = labels.merge(products, how='left', on='product_id')

    # Sample n rows
    labels = labels.sample(10000, random_state=42)

    # Get pairwise
    pairwise = labels.merge(labels, on='query_id')
    # Shuffle completely, otherwise they're somewhat sorted on query
    pairwise = pairwise.sample(frac=1, random_state=42)

    # Drop same id
    pairwise = pairwise[pairwise['product_id_x'] != pairwise['product_id_y']]

    # Drop same rating
    pairwise = pairwise[pairwise['label_x'] != pairwise['label_y']]

    assert n <= len(pairwise), f"Only {len(pairwise)} rows available"
    return pairwise.head(n)
