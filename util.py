__all__ = ["stopwords", "load_process", "STOPWORDS", "DATA_PATH"]

import os

import pandas as pd
from nltk.corpus import stopwords

STOPWORDS = {'of', 'in', 'the', 'with'}

FILE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(FILE_DIR, "data")
ORDER_PATH = os.path.join(DATA_PATH, "order_products__train.csv")
PRODUCT_PATH = os.path.join(DATA_PATH, "products.csv")


def load_process(order_path: str = ORDER_PATH,
                 product_path: str = PRODUCT_PATH):
    o = pd.read_csv(order_path).dropna()
    p = pd.read_csv(product_path).dropna()
    p = p.rename(columns={"product_name": "name"})
    p["name"] = p["name"].str.replace("[^a-zA-Z0-9 ]", "").str.lower()
    p["name"] = p["name"].apply(
        lambda s: " ".join([w for w in s.split() if w not in STOPWORDS])
    )
    p["shortname"] = p["name"].str.split().str[-2:].apply(" ".join)
    id2short = dict(zip(p["product_id"], p["shortname"]))
    counts = o["product_id"].value_counts().to_dict()
    o = o[o["product_id"].map(counts) > 20]
    o["shortname"] = o["product_id"].map(id2short)

    data = o.groupby("order_id")["shortname"].apply(list)
    data = data[(data.apply(len) >= 10) & (data.apply(len) <= 50)]
    return o, data, o["shortname"].unique()
