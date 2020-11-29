import pandas as pd
import os


ORDER_PATH = os.path.join(".", "data", "order_products__train.csv")
PRODUCT_PATH = os.path.join(".", "data", "product_name.csv")
STOPWORDS = {'of', 'in', 'the', 'with'}


def load_process(order_path=ORDER_PATH, product_path=PRODUCT_PATH):
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
    return data