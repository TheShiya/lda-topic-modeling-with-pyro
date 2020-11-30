__all__ = ["stopwords", "load_process", "STOPWORDS", "DATA_PATH"]

import os
import pyro
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def mc(q, func, n_trials=200, n_samples=100):
    eval_means = []
    for _ in range(n_trials):
        evals = []
        for i in range(n_samples):
            x_i = pyro.sample('x_{}'.format(i), q)
            evals.append(func(x_i).numpy())
        eval_means.append(np.mean(evals, axis=0))
    eval_means = np.array(eval_means)
    return eval_means


def run_mc_experiment(dimensions, funcs, q, n_trials=50, n_samples=100):
    sds = []
    print('finished:', end=' ')
    for dimension in dimensions:
        print(dimension, end=' ')
        for func in funcs:
            param = torch.FloatTensor(abs(np.random.randn(dimension)))
            param /= param.sum()
            means = mc(q(param), func, n_trials=n_trials, n_samples=n_samples)
            sd = (means.std(0) / (means.mean(0) + 1e-15)).mean()
            sds.append(sd)
    plt.plot(dimensions, np.array(sds).reshape(-1, len(funcs)))
    plt.title('Mean coefficient of variation (sd/mean) across dimensions')
    plt.xlabel('Dimension')
