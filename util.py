__all__ = ["STOPWORDS", "load_process", "STOPWORDS", "DATA_PATH",
           "run_mc_experiment"]

import os

import matplotlib.pyplot as plt
from ldavi.cavi import LDACAVI
import numpy as np
import pandas as pd
import pyro
import torch

STOPWORDS = {"of", "in", "the", "with", "oz", "liter", "count", "stem",
             "calorie", "a", "aa", "an", "ct", "gallon", "bag", "inch",
             "inches", "ounce", "packs", "pack", "pk", "c", "fl", "links",
             "month", "z", "hundred", "million", "billion", "d", "go", "on",
             "os"}

FILE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(FILE_DIR, "data")
ORDER_PATH = os.path.join(DATA_PATH, "order_products__train.csv")
PRODUCT_PATH = os.path.join(DATA_PATH, "products.csv")


def load_process(order_path: str = ORDER_PATH,
                 product_path: str = PRODUCT_PATH,
                 nrows=500):
    o = pd.read_csv(order_path).dropna()
    p = pd.read_csv(product_path).dropna()
    p = p.rename(columns={"product_name": "name"})
    p["name"] = p["name"].str.replace("[^a-zA-Z ]", "").str.lower()
    p["name"] = p["name"].apply(
        lambda s: " ".join([w for w in s.split() if w not in STOPWORDS])
    )
    p["shortname"] = p["name"].str.split().str[-2:].apply(" ".join)
    p["shortname"] = p["shortname"].replace("", "unknown")
    id2short = dict(zip(p["product_id"], p["shortname"]))
    counts = o["product_id"].value_counts().to_dict()
    o = o[o["product_id"].map(counts) > 20]
    o["shortname"] = o["product_id"].map(id2short)

    data = o.groupby("order_id")["shortname"].apply(list)
    data = data[(data.apply(len) >= 10) & (data.apply(len) <= 50)]
    if nrows is not None:
        sub_data = data.values[:nrows]
    else:
        sub_data = data.values
    vocab = []
    for order in sub_data:
        vocab += order
    vocab = np.unique(vocab)
    return o, sub_data, vocab


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


def run_mc_experiment(dimensions, funcs, q, func_names: list = None,
                      n_trials=50, n_samples=100,):
    sds = []
    print('finished:', end=' ')
    for dimension in dimensions:
        print(dimension, end=' ')
        for func in funcs:
            param = torch.tensor(abs(np.random.randn(dimension)))
            param /= param.sum()
            means = mc(q(param), func, n_trials=n_trials, n_samples=n_samples)
            sd = (means.std(0) / (means.mean(0) + 1e-15)).mean()
            sds.append(sd)
    plt.plot(dimensions, np.array(sds).reshape(-1, len(funcs)))
    plt.title('Mean coefficient of variation (sd/mean) across dimensions')
    plt.xlabel('Dimension')
    if func_names is not None:
        plt.legend(func_names)


def cavi_topic_criticism(corpora: list, train_data, valid_data,
                         topic_list: list, max_iter=500):
    for num_topics in topic_list:
        print(f"running num_topics={num_topics}")
        alpha = torch.rand(size=(num_topics,)) * 10
        beta = torch.rand(size=(len(alpha), len(corpora)))
        beta /= beta.sum(-1).view(-1, 1)

        cavi_obj = LDACAVI(alpha, beta,
                           corpora, num_topics,
                           num_particles=1)

        cavi_obj.reset_graph()

        # Parameter estimation
        _, _ = cavi_obj.estimate_params(
            train_data, valid_data, tol=2e-3,
            show_step=np.nan, max_iter=max_iter)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Num of topic: {num_topics}')
        ax1.plot(np.linspace(1, len(cavi_obj.trace_elbo),
                             len(cavi_obj.trace_elbo)),
                 cavi_obj.trace_elbo)
        ax1.set_title("EBLO")
        ax1.grid()
        ax2.plot(np.linspace(1, len(cavi_obj.trace_log_prob),
                             len(cavi_obj.trace_log_prob)),
                 cavi_obj.trace_log_prob)
        ax2.set_title("Log Predictive Prob")
        ax2.grid()
        ax3.plot(np.linspace(1, len(cavi_obj.trace_validate_prob),
                             len(cavi_obj.trace_validate_prob)),
                 cavi_obj.trace_validate_prob)
        ax3.grid()
        ax3.set_title("Log Validate Prob")
