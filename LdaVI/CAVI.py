import numpy as np
import torch
import pyro.distributions as dist
import pyro
from functools import lru_cache
from typing import List, Union
from Numerical.LDALinearRN import lda_linear_newton


class LDACAVI(object):
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor,
                 corpora: list, num_topics: int, num_particles: int = 10):
        """
        Parameters
        ----------
        alpha: (num_topics)
        beta: (num_topics, len(corpora))
        corpora: the unique vocabularies with a give order.
        num_topics

        Attributes
        ----------
        phi: multinomial parameters. For each word, there is a multinomial
        distribution that control the probability of this word appearing
        in the document.

        gamma: Dirichlet parameters. gamma controls the probability
        of the distribution of the topics.
        """
        self.alpha = alpha
        self.beta = beta
        self.vocab_len = len(corpora)
        self.num_topics = num_topics
        self.phi = None
        self.gamma = None
        self.corpora = corpora
        self.trace_elbo = []
        self.num_particles = num_particles

    @lru_cache(maxsize=None)
    def encoding_doc(self, document: tuple) -> torch.Tensor:
        return torch.tensor(np.array([self.corpora.index(order)
                                      for order in document]))

    def cavi(self, document: Union[np.array, List[str]],
             keep_elbo=True, show_step=np.nan, tol: float = 1e-3,
             max_step: int = 5000):
        document = self.encoding_doc(tuple(document))
        self.gamma = torch.tensor(data=self.alpha.detach().
                                  numpy() + len(document) / self.num_topics,
                                  requires_grad=True)
        self.phi = torch.tensor(data=np.full(shape=(len(document),
                                                    self.num_topics),
                                             fill_value=1 / self.num_topics))
        num_step, gamma_diff_norm, phi_diff_norm = 0, 1, 1

        while (num_step <= max_step and (
                gamma_diff_norm + phi_diff_norm) >= tol):
            old_gamma = self.gamma.clone().detach()
            old_phi = self.phi.clone().detach()
            for word_ind in range(len(document)):
                cur_word = document[word_ind]
                for topic in range(self.num_topics):
                    tmp_exp = torch.digamma(self.gamma[topic]).exp()
                    beta_tmp = self.beta[topic][cur_word]
                    self.phi[word_ind][topic] = beta_tmp * tmp_exp
                phi_sum = self.phi[word_ind, :].sum()
                self.phi[word_ind, :] = self.phi[word_ind, :] / phi_sum
            self.gamma = self.alpha + torch.sum(self.phi, dim=0)

            if keep_elbo:
                self.trace_elbo.append(self.calc_elbo(document))
            gamma_diff_norm = torch.norm(self.gamma - old_gamma)
            phi_diff_norm = torch.norm(self.phi - old_phi)
            if num_step % show_step == 0:
                print(f"Step {num_step} | phi diff norm={phi_diff_norm} |"
                      f" gamma diff norm={gamma_diff_norm}")
            num_step += 1
        return self.gamma, self.phi

    def reset_graph(self):
        self.gamma = None
        self.phi = None
        self.trace_elbo = []

    def guide(self, order: np.ndarray):
        with pyro.plate("theta_", 1):
            theta = pyro.sample("theta", dist.Dirichlet(self.gamma))
        with pyro.plate("product_", len(order)) as ind:
            z = pyro.sample("z", dist.Categorical(self.phi[ind]))
        return theta, z

    def model(self, order: np.array):
        with pyro.plate("theta_", 1):
            theta = pyro.sample("theta", dist.Dirichlet(self.alpha))
        with pyro.plate("product_", len(order)):
            z = pyro.sample("z", dist.Categorical(theta))
            w = pyro.sample("products",
                            dist.Categorical(self.beta[z]), obs=order)
        return theta, z, w

    def calc_elbo(self, order: torch.Tensor) -> float:
        elbo = torch.tensor(0.0)
        for _ in range(self.num_particles):
            guide_trace = pyro.poutine.trace(self.guide).get_trace(order)
            model_trace = pyro.poutine.trace(
                pyro.poutine.replay(self.model,
                                    trace=guide_trace)).get_trace(order)
            elbo += model_trace.log_prob_sum() - guide_trace.log_prob_sum()

        return -elbo.detach().numpy() / self.num_particles

    def gen_w_matrix(self, documents: np.ndarray) -> [torch.Tensor]:
        array = [torch.nn.functional.one_hot(self.encoding_doc(tuple(doc))
                                             .to(torch.int64),
                                             num_classes=len(self.corpora))
                 for doc in documents]
        return array

    def _estimate_params_e(self, documents: np.ndarray):
        gamma_list = []
        phi_list = []
        sum_elbo = 0
        for doc in documents:
            doc_gamma, doc_phi = self.cavi(doc, keep_elbo=False)
            gamma_list.append(doc_gamma.detach().numpy())
            phi_list.append(doc_phi.detach())
            _doc = self.encoding_doc(tuple(doc))
            sum_elbo += self.calc_elbo(_doc)
        return sum_elbo, torch.tensor(data=gamma_list), phi_list

    def _estimate_params_m(self, phi_list: List[torch.Tensor],
                           gamma_list: torch.Tensor,
                           w_list: List[torch.Tensor]):
        beta_unscale = torch.zeros(size=self.beta.shape)
        for doc_ind in range(len(w_list)):
            doc = w_list[doc_ind]
            for word_ind in range(len(doc)):
                beta_unscale += torch.einsum("i, j -> ij",
                                             phi_list[doc_ind][word_ind],
                                             doc[word_ind])
        self.beta = beta_unscale / beta_unscale.sum(-1).view(-1, 1)
        self.alpha = lda_linear_newton(self.alpha, gamma=gamma_list)

    def estimate_params(self, documents: np.array, tol: float = 5e-3,
                        max_iter: int = 500, show_step=1):
        w_list = self.gen_w_matrix(documents)
        beta_diff_norm, alpha_diff_norm, _iter = 1, 1, 0
        while beta_diff_norm + alpha_diff_norm > tol and _iter <= max_iter:
            old_alpha = self.alpha.detach()
            old_beta = self.beta.detach()
            elbo, gamma_list, phi_list = self._estimate_params_e(documents)
            self._estimate_params_m(phi_list, gamma_list, w_list)
            beta_diff_norm = torch.norm(self.beta - old_beta)
            alpha_diff_norm = torch.norm(self.alpha - old_alpha)
            _iter += 1
            self.trace_elbo.append(elbo)
            if _iter % show_step == 0:
                print(f"Step {_iter} | beta diff norm={beta_diff_norm} |"
                      f" alpha diff norm={alpha_diff_norm} | ELBO={elbo}")
        return self.alpha, self.beta


if __name__ == "__main__":
    corpora_ = ["a", "b", "c", "d", "e", "f"]
    num_topics_ = 2
    alpha = torch.tensor(data=[20., 5])
    beta = torch.tensor(data=[[0.2, 0.2, 0.2, 0.2, 0.2],
                              [0.2, 0.2, 0.2, 0.2, 0.2]])
    topic = dist.DirichletMultinomial(alpha)()
    beta_p = torch.matmul(topic, beta)
    words = torch.distributions.multinomial.Multinomial(total_count=20,
                                                        probs=beta_p).sample()
    # A document with 20 words
    document_ = []
    for i in range(len(words)):
        document_ += [corpora_[i]] * int(words[i])
    obj = LDACAVI(alpha=torch.tensor(data=[5., 20.]),
                  beta=beta, corpora=corpora_, num_topics=num_topics_,
                  num_particles=10)
    gamma, phi = obj.cavi(document_, show_step=2)
    # import pandas as pd
    # # DATA_PATH = r"F:\PSun-dev\Python\mlpp_project\data\order_data.csv"
    # DATA_PATH = r"F:\PSun-data\MLPPdata\order_products__train.csv"
    # VOCAB_PATH = r"F:\PSun-data\MLPPdata\products.csv"
    # data = pd.read_csv(DATA_PATH)[["order_id", "product_id"]]
    # data = data.groupby("order_id")
    # data = data.apply(lambda x: x["product_id"].to_list()).to_list()[:20]
    # vocab = pd.read_csv(VOCAB_PATH)["product_id"].to_list()
    # # data = pd.read_csv(DATA_PATH).values
    # # vocab = list(np.unique(data.reshape(-1, 1)))
    # alpha = torch.tensor(data=[50., 50, 50, 50, 50])
    # beta = torch.rand(size=(len(alpha), len(vocab)))
    # beta = beta / beta.sum(-1).view(-1, 1)
    # obj = LDACAVI(alpha=alpha, beta=beta,
    #               corpora=vocab, num_topics=5, num_particles=10)
    # alpha, beta = obj.estimate_params(data)
