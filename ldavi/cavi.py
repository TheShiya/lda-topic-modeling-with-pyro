"""
Implemented LDA with CAVI algorithm.
"""

from functools import lru_cache
from typing import List, Union

import numpy as np
import pyro
import pyro.distributions as dist
import torch

from numerical.lda_alpha_gradient_descent import lda_linear_newton


class LDACAVI(object):
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor,
                 corpora: list, num_topics: int, num_particles: int = 10):
        """
        Parameters
        ----------
        alpha: (num_topics)
        beta: (num_topics, len(corpora))
        corpora: the unique vocabularies with a give order.
        num_topics: the hyperparameter in LDA model.
        num_particles: the number of particles used in estimating expection.
            (like ELBO or other MC integrals)
            
        References
        ----------
        Latent Dirichlet Allocation
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
        if num_topics != alpha.shape[0]:
            raise ValueError("Mismatched size of alpha and num_topics.")

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
        """
        Document/ Order level guide.

        Parameters
        ----------
        order: Document/ Order.

        Returns
        -------
        Latent variables theta and z.
        """
        with pyro.plate("theta_", 1):
            theta = pyro.sample("theta", dist.Dirichlet(self.gamma))
        with pyro.plate("product_", len(order)) as ind:
            z = pyro.sample("z", dist.Categorical(self.phi[ind]))
        return theta, z

    def model(self, order: np.array):
        """
        Document/ Order level model.

        Parameters
        ----------
        order: Document/ Order (works as the observation of the model).

        Returns
        -------
        Latent variables theta and z.
        """
        with pyro.plate("theta_", 1):
            theta = pyro.sample("theta", dist.Dirichlet(self.alpha))
        with pyro.plate("product_", len(order)):
            z = pyro.sample("z", dist.Categorical(theta))
            w = pyro.sample("products",
                            dist.Categorical(self.beta[z]), obs=order)
        return theta, z, w

    def calc_elbo(self, order: torch.Tensor) -> float:
        """
        Calculate ELBO given guide and model.

        Parameters
        ----------
        order: observations

        Returns
        -------
        The evidence of lower bound.
        """
        elbo = torch.tensor(0.0)
        for _ in range(self.num_particles):
            guide_trace = pyro.poutine.trace(self.guide).get_trace(order)
            model_trace = pyro.poutine.trace(
                pyro.poutine.replay(self.model,
                                    trace=guide_trace)).get_trace(order)
            elbo += model_trace.log_prob_sum() - guide_trace.log_prob_sum()

        return -elbo.detach().numpy() / self.num_particles

    def gen_w_matrix(self, documents: np.ndarray) -> List[torch.Tensor]:
        """
        Generate the one-hot encoding format for words. Will be used in EM algo

        Parameters
        ----------
        documents

        Returns
        -------

        """
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
