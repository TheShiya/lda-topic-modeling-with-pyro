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
        self.__init_alpha = alpha
        self.__inti_beta = beta
        self.trace_log_prob = []
        self.trace_validate_prob = []
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
        self.alpha = self.__init_alpha
        self.beta = self.__inti_beta
        self.trace_elbo = []
        self.trace_log_prob = []
        self.trace_validate_prob = []

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
        return elbo.detach().numpy() / self.num_particles

    # def calc_log_prob(self, order: torch.Tensor):
    #     prob_w = torch.tensor(data=0., dtype=torch.float64)
    #     for _ in range(self.num_particles):
    #         model_trace = pyro.poutine.trace(self.model).get_trace(order)
    #         model_trace.log_prob_sum()
    #         prob_w_tmp = model_trace.nodes["products"]["log_prob_sum"]
    #         prob_w_tmp = torch.tensor(float(prob_w_tmp),
    #                                   dtype=torch.float64).exp()
    #         prob_w += prob_w_tmp / self.num_particles
    #     return torch.log(prob_w)

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
        sum_elbo = 0.
        # sum_log_prob = 0.
        for doc in documents:
            doc_gamma, doc_phi = self.cavi(doc, keep_elbo=False)
            gamma_list.append(doc_gamma.detach().numpy())
            phi_list.append(doc_phi.detach())
            _doc = self.encoding_doc(tuple(doc))
            sum_elbo += self.calc_elbo(_doc)
            # sum_log_prob += self.calc_log_prob(_doc)
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

    def estimate_params(self, documents: np.array, tol: float = 5e-4,
                        max_iter: int = 500, show_step=1):
        w_list = self.gen_w_matrix(documents)
        beta_diff_norm, alpha_diff_norm, _iter = 1, 1, 0
        while beta_diff_norm + alpha_diff_norm > tol and _iter <= max_iter:
            old_alpha = self.alpha.detach()
            old_beta = self.beta.detach()
            elb, gamma_list, phi_list = self._estimate_params_e(documents)
            self._estimate_params_m(phi_list, gamma_list, w_list)
            beta_diff_norm = torch.norm(self.beta - old_beta)
            alpha_diff_norm = torch.norm(self.alpha - old_alpha)
            _iter += 1
            self.trace_elbo.append(elb)
            # self.trace_log_prob.append(prob)

            # if validate_data is not None:
            #     valid_prob = self._validate(validate_data)
            #     self.trace_validate_prob.append(valid_prob)

            if _iter % show_step == 0:
                if len(self.trace_validate_prob) == 0:
                    print("Step {_iter} | beta diff norm={beta_diff_norm} |"
                          " alpha diff norm={alpha_diff_norm} | ELBO={elb}".
                          format(_iter=_iter,
                                 beta_diff_norm=round(float(beta_diff_norm),
                                                      4),
                                 alpha_diff_norm=round(float(alpha_diff_norm),
                                                       4),
                                 elb=round(float(elb), 3)))

                # if len(self.trace_validate_prob) > 0:
                #     valid_prob = self.trace_validate_prob[-1]
                #     print("Step {_iter} | beta diff norm={beta_diff_norm} |"
                #           " alpha diff norm={alpha_diff_norm} | ELBO={elb} "
                #           "| Log_prob={prob} | Validate_prob={valid_prob}"
                #           .format(_iter=_iter,
                #                   beta_diff_norm=round(float(beta_diff_norm),
                #                                        4),
                #                   alpha_diff_norm=round(float(alpha_diff_norm),
                #                                         4),
                #                   elb=round(float(elb), 3),
                #                   prob=round(float(prob), 3),
                #                   valid_prob=round(float(valid_prob), 3)))

        return self.alpha, self.beta

    # def _validate(self, validate_data: np.array):
    #     prob = 0.
    #     for doc in validate_data:
    #         prob += self.calc_log_prob(
    #             self.encoding_doc(tuple(doc)))
    #     return prob
