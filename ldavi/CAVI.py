import numpy as np
import torch
import pyro.distributions as dist


class LDACAVI(object):
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor,
                 corpora: np.ndarray, num_topics: int, tol: float = 1e-4,
                 max_step: int = 5000, show_step: int = 2):
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
        self.max_step = max_step
        self.tol = tol
        self.show_step = show_step

    def doc_cavi(self, document: np.ndarray):
        self.gamma = torch.tensor(data=self.alpha.detach().
                                  numpy() + len(document) / self.num_topics,
                                  requires_grad=True)
        self.phi = torch.tensor(data=np.full(shape=(len(document),
                                                    self.num_topics),
                                             fill_value=1 / self.num_topics))
        num_step = 0
        gamma_diff_norm = 1
        phi_diff_norm = 1
        while num_step <= self.max_step and (gamma_diff_norm + phi_diff_norm) >= self.tol:
            old_gamma = self.gamma.clone().detach()
            old_phi = self.phi.clone().detach()
            for word_ind in range(len(document)):
                cur_word = document[word_ind]
                word_corpora_ind = np.where(self.corpora == cur_word)[0]
                for topic in range(self.num_topics):
                    tmp_sum_gamma = torch.sum(self.gamma)
                    tmp_sum_gamma.retain_grad = True
                    tmp_exp = torch.lgamma(self.gamma[topic]).grad_fn(self.gamma[topic])
                    tmp_exp -= torch.lgamma(tmp_sum_gamma).grad_fn(tmp_sum_gamma)
                    self.phi[word_ind][topic] = self.beta[topic][word_corpora_ind] * tmp_exp
                self.phi[word_ind, :] = self.phi[word_ind, :] / self.phi[word_ind, :].sum()
            self.gamma = self.alpha + torch.sum(self.phi, dim=0)
            gamma_diff_norm = torch.norm(self.gamma - old_gamma)
            phi_diff_norm = torch.norm(self.phi - old_phi)
            if num_step % self.show_step == 0:
                print(f"Step {num_step} | phi diff norm={phi_diff_norm} | gamma diff norm={gamma_diff_norm}")
            num_step += 1

        return self.gamma, self.phi


if __name__ == "__main__":
    corpora = np.array(["a", "b", "c", "d", "e", "f"])
    num_topics = 2
    alpha = torch.tensor(data=[20., 10.])
    beta = torch.tensor(data=[[0.1, 0.2, 0.3, 0.2, 0.1, 0.1], [0.2, 0.1, 0.1, 0.1, 0.2, 0.4]])
    topic = dist.DirichletMultinomial(alpha)()
    beta_p = torch.matmul(topic, beta)
    words = torch.distributions.multinomial.Multinomial(total_count=20, probs=beta_p).sample()
    # A document with 20 words
    document = []
    for i in range(len(words)):
        document += [corpora[i]] * int(words[i])
    obj = LDACAVI(alpha=alpha, beta=beta, corpora=corpora, num_topics=num_topics)
    gamma, phi = obj.doc_cavi(np.array(document))
    print(gamma, phi)
