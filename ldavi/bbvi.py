import pyro
import pyro.distributions as dist
from functools import lru_cache
import numpy as np
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from torch.distributions.constraints import positive, greater_than


class LDABBVI(object):
    def __init__(self, data: np.ndarray, corpora: list,
                 valid_data: np.ndarray, optimizer,
                 optimizer_params, n_topics=5):
        self.corpora = corpora
        self.data = [self.encoding_doc(tuple(doc)) for doc in data]
        self.valid_data = [self.encoding_doc(tuple(doc)) for doc in valid_data]
        self.vocab_size = len(corpora)
        self.n_topics = n_topics
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

    @lru_cache(maxsize=None)
    def encoding_doc(self, document: tuple) -> torch.Tensor:
        return torch.tensor(np.array([self.corpora.index(order)
                                      for order in document]))

    def model(self, data):
        with pyro.plate("topics", self.n_topics):
            alpha = pyro.sample("alpha", dist.Gamma(1. / self.n_topics, 1.))
            beta_param = torch.ones(self.vocab_size) / self.vocab_size
            betas = pyro.sample("beta", dist.Dirichlet(beta_param))

        words = []
        for d in pyro.plate("doc_loop", len(data)):
            doc = data[d]
            theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha))
            n_words = len(data[d])
            for w in pyro.plate(f"word_loop_{d}", n_words):
                z = pyro.sample(f"z{d}_{w}", dist.Categorical(theta))
                w = pyro.sample(f"w{d}_{w}", dist.Categorical(betas[z]),
                                obs=doc[w])
                words.append(w)
        return words

    def guide(self, data):
        alpha_posterior = pyro.param(
            "topic_weights_posterior",
            lambda: torch.ones(self.n_topics),
            constraint=positive)
        beta_posterior = pyro.param(
            "topic_words_posterior",
            lambda: torch.ones(self.n_topics, self.vocab_size),
            constraint=greater_than(0.5))

        with pyro.plate("topics", self.n_topics):
            alpha = pyro.sample("alpha", dist.Gamma(alpha_posterior, 1.))
            betas = pyro.sample("beta", dist.Dirichlet(beta_posterior))

        theta = None
        z = None

        for d in pyro.plate("doc_loop", len(data)):
            gamma_q = pyro.param(
                f"gamma_{d}", torch.ones(self.n_topics), constraint=positive
            )
            theta = pyro.sample(f"theta_{d}", dist.Dirichlet(gamma_q))
            nwords = len(data[d])
            for w in pyro.plate(f"word_loop_{d}", nwords):
                phi_q = pyro.param(
                    f"phi{d}_{w}", torch.ones(self.n_topics),
                    constraint=positive
                )
                z = pyro.sample(f"z{d}_{w}", dist.Categorical(phi_q))
        return theta, z, alpha, betas

    def calc_log_sum(self, data, num_particles):
        prob_w = torch.tensor(data=0., dtype=torch.float64)
        for _ in range(num_particles):
            guide_trace = pyro.poutine.trace(self.guide).get_trace(data)
            # sample latent variables from guide_trace
            model_trace = pyro.poutine.trace(
                pyro.poutine.replay(self.model,
                                    trace=guide_trace)).get_trace(data)
            model_trace.log_prob_sum()
            prob_w_tmp = torch.tensor(data=0.)
            for key in model_trace.nodes:
                if key[0] == "w":
                    prob_w_tmp += model_trace.nodes[key]["log_prob_sum"]
            prob_w_tmp = torch.tensor(float(prob_w_tmp),
                                      dtype=torch.float64)
            prob_w += prob_w_tmp
        return prob_w

    def run_svi(self, n_steps=100,
                num_particles=10, clear_params=False):
        if not clear_params:
            pyro.clear_param_store()
        opt = ClippedAdam(self.optimizer_params)
        svi = SVI(self.model, self.guide, opt,
                  loss=Trace_ELBO(num_particles=num_particles))
        loss = []
        pred_prob = []
        valid_prob = []
        for step in range(n_steps):
            curr_loss = svi.step(self.data)
            prob = self.calc_log_sum(self.data, num_particles)
            valid_p = self.calc_log_sum(self.valid_data, num_particles)
            loss.append(curr_loss)
            pred_prob.append(prob)
            valid_prob.append(valid_p)
            if step % (n_steps // 20) == 0:
                message = '{:.0%} ({:.1f}) ({:.1f}) ({:.1f})'.format(
                    step / n_steps, curr_loss, prob, valid_p)
                print(message, end=' | ')
        return loss, pred_prob, valid_prob
