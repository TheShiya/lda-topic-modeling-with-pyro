import pyro
import torch
import pyro.distributions as dist
from torch.distributions.constraints import positive
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class LDABBVI(object):

    def __init__(self, data, vocabulary, ntopics=5,
                 optimizer=Adam, optimizer_params={}):
        self.n_topics = ntopics
        self.n_words = data.shape[0]
        self.n_docs = data.shape[1]
        self.vocab_size = len(set(vocabulary))
        self.data = data
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

    def lda_model(self):
        alpha_prior = torch.randint(1, 10, (self.n_topics,)).double()
        beta_prior = torch.ones(self.n_topics, self.vocab_size)
        words = []
        with pyro.plate("doc_loop", self.n_docs) as ind:
            data = self.data[:, ind]
            theta = pyro.sample("theta", dist.Dirichlet(alpha_prior))
            with pyro.plate("word_loop", self.n_words):
                z = pyro.sample("z", dist.Categorical(theta))
                w = pyro.sample("w", dist.Categorical(beta_prior[z]), obs=data)
                words.append(w)
        return words

    def guide(self):
        gamma_q = pyro.param(
            "gamma_q", torch.ones(self.n_docs, self.n_topics),
            constraint=positive
        )
        phi_q = pyro.param(
            "phi_q", torch.ones(self.n_words, self.n_docs, self.n_topics),
            constraint=positive
        )
        with pyro.plate("doc_loop", self.n_docs):
            theta = pyro.sample("theta", dist.Dirichlet(gamma_q))
            with pyro.plate("word_loop", self.n_words):
                z = pyro.sample("z", dist.Categorical(phi_q))
        return theta, z

    def generate(self, alpha_prior, beta_prior):
        data = torch.zeros([self.n_words, self.n_docs])
        topics = beta_prior
        for d in pyro.plate("doc_loop", self.n_docs):
            theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha_prior))
            for w in pyro.iarange("word_loop", self.n_words):
                z = pyro.sample(f"z{d}_{w}",
                                dist.Categorical(theta))
                word = pyro.sample(f"w{d}_{w}",
                                   dist.Categorical(topics[z.item()]))
                data[w, d] = word
        return data, topics

    def run_svi(self, data, n_steps=100, batch_size=1, clear_params=False):
        if not clear_params:
            pyro.clear_param_store()
        loss_func = Trace_ELBO(num_particles=batch_size)
        svi = SVI(self.model, self.guide, self.optimizer, loss=loss_func)
        loss = []
        for step in range(n_steps):
            curr_loss = svi.step(data)
            loss.append(curr_loss)
            if step % (n_steps // 100) == 0:
                print('{} - {}'.format(n_steps // step * 100, curr_loss))
        return loss
