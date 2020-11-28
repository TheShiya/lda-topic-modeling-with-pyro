import pyro
import torch
import pyro.distributions as dist
from torch.distributions.constraints import positive
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam


class LDABBVI(object):

    def __init__(self, data, optimizer, n_topics=5, optimizer_params={}):
        enum = enumerate(set("|".join(data.apply("|".join)).split("|")))
        name2id = {v: i for i, v in enum}
        data = data.apply(lambda x: torch.FloatTensor([name2id[n] for n in x]))

        self.vocab_size = len(name2id)
        self.n_topics = n_topics
        self.n_docs = len(data)
        self.data = data.values
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

    def model(self, data):
        alpha = torch.randint(1, 10, (self.n_topics,)).double()
        beta = torch.ones(self.n_topics, self.vocab_size)
        words = []
        for d in pyro.plate("doc_loop", self.n_docs):
            doc = data[d]
            theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha))
            nwords = len(data[d])
            for w in pyro.plate(f"word_loop_{d}", nwords):
                z = pyro.sample(f"z{d}_{w}", dist.Categorical(theta))
                w = pyro.sample(f"w{d}_{w}", dist.Categorical(beta[z]),
                                obs=doc[w])
                words.append(w)
        return words

    def guide(self, data):
        for d in pyro.plate("doc_loop", self.n_docs):
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

    def run_svi(self, n_steps=100, num_particles=1, clear_params=False):
        if not clear_params:
            pyro.clear_param_store()
        opt = ClippedAdam(self.optimizer_params)
        svi = SVI(self.model, self.guide, opt,
                  loss=Trace_ELBO(num_particles=num_particles))
        loss = []
        for step in range(n_steps):
            curr_loss = svi.step(self.data)
            loss.append(curr_loss)
            if step % (n_steps // 20) == 0:
                message = '{:.0%} ({:.1f})'.format(step / n_steps, curr_loss)
                print(message, end=' | ')
        return loss
