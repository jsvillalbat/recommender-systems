import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit(object):
    def __init__(self, p):
        self.p = p # true succes rate
        self.a = 1 # beta parameter prior
        self.b = 1 # beta parameter prior

    def pull(self):
        # draw a 1 with probability p from the Bernoulli distribution
        # Draws a sample from the true distribution
        return np.random.random() < self.p

    def sample(self):
        # draw a sample from the beta distribution
        # Draws a sample from the estimated distribution (posterior distribution)
        return np.random.beta(self.a, self.b)

    def update(self, x):
        # update the beta distribution given the observation(sample) x
        self.a += x
        self.b += 1 - x

def plot(bandits, trial):
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label="real p: %.4f" % b.p)
    plt.title("Bandit distributions after %s trials" % trial)
    plt.legend()
    plt.show()


def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
    for i in range(NUM_TRIALS):

        # take a sample from each bandit
        bestb = None
        maxsample = -1
        allsamples = []
        for b in bandits:
            sample = b.sample()
            allsamples.append("%.2f" % sample)
            if sample > maxsample:
                maxsample = sample
                bestb = b
        if i in sample_points:
            print("current samples: %s" % allsamples)
            plot(bandits, i)

        # pull the arm for the bandit with the largest sample
        x = bestb.pull()

        # update the distribution for the bandit whose arm we just pulled
        bestb.update(x)

if __name__ == "__main__":
    experiment()
