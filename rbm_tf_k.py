import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from datetime import datetime


def one_hot_encode(X, K):
    # input is N x D
    # output is N x D x K
    N, D = X.shape
    Y = np.zeros((N, D, K))
    for n,d in zip(*X.nonzero()):
        k = int(X[n,d]*2 - 1)
        Y[n,d,k] = 1
    return Y

def one_hot_mask(X, K):
    # input is N x D
    # output is N x D x K
    N, D = X.shape
    Y = np.zeros((N, D, K))
    # if X[n,d] == 0, there is a missing rating
    # so the mask shoould be all zeros
    # else, it should be all ones
    for n,d in zip(*X.nonzero()):
        Y[n,d,:] = 1
    return Y

one_to_ten = np.arange(10) + 1 # [1, 2, ..... , 10]
def convert_probs_to_ratings(probs):
    # probs is N x D x K
    # output is N x D matriz of predicted ratings
    # N, D, K = probs.shape
    # out = np.zeros((N, D))
    # each preditec rating is a weighted average using probabilities
    # for n in range(N):
    #     for d in range(D):
    #         out[n,d] = probs[n,d].dot(one_to_ten)
    # return out
    return probs.dot(one_to_ten)/ 2

## Definir our Loss function L
def dot1(V,W):
    # V is N x D x K (batch of visible units)
    # W is D x M (weights)
    # returns N X M (hidden layer sshape)
    return tf.tensordot(V, W, axes=[[1,2],[0,1]])

def dot2(H,W):
    # H is N x M (batch of hidden units)
    # W is D x M (weights)
    # returns N x D x K (visible layer shape)
    return tf.tensordot(H, W, axes=[[1],[2]])


class RBM:
    """
    A Restricted Boltzmann Machine to recommend movies
    """
    def __init__(self, D, M, K) -> None:
        self.D = D # input feature size
        self.M = M # hidden layer size
        self.K = K # number of ratings
        self.build(D, M, K)

    def build(self, D, M, K):
        """Function that defines the architecture of the RBM
        """
        # params
        self.W = tf.Variable(tf.random.normal(shape=(D, M)) * np.sqrt(2.0 / M))
        self.c = tf.Variable(np.zeros(M).astype(np.float32))
        self.b = tf.Variable(np.zeros((D, K)).astype(np.float32))

        # data
        self.X_in = tf.placeholder(tf.float32, shape=(None, D, K))
        self.mask = tf.placeholder(tf.float32, shape=(None, D, K))

        # conditional probabilities
        # NOTE: tf.contrib.distribuition.Bernoulli AOU has changed n Tensorflow v1.2
        V = self.X_in
        p_h_given_v = tf.nn.sigmoid(dot1(V, self.W) + self.c)
        self.p_h_given_v = p_h_given_v # save for later

        # draw a sample from p(h | v)
        r = tf.random_uniform(shape=tf.shape(p_h_given_v))
        H = tf.to_float(r < p_h_given_v)

        # draw a sample from p(v | h)
        # note: we don't have to actually do the softmax
        logits = dot2(H, self.W) + self.b
        cdist = tf.distributions.Categorical(logits=logits)
        X_sample = cdist.sample() # shape is (N, D)
        X_sample = tf.one_hot(X_sample, depth=K) # turn it into (N, D, K)
        X_sample = X_sample * self.mask # missing data has been zeroed

        # build the objective
        objective = tf.reduce_mean(self.free_energy(self.X_in)) - tf.reduce_mean(self.free_energy(X_sample))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(objective)
        # self.train_op = tf.compat.v1.train.AdamOptimizer(1e-2).minimize(objective)
        # self.train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(objective)
        
        # build the cost
        logiits = self.forward_logits(self.X_in)
        self.cost = tf.reduce_sum(
            input_tensor=tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(self.X_in),
                logits=logiits
            )
        )

        # to get the output
        self.output_visible = self.forward_output(self.X_in)

        initop = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session()
        self.sess.run(initop)

    def fit(self, X, mask, X_test, mask_test, epochs=10, batch_sz=256, show_fig=False):
        N, D = X.shape
        n_batches = N // batch_sz

        costs = []
        test_costs = []
        for i in range(epochs):
            t0 = datetime.now()
            print(f"epoch: {i}")
            X, mask, X_test, mask_test = shuffle(X, mask, X_test, mask_test)
            for j in range(n_batches):
                x = X[j*batch_sz:(j*batch_sz + batch_sz)].toarray()
                m = mask[j*batch_sz:(j*batch_sz + batch_sz)].toarray()

                # both visiible units and mas have to be in one-hot encoded form
                # N X D -> N x D x K
                batch_one_hot = one_hot_encode(x, self.K)
                m = one_hot_mask(m, self.K)

                _, c = self.sess.run(
                    (self.train_op, self.cost),
                    feed_dict={self.X_in: batch_one_hot, self.mask: m}
                )

                if j % 100 == 0:
                    print(f"j / n_batches: {j} / {n_batches}, cost: {c}")
            print("duration:", datetime.now() - t0)

            t0 = datetime.now()
            sse = 0
            test_sse = 0
            n = 0
            test_n = 0
            for j in range(n_batches):
                x = X[j*batch_sz:(j*batch_sz + batch_sz)].toarray()
                m = mask[j*batch_sz:(j*batch_sz + batch_sz)].toarray()

                # only visible input has to be in one-hot encoded form
                x_one_hot = one_hot_encode(x, self.K)

                probs = self.get_visible(x_one_hot)
                x_hat = convert_probs_to_ratings(probs)
                sse += (m * (x_hat - x)**2).sum()
                n += m.sum()

                # the test predictions come from the train data
                # X_test and mas_test are nly used for targets
                xt = X_test[j*batch_sz:(j*batch_sz + batch_sz)].toarray()
                mt = mask_test[j*batch_sz:(j*batch_sz + batch_sz)].toarray()

                test_sse += (mt * (x_hat - xt)**2).sum()
                test_n += mt.sum()
            c = sse / n
            ct = test_sse / test_n
            print(f"train mse: {c}, test mse: {ct}")
            print("calculate mse duration:", datetime.now() - t0)
            costs.append(c)
            test_costs.append(ct)
        if show_fig:
            plt.plot(costs, label='train mse')
            plt.plot(test_costs, label='test mse')
            plt.legend()
            plt.show()

    def free_energy(self, V):
        first_term = -tf.reduce_sum(dot1(V, self.b))
        second_term = -tf.reduce_sum(
            # tf.log(1 + tf.exp(dot1(V, self.W) + self.c)),
            tf.nn.softplus(dot1(V, self.W) + self.c),
            axis=1
        )
        return first_term + second_term
    
    def forward_hidden(self, X):
        return tf.nn.sigmoid(dot1(X, self.W) + self.c)
    
    def forward_logits(self, X):
        Z = self.fprward_hidden(X)
        return dot2(Z, self.W) + self.b
    
    def forward_output(self, X):
        return tf.nn.softmax(self.forward_logits(X))
    
    def transform(self, X):
        # accepts and returns a real numpy array
        # unlike forward_hidden and forward_output
        # which deal with tensorflow tensors
        return self.sess.run(self.hidden, feed_dict={self.X_in: X})
    
    def get_visible(self, X):
        return self.sess.run(self.output_visible, feed_dict={self.X_in: X})
    


def main():
    A = load_npz("Atrain.npz")
    A_test = load_npz("Atest.npz")
    mask = (A > 0) * 1.0
    mask_test = (A_test > 0) * 1.0

    N, M = A.shape
    rbm = RBM(M, 50, 10)
    rbm.fit(A, mask, A_test, mask_test)

if __name__ == '__main__':
    main()