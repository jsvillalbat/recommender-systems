import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

# Load in the data
import os
if not os.path.exists('user2movie.json') or \
    not os.path.exists('movie2user.json') or \
    not os.path.exists('usermovie2rating.json') or \
    not os.path.exists('usermovie2rating_test.json'):
    import preprocess2dict


with open('user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)

with open('movie2user.json', 'rb') as f:
    movie2user = pickle.load(f)

with open('usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)

with open('usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)

N = np.max(list(user2movie.keys())) + 1
# The test set may contain movies the train set doesn't have data for
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

# Initialize variables
K = 10 # latent dimensionality
W = np.random.randn(N, K)
b = np.zeros(N)
U = np.random.randn(M, K)
c = np.zeros(M)
mu = np.mean(list(usermovie2rating.values()))

# prediction[i,j] = W[i].dot(U[j]) + b[i] + c.T[j] + mu

def get_loss(d):
    # d: (user_id, movie_id) -> rating
    N = float(len(d))
    sse = 0
    for k, r in d.items():
        i, j = k
        p = W[i].dot(U[j]) + b[i] + c[j] + mu
        sse += (p - r)*(p - r)
    return sse / N

# Train the parameters
epochs = 25
reg = 0.01
train_losses = []
test_losses = []
for epoch in range(epochs):
    print("epoch:", epoch)
    epoch_start = datetime.now()
    # Update W and b
    t0 = datetime.now()
    for i in range(N):
        # Update W[i]
        matrix = np.eye(K) * reg
        vector = np.zeros(K)

        # Update b[i]
        bi = 0
        for j in user2movie[i]:
            r = usermovie2rating[(i,j)]
            matrix += np.outer(U[j], U[j])
            vector += (r - b[i] - c[j] - mu)*U[j]
            bi += (r - W[i].dot(U[j]) - c[j] - mu)

        # Set the updates
        W[i] = np.linalg.solve(matrix, vector)
        b[i] = bi / (len(user2movie[i]) + reg)

        if i % (N//10) == 0:
            print("i:", i, "N:", N)
    print("updated W and b:", datetime.now() - t0)

    # Update U and c
    t0 = datetime.now()
    for j in range(M):
        # Update U[j]
        matrix = np.eye(K) * reg
        vector = np.zeros(K)

        # Update c[j]
        cj = 0
        try:
            for i in movie2user[j]:
                r = usermovie2rating[(i,j)]
                matrix += np.outer(W[i], W[i])
                vector += (r - b[i] - c[j] - mu)*W[i]
                cj += (r - W[i].dot(U[j]) - b[i] - mu)
        except KeyError:
            # Possible not to have any ratings for a movie
            pass

        # Set the updates
        try:
            U[j] = np.linalg.solve(matrix, vector)
            c[j] = cj / (len(movie2user[j]) + reg)
        except np.linalg.LinAlgError:
            # Possible that a movie has no ratings
            pass

        if j % (M//10) == 0:
            print("j:", j, "M:", M)
    print("updated U and c:", datetime.now() - t0)
    print("epoch duration:", datetime.now() - epoch_start)

    # store train loss
    t0 = datetime.now()
    train_losses.append(get_loss(usermovie2rating))

    # store test loss
    test_losses.append(get_loss(usermovie2rating_test))
    print("calculate cost:", datetime.now() - t0)
    print("train loss:", train_losses[-1])
    print("test loss:", test_losses[-1])


print("train losses:", train_losses)
print("test losses:", test_losses)

# plot losses
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.show()
