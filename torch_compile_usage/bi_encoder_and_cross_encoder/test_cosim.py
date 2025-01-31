import numpy as np

simple_embeddings = np.load("outputs/simple_embeddings.npy")
opt_embeddings = np.load("outputs/opt_embeddings.npy")

print(simple_embeddings.shape)
print(opt_embeddings.shape)

# Get cosine similarity between all pairs of embeddings corresponding to the same sentence
cosine_sim = np.dot(simple_embeddings, opt_embeddings.T) / (np.linalg.norm(simple_embeddings, axis=1) * np.linalg.norm(opt_embeddings, axis=1))

print(cosine_sim.shape)

print(cosine_sim.diagonal())
# [0.99999964 0.9999995  0.9999993  ... 0.9999996  0.99999887 0.99999887]

print(cosine_sim.diagonal().sum(), cosine_sim.shape[0])
# 4095.9976 4096
