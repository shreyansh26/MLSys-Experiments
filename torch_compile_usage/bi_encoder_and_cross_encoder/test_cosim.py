import numpy as np

st_simple_embeddings = np.load("outputs/basic_simple_embeddings.npy")
simple_embeddings = np.load("outputs/simple_embeddings.npy")
opt_embeddings = np.load("outputs/opt_embeddings.npy")

print(simple_embeddings.shape)
print(opt_embeddings.shape)

cosine_sim = np.dot(simple_embeddings, st_simple_embeddings.T) / (np.linalg.norm(simple_embeddings, axis=1) * np.linalg.norm(st_simple_embeddings, axis=1))

print("Cosine similarity between ST and non-ST embeddings:")
print(cosine_sim.shape)

print(cosine_sim.diagonal())
print(cosine_sim.diagonal().sum(), cosine_sim.shape[0])


cosine_sim = np.dot(simple_embeddings, opt_embeddings.T) / (np.linalg.norm(simple_embeddings, axis=1) * np.linalg.norm(opt_embeddings, axis=1))

print("Cosine similarity between non-ST and OPT embeddings:")
print(cosine_sim.shape)

print(cosine_sim.diagonal())
print(cosine_sim.diagonal().sum(), cosine_sim.shape[0])
