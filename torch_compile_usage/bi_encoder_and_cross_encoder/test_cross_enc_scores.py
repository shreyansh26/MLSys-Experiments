import numpy as np

simple_scores1 = np.load("outputs/simple_cross_encoder_scores_1.npy")
opt_scores1 = np.load("outputs/opt_cross_encoder_scores_1.npy")

simple_scores2 = np.load("outputs/simple_cross_encoder_scores_2.npy")
opt_scores2 = np.load("outputs/opt_cross_encoder_scores_2.npy")

print(simple_scores1.shape)
print(opt_scores1.shape)

print(simple_scores2.shape)
print(opt_scores2.shape)

print(np.abs(simple_scores1 - opt_scores1).sum())
print(np.abs(simple_scores1 - opt_scores1).mean())
print(np.abs(simple_scores1 - opt_scores1).std())
print(np.abs(simple_scores2 - opt_scores2).sum())
print(np.abs(simple_scores2 - opt_scores2).mean())
print(np.abs(simple_scores2 - opt_scores2).std())

