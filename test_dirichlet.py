import torch
from dirichlet import dirichlet_cost

print("\nDistributions with Identical and Duplicate Points")
print("----------\n")
P = torch.tensor([
    [0.5, 0.5],
    [0.7, 0.3],
    [0.2, 0.8]
])
Q = torch.tensor([
    [0.1, 0.9],
    [0.4, 0.6],
    [0.2, 0.8]
])
dirichlet_cost(P, Q, log=True)

print("\nSimilar Distributions")
print("----------\n")
P = torch.tensor([
    [0.25, 0.25, 0.25, 0.25],
    [0.4, 0.4, 0.1, 0.1],
    [0.3, 0.4, 0.2, 0.1]
])
Q = torch.tensor([
    [0.23, 0.27, 0.24, 0.26],
    [0.36, 0.44, 0.09, 0.11],
    [0.32, 0.38, 0.21, 0.09]
])
dirichlet_cost(P, Q, log=True)

print("\nDissimilar Distributions")
print("----------\n")
P = torch.tensor([
    [0.2, 0.8],
    [0.1, 0.9],
    [0.05, 0.95]
])
Q = torch.tensor([
    [0.5, 0.5],
    [0.45, 0.55],
    [0.48, 0.52]
])
dirichlet_cost(P, Q, log=True)
