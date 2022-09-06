import ot
import torch

#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"


def relative_entropy(p, q):
    """
    Computes the relative Shannon entropy of two probability vectors.

    Parametres:

    p (tensor): probability vector
    q (tensor): probability vector

    Returns:

    float: relative Shannon entropy H(p||q)
    """
    div = torch.div(q, p)
    rel_ent = -torch.sum(p * torch.log(div))
    return rel_ent


def simplex_group_product(p, q):
    """
    Computes the group product of two points in the simplex.

    Parametres:

    p (tensor): point in the simplex
    q (tensor): point in the simplex

    Returns:

    array: group-theoretic product p o q
    """
    prod = p*q
    prod = torch.div(prod, torch.sum(prod))
    return prod


def simplex_group_inverse(p):
    """
    Computes the group inverse of a point in the simplex.

    Parametres:

    p (tensor): point in the simplex

    Returns:

    array: group-theoretic inverse p^{-1}
    """
    p_inv = torch.div(1., p)
    p_inv = torch.div(p_inv, torch.sum(p_inv))
    return p_inv


def simplex_cost(p, q):
    """
    Computes the transporation cost between two points in the simplex.

    Parametres:

    p (tensor): point in the simplex
    q (tensor): point in the simplex

    Returns:

    float: transportation cost c(p,q) = H(e || q o p^{-1})
    """
    n = p.shape[0]
    barycentre = torch.full((n,), 1/n).to(DEVICE)
    p_inv = simplex_group_inverse(p)
    prod = simplex_group_product(q, p_inv)
    return relative_entropy(barycentre, prod)


def dirichlet_cost(P, Q, reg=1., max_iter=1000, log=False):
    """
    Computes the Dirichlet cost of two distributions over the simplex.

    Parametres:

    P (tensor): array whose rows are points in the simplex
    Q (tensor): array whose rows are points in the simplex
    reg (float): parametre for entropic regularizer in Sinkhorn algorithm (default = 1.)
    num_iter (int): max number of iterations for Sinkhorn algorithm (default = 1000)
    log (bool): log debug data (default = False)

    Returns:

    float: Dirichlet cost inf_{R in Pi(P,Q)} E_{(p,q)~R} C(p,q)
    """
    # get empirical distributions.
    #
    # problem is that torch.unique() isnt differentiable,
    # hence we cant use this and will just use a uniform
    # probability vector with duplicate points. numerically,
    # this is identical, just less interpretable.
    #
    """P_points, P_counts = torch.unique(
        P, dim=0, return_counts=True)
    Q_points, Q_counts = torch.unique(
        Q, dim=0, return_counts=True)
    P_empirical = torch.div(P_counts, torch.sum(P_counts))
    Q_empirical = torch.div(Q_counts, torch.sum(Q_counts))
    if log:
        print("Emprical Histograms:")
        print(P_points, P_counts)
        print(Q_points, Q_counts)"""

    """num_P = P_counts.shape[0]
    num_Q = Q_counts.shape[0]"""

    num_P = P.shape[0]
    num_Q = Q.shape[0]
    P_empirical = torch.full((num_P,), 1/num_P).to(DEVICE)
    Q_empirical = torch.full((num_Q,), 1/num_Q).to(DEVICE)

    costs = torch.empty((num_P, num_Q)).to(DEVICE)
    for i in range(num_P):
        for j in range(num_Q):
            costs[i, j] = simplex_cost(P[i], Q[j])
            if log:
                print(
                    f"Moving {P[i]} to {Q[j]} costs {costs[i,j]}")
    # costs can be very small so we normalize them
    # to prevent numerical errors
    # costs /= np.min(costs)

    #transport_plan = ot.sinkhorn(P_empirical, Q_empirical, costs, reg, numItermax=max_iter)
    transport_plan = ot.emd(P_empirical, Q_empirical, costs)

    if log:
        print("Transporation Plan:")
        print(transport_plan)

    cost = torch.sum(costs * transport_plan)
    if log:
        print(f"Cost: {cost}")
    return cost
