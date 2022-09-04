import ot
import numpy as np


def relative_entropy(p, q):
    """
    Computes the relative Shannon entropy of two probability vectors.

    Parametres:

    p (array): probability vector
    q (array): probability vector

    Returns:

    float: relative Shannon entropy H(p||q)
    """
    return -np.sum(p * np.log(q/p))


def simplex_group_product(p, q):
    """
    Computes the group product of two points in the simplex.

    Parametres:

    p (array): point in the simplex
    q (array): point in the simplex

    Returns:

    array: group-theoretic product p o q
    """
    prod = p*q
    prod /= np.sum(prod)
    return prod


def simplex_group_inverse(p):
    """
    Computes the group inverse of a point in the simplex.

    Parametres:

    p (array): point in the simplex

    Returns:

    array: group-theoretic inverse p^{-1}
    """
    p_inv = 1. / p
    p_inv /= np.sum(p_inv)
    return p_inv


def simplex_cost(p, q):
    """
    Computes the transporation cost between two points in the simplex.

    Parametres:

    p (array): point in the simplex
    q (array): point in the simplex

    Returns:

    float: transportation cost c(p,q) = H(e || q o p^{-1})
    """
    n, = p.shape
    barycentre = np.full(n, 1/n)
    p_inv = simplex_group_inverse(p)
    prod = simplex_group_product(q, p_inv)
    return relative_entropy(barycentre, prod)


def dirichlet_cost(P, Q, reg=1., num_iter=1000, log=False):
    """
    Computes the Dirichlet cost of two distributions over the simplex.

    Parametres:

    P (array): array whose rows are points in the simplex
    Q (array): array whose rows are points in the simplex
    reg (float): parametre for entropic regularizer in Sinkhorn algorithm (default = 1.)
    num_iter (int): max number of iterations for Sinkhorn algorithm (default = 1000)
    log (bool): log debug data (default = False)

    Returns:

    float: Dirichlet cost inf_{R in Pi(P,Q)} E_{(p,q)~R} C(p,q)
    """
    # get empirical distributions
    P_points, P_counts = np.unique(P, axis=0, return_counts=True)
    Q_points, Q_counts = np.unique(Q, axis=0, return_counts=True)
    P_empirical = P_counts / np.sum(P_counts)
    Q_empirical = Q_counts / np.sum(Q_counts)
    if log:
        print("Emprical Histograms:")
        print(P_points, P_counts)
        print(Q_points, Q_counts)

    # populate transport cost matrix
    num_P, = P_counts.shape
    num_Q, = Q_counts.shape
    costs = np.zeros((num_P, num_Q))
    for i in range(num_P):
        for j in range(num_Q):
            costs[i][j] = simplex_cost(P_points[i], Q_points[j])
            if log:
                print(
                    f"Moving {P_points[i]} to {Q_points[j]} costs {costs[i][j]}")
    # costs can be very small so we normalize them
    # to prevent numerical errors
    # costs /= np.min(costs)

    transport_plan = ot.sinkhorn(
        P_empirical, Q_empirical, costs, reg, numItermax=num_iter)
    if log:
        print("Transporation Plan:")
        print(transport_plan)

    cost = np.sum(costs * transport_plan)
    if log:
        print(f"Cost: {cost}")
    return cost
