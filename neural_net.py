from dirichlet import dirichlet_cost
from dirichlet import simplex_cost
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()


class dirichlet_net(nn.Module):

    def __init__(self, layers, x_source, y_source, x_target):
        """
        Initializes Dirichlet neural net.

        Parametres:

        layers (list): sizes of each layers of the neural net, including input and out layers
        x_source (tensor): tensor whose rows correspond to data points in the source domain
        y_source (tensor): tensor whose entries correspond to true labels for rows in the source domain
        x_target (tensor): tensor whose rows correspond to data points in the target domain
        """
        super().__init__()
        self.x_source = x_source
        self.y_source = y_source
        self.x_target = x_target

        self.num_layers = len(layers)
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i-1], layers[i]) for i in range(1, self.num_layers)])
        self.num_layers -= 1
        self.apply(self.init_values)

    def init_values(self, module):
        """
        Initializes the weights and biases of each layer in the network.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            module.bias.data.normal_(mean=0.0, std=1.0)

    def forward(self, x):
        """
        Forward propogates in neural network.

        Parametres:

        x (tensor): tensor whose rows correspond to data to propogate

        Returns:

        tensor: tensor whose rows correspond to classification hypotheses
        """
        out = x
        for i in range(self.num_layers - 1):
            out = F.sigmoid(self.layers[i](out))
        out = F.softmax(self.layers[-1](out), dim=1)
        return out

    def train(self, learning_rate=4e-3, num_iter=10000, batch_size=32, log=False):
        """
        Trains neural network using Adam and cross-entropy loss.

        learning_rate (float): learning rate for Adam (default = 1e-4)
        num_iter (int): number of iterations to perform before finishing training (default = 10000)
        batch_size (int): number of points to sample per iteration (default = 32)
        log (bool): print debug logging (default = False)
        """

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # precompute permutations since this step is slow
        num_source = self.x_source.shape[0]
        perm_source = torch.randperm(num_source)
        window = 0

        if log:
            start_time = timeit.default_timer()
        for iter in range(num_iter):

            # only recompute permutations if window will overflow
            if window*batch_size + batch_size >= num_source:
                perm_source = torch.randperm(num_source)
                window = 0

            # window which slides down permutation
            perm_source_window = perm_source[window *
                                             batch_size:window*batch_size+batch_size]
            window += 1

            x_source_batch = self.x_source[perm_source_window]
            y_source_batch = self.y_source[perm_source_window].long()
            source_pred = self(x_source_batch)

            loss = F.cross_entropy(source_pred, y_source_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if log and (iter % 100 == 0):
                elapsed = timeit.default_timer() - start_time
                print(
                    f"iter {iter}: loss = {loss}, time = {elapsed}")

                start_time = timeit.default_timer()

    def train_dirichlet(self, comparison, learning_rate=4e-3, eta=1., nu=1., gamma=1., num_iter=10000, reg=1., max_iter_sinkhorn=10000, batch_size=32, log=False):
        """
        Trains neural network using Adam.

        Parametres:

        comparison (model): trained neural network used for comparison
        learning_rate (float): learning rate for Adam (default = 1e-4)
        nu (float): hyperparametre on the comparison drift (default = 1.)
        eta (float): hyperparametre on domain divergence (default = 1.)
        gamma (float): hyperparametre on the source risk (default = 1.)
        num_iter (int): number of iterations to perform before finishing training (default = 10000)
        reg (float): entropic regularizing parametre for Sinkhorn (default = 1.)
        max_iter_sinkhorn (int): max number of iterations to do in Sinkhorn (default = 10000)
        batch_size (int): number of points to sample per iteration (default = 32)
        log (bool): print debug logging (default = False)
        """

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # precompute permutations since this step is slow
        num_source = self.x_source.shape[0]
        num_target = self.x_target.shape[0]
        perm_source = torch.randperm(num_source)
        perm_target = torch.randperm(num_target)
        window = 0

        if log:
            start_time = timeit.default_timer()
        for iter in range(num_iter):

            # only recompute permutations if window will overflow
            if window*batch_size + batch_size >= min(num_source, num_target):
                perm_source = torch.randperm(num_source)
                perm_target = torch.randperm(num_target)
                window = 0

            # window which slides down permutation
            perm_source_window = perm_source[window *
                                             batch_size:window*batch_size+batch_size]
            perm_target_window = perm_target[window *
                                             batch_size:window*batch_size+batch_size]
            window += 1

            x_source_batch = self.x_source[perm_source_window]
            y_source_batch = self.y_source[perm_source_window].long()
            x_target_batch = self.x_target[perm_target_window]

            source_pred_comparison = comparison(x_source_batch)
            target_pred_comparison = comparison(x_target_batch)
            source_pred = self(x_source_batch)
            target_pred = self(x_target_batch)

            comparison_drift = eta * dirichlet_cost(
                source_pred, source_pred_comparison, reg=reg, max_iter=max_iter_sinkhorn)

            domain_divergence = nu * \
                dirichlet_cost(target_pred, source_pred,
                               reg=reg, max_iter=max_iter_sinkhorn)

            source_risk = gamma * \
                F.cross_entropy(source_pred, y_source_batch)

            loss = comparison_drift + domain_divergence + source_risk
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if log and (iter % 100 == 0):
                elapsed = timeit.default_timer() - start_time
                print(
                    f"iter {iter}: loss = {loss}, source risk = N/A, comparison drift = {comparison_drift}, domain divergence = N/A, time = {elapsed}")

                start_time = timeit.default_timer()

    def test(self, x_test, y_test, log=False):
        """
            Tests neural network given labelled data.

            Parametres:

            x_test (tensor): tensor whose rows are the data points
            y_test (tensor): tensor whose entries correspond to true labels for rows
            log (bool): print debug logs (default = False)

            Returns:

            float: percentage of successful predictions
        """
        with torch.no_grad():
            num_pred = x_test.shape[0]
            pred = self(x_test)
            pred_labels = torch.argmax(pred, dim=1)
            hits = torch.sum(pred_labels == y_test)
            success = hits / num_pred
            if log:
                print(f"predictions:")
                print(pred_labels)
                print(f"results: {hits}/{num_pred}, {success*100}%.")
            return hits / num_pred
