# Mode-coverage metrics, for any target that carries a discrete mode descriptor

# Libraries
import math
import torch
from ..metrics.weights import normalize_weights
from .base import cycle, generator
from .kernels import random_walk_mh


def mode_entropy(hist):
    """Entropy of the empirical mode weights, normalised by log(n_states) so it lands in [0, 1]"""
    entropy = -torch.sum(torch.xlogy(hist, hist), dim=-1) / math.log(hist.shape[-1])
    return float(entropy.mean())


def kl_mode_weights(hist, weights):
    """KL(true || empirical) between the mode weights, which is +inf when a mode gets no mass"""
    return float(torch.sum(weights * torch.log(weights / hist), dim=-1).mean())


def tv_mode_weights(hist, weights):
    """L1 distance sum_k |hist_k - weights_k| between the empirical and true mode weights

    NOTE: the sum of the absolute differences is twice the total variation distance, so it lies
    in [0, 2].
    """
    return float(torch.sum(torch.abs(hist - weights), dim=-1).mean())


def forgotten_modes(hist, weights, tol=0.05):
    """Fraction of modes carrying less than tol times the smallest true weight"""
    return float((hist < tol * weights.min()).to(hist.dtype).mean())


def skewed_allocation(weights, ratio):
    """Returns the share of train chains every mode gets, far from its true weight

    Modes are ranked by true weight, heaviest first, ties keeping their index order, and the shares
    are geometric along that ranking from 1 to ratio: the heaviest mode gets the smallest share and
    the lightest the largest. Equal weights -- GMM-40, MoS, EightGaussians -- get the same skew
    along the index order, where an allocation by weight would be stratified sampling and give the
    proportions away.

    Args:
        * weights (torch.Tensor of shape (n_modes,)): True mode weights
        * ratio (float): Largest share over the smallest

    Returns:
        * share (torch.Tensor of shape (n_modes,)): Shares, summing to one
    """
    rank = torch.argsort(torch.argsort(-weights, stable=True), stable=True)
    share = ratio ** (rank.to(weights.dtype) / max(len(weights) - 1, 1))
    return share / share.sum()


class ModeCoverage:
    """Mixin giving a target with a discrete mode descriptor the mixture coverage panel

    The four numbers are the ones every mixture in the catalog reports, computed from the same
    helpers, so a rings number and a GMM-40 number mean the same thing.

    A subclass provides mode_labels(x) and a mode_weights buffer. Both may carry a GROUP axis,
    and that axis is the whole difference between the two cases in the catalog:

    * an ordinary mixture has ONE group -- labels of shape (n,), weights of shape (n_modes,) --
      and the mean over groups is the identity, so the four numbers are a flat mixture panel;
    * a product of independent double wells has one group PER WELL -- labels of shape (n, n_wells)
      over two states each -- because its 2^(d/2) joint modes can never all be populated by any
      finite sample, while the per-well marginals are estimable and are what the literature
      reports. Averaging over groups is then the per-well mean.

    Both shapes are normalised to (n_groups, n_states) here, so the four helpers above are written
    once. The empirical weights are the fraction of the samples carrying each label, except for a
    Mixture, which overrides compute_mode_weights with its mean responsibilities.

    The train recipe of a one-group target lives here too, since it only needs the modes: a
    subclass provides sample_modes and train_step_size.
    """

    train_recipe = 'skewed-modes'

    # Ratio between the largest and the smallest share of train chains a mode gets
    train_skew = 10.0

    def mode_labels(self, x):
        """Returns the state index of every sample, of shape (n,) or (n, n_groups)"""
        raise NotImplementedError('mode_labels is not implemented.')

    def sample_modes(self, modes, generator=None):
        """Draws one exact sample from each given mode, on the device of modes

        Args:
            * modes (torch.Tensor of shape (n,)): Mode index of every sample
            * generator (torch.Generator or None): RNG, on the device of modes (default is None)

        Returns:
            * x (torch.Tensor of shape (n, dim)): Exact draws, x[i] from mode modes[i]
        """
        raise NotImplementedError('sample_modes is not implemented.')

    def train_modes(self, n, generator=None):
        """Draws the mode every train chain starts in, from skewed_allocation's shares

        Args:
            * n (int): Number of chains
            * generator (torch.Generator or None): RNG, on the target's device (default is None)

        Returns:
            * modes (torch.Tensor of shape (n,)): Mode index of every chain
        """
        share = skewed_allocation(self.mode_weights.to(torch.float64), self.train_skew)
        return torch.multinomial(share, n, replacement=True, generator=generator)

    def generate_train_samples(self, n, seed=0, x0=None, n_steps=20, step_size=None):
        """Runs the skewed-modes recipe: every mode, in the wrong proportions

        Each chain draws a mode from skewed_allocation, the heaviest mode getting the fewest chains,
        starts from an exact draw of that mode, and takes n_steps random-walk Metropolis steps of
        size train_step_size. Every mode is visited with its own shape, in the proportions of the
        allocation. The steps are few so that no chain crosses to another mode and undoes them.

        Args:
            * n (int): Number of samples, one per chain
            * seed (int): Seed of the recipe (default is 0)
            * x0 (torch.Tensor of shape (m, dim) or None): Starting points, chain i starting from
                x0[i % m]; None draws them as above (default is None)
            * n_steps (int): Random-walk steps per chain (default is 20)
            * step_size (float or None): Proposal scale; None uses train_step_size
                (default is None)

        Returns:
            * x (torch.Tensor of shape (n, dim)): The biased samples
        """
        gen = generator(seed, device=self.device)
        if x0 is None:
            x0 = self.sample_modes(self.train_modes(n, generator=gen), generator=gen)
        step_size = self.train_step_size if step_size is None else step_size
        return random_walk_mh(self.log_prob, cycle(self.cast(x0), n), n_steps, step_size,
                              generator=gen)

    @property
    def true_mode_weights(self):
        """Returns mode_weights with an explicit group axis, of shape (n_groups, n_states)"""
        weights = self.mode_weights
        return weights[None] if weights.dim() == 1 else weights

    def compute_mode_weights(self, samples, weights=None):
        """Returns the empirical weight of every state, per group, summing to one within a group

        Args:
            * samples (torch.Tensor of shape (n, *data_shape)): Samples
            * weights (torch.Tensor of shape (n,)): Weights of the samples (default is None)

        Returns:
            * hist (torch.Tensor of shape (n_groups, n_states)): Empirical mode weights
        """
        index = self.mode_labels(samples.to(self.mode_weights))
        index = index[:, None] if index.dim() == 1 else index
        mass = normalize_weights(weights, n=len(index), dtype=self.dtype, device=index.device)

        # One scatter per group, the groups being the columns of the label tensor
        hist = torch.zeros(index.shape[1], self.true_mode_weights.shape[-1], dtype=self.dtype,
                           device=index.device)
        for group in range(hist.shape[0]):
            hist[group].scatter_add_(0, index[:, group], mass.to(hist))
        return hist

    def compute_mode_metrics(self, samples, weights=None, hist=None):
        """Returns the four mode-coverage metrics and the empirical weights they rest on

        Args:
            * samples (torch.Tensor of shape (n, *data_shape)): Samples
            * weights (torch.Tensor of shape (n,)): Weights of the samples (default is None)
            * hist (torch.Tensor or None): Empirical weights to reuse (default is None)

        Returns:
            * metrics (dict): 'emc', 'kl_weights', 'tv_weights', 'perc_forgotten_modes'
            * hist (torch.Tensor of shape (n_groups, n_states)): Empirical mode weights
        """
        if hist is None:
            hist = self.compute_mode_weights(samples, weights=weights)
        true_weights = self.true_mode_weights
        metrics = {
            'emc': mode_entropy(hist),
            'kl_weights': kl_mode_weights(hist, true_weights),
            'tv_weights': tv_mode_weights(hist, true_weights),
            'perc_forgotten_modes': forgotten_modes(hist, true_weights),
        }
        return metrics, hist

    def compute_metrics(self, samples, weights=None, ref_samples=None,
                        compute_standard_metrics=True, skip_costly_metrics=True, generator=None,
                        return_hist=False):
        """Adds the mode-coverage panel, whose reference is the target's own true weights

        Arguments and returns are Target.compute_metrics'; the histogram is the empirical mode
        weights, of shape (n_groups, n_states).
        """
        ret = super().compute_metrics(samples, weights=weights, ref_samples=ref_samples,
                                      compute_standard_metrics=compute_standard_metrics,
                                      skip_costly_metrics=skip_costly_metrics, generator=generator)
        samples, weights = self.cast(samples), self.cast(weights)
        weights = None if weights is None else normalize_weights(weights)
        metrics, hist = self.compute_mode_metrics(samples, weights=weights)
        ret.update(metrics)
        if return_hist:
            return ret, hist
        return ret
