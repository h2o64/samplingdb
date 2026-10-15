# Mode-coverage metrics, for any target that carries a discrete mode descriptor

# Libraries
import math
import torch
from ..metrics.weights import normalize_weights


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
    """

    def mode_labels(self, x):
        """Returns the state index of every sample, of shape (n,) or (n, n_groups)"""
        raise NotImplementedError('mode_labels is not implemented.')

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
