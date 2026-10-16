# Base class for unnormalised sampling targets

# Libraries
import hashlib
import torch
from ..data import cache_dir, cached_local
from ..metrics.weights import normalize_weights
from ..metrics.standard import compute_standard_metrics as standard_metrics

# Samples in a split when the caller does not ask for a size. Both splits are this big, and both
# accept any n: pass a larger one and the extra samples are generated (or read) on the spot.
DEFAULT_N = 100000


class Target(torch.nn.Module):
    """Base class for an unnormalised sampling target

    Every subclass implements log_prob and score ANALYTICALLY -- there is no autograd fallback
    here, because a score built by differentiating through log_prob is both slower and a worse
    reference than the derivative written out by hand. score_autograd() below exists only so the
    test suite can check the analytic expression against autograd.

    NOTE: a target carries the density and nothing about how to sample it. There is no reference
    distribution and no tempered score here: an annealed sampler owns its own interpolation, and
    prior_scale is the one hint the target offers about what scale a Gaussian reference would
    need. Building beta * score(x) + (1 - beta) * grad log reference(x) is three lines in the
    sampler, and only the sampler knows which reference it is annealing from.

    Class attributes carry the metadata a metric needs to know what it is allowed to compute:

    * has_ground_truth (bool or str): True, False, or 'approximate' when a reference ensemble
        exists but is not an equilibrium one
    * truth_tier (int): Ground-truth cost tier, 0 for a closed-form density up to 4 for one whose
        reference ensemble costs core-months
    * n_modes (int or None): Total number of modes, when the target has a mode descriptor
    * default_n (int): Size of either split when n is not given
    * standard_panel (bool): Whether the target-agnostic coordinate-space panel means anything
        here; see compute_metrics
    * reference_folder (str or None): Cache sub-directory of the generator that writes the
        reference set, None for a target that has no generated one
    """

    dist = None
    has_ground_truth = True
    truth_tier = 0
    n_modes = None
    default_n = DEFAULT_N
    standard_panel = False
    reference_folder = None

    # Scale a Gaussian reference distribution should use for this target, for a sampler that
    # anneals towards it. Not used by the density itself
    prior_scale = 1.0

    # Elements a single log_prob or score call may hold in its temporaries, which sets the batch
    # size chunked() splits at. 2 ** 27 float64 is a gigabyte; every target in the corpus is
    # cheap enough at that budget that a sampler can hand in as many walkers as it likes.
    max_elements = 2 ** 27

    def __init__(self, dim, name, log_Z=None, temperature=None, dtype=torch.float64):
        """Builds the target

        Args:
            * dim (int): Dimension of x
            * name (str): Short lowercase identifier, carrying only what changes the density --
                'lj13', 'seeds', 'ka-binary-T0.2'. Cache file names are built from it
            * log_Z (float or None): log-normaliser, if known (default is None)
            * temperature (float or None): Physical temperature folded into log_gamma,
                for targets where it matters (default is None)
            * dtype (torch.dtype): Working dtype; float64 by default because the Funnel and
                LJ energies overflow in float32 (default is torch.float64)
        """
        # Call the parent constructor
        super().__init__()

        # 1. Identity of the target
        self.dim = dim
        self.name = name
        self.log_Z = log_Z
        self.temperature = temperature
        self.dtype = dtype

        # 2. A dummy buffer, so .to(device) moves the target and self.device is meaningful
        self.register_buffer('_device_probe', torch.zeros((), dtype=dtype))

    @property
    def device(self):
        """Returns the device the target currently lives on"""
        return self._device_probe.device

    @property
    def data_shape(self):
        """Returns the shape of a single sample"""
        return (self.dim,)

    def build_dist(self):
        """Builds the inner dist object"""
        raise NotImplementedError('build_dist is not implemented.')

    def log_prob(self, x):
        """Evaluates log gamma(x), the UNNORMALISED log density"""
        raise NotImplementedError('log_prob is not implemented.')

    def score(self, x):
        """Evaluates grad log gamma(x), analytically"""
        raise NotImplementedError('score is not implemented.')

    def log_prob_and_grad(self, x):
        """Evaluates the log-likelihood and the score of the target at x"""
        return self.log_prob(x), self.score(x)

    def chunked(self, fn, x, cost=1):
        """Applies fn to x in slices whose temporaries stay inside max_elements

        Densities whose cost per configuration is large -- an all-pairs tensor, a per-mode
        Gaussian -- would otherwise decide for the caller how many points may be scored at once.
        Wrapping them here means log_prob and score take any batch size at any dimension and a
        sampler never has to loop itself. Slicing is exact rather than approximate: the result is
        whatever fn(x) would have returned.

        Args:
            * fn (function): Maps a slice of x to a tensor, or to a tuple of tensors, whose
                leading dimension matches the slice
            * x (torch.Tensor of shape (n, *data_shape)): Evaluation points
            * cost (int): Elements the temporaries of fn hold per point (default is 1)

        Returns:
            * out (torch.Tensor or tuple of torch.Tensor): fn applied to all of x
        """
        chunk_size = self.chunk_size(cost)
        if len(x) <= chunk_size:
            return fn(x)

        parts = [fn(x[start:start + chunk_size]) for start in range(0, len(x), chunk_size)]
        if isinstance(parts[0], tuple):
            return tuple(torch.cat(block) for block in zip(*parts))
        return torch.cat(parts)

    def chunk_size(self, cost=1):
        """Returns the batch size whose temporaries stay inside max_elements at this cost

        Exposed because a reduction accumulates rather than concatenates -- a histogram over a
        whole reference set has no per-sample result to cat -- so it loops over the slices itself.

        Args:
            * cost (int): Elements the temporaries hold per point (default is 1)

        Returns:
            * chunk_size (int): Points per slice
        """
        return max(self.max_elements // max(cost, 1), 1)

    def sample(self, sample_shape, generator=None):
        """Returns samples from the target of shape sample_shape, where exact sampling exists"""
        raise NotImplementedError('sample is not implemented.')

    def spec(self):
        """Returns the parameters that define this density, which version its cached splits

        The name alone does not version a density: GMM-40 at dim 2 and at dim 50 are two different
        targets under one name, as are two Funnels of different scale, and a split cached under the
        name would be read back for either. Every subclass adds whatever its constructor accepts
        that changes log_gamma -- a mean matrix, a set of pair constants, a prior width -- and
        train_cache_key hashes the result, so a spec change surfaces as a cache miss rather than as
        the wrong array.

        Returns:
            * spec (dict): Density-defining values, each a number, a string or a tensor
        """
        return {'name': self.name, 'dim': self.dim}

    def train_samples(self, n=None):
        """Returns the bundled biased/cheap sample set, if one exists for this target

        For host-generated splits this runs generate_train_samples() once and caches the
        result under a spec-keyed name.

        NOTE: n is not capped. Asking for more than the cached split holds regenerates it at that
        size, under its own cache entry, so a longer run is one argument away:

            x = target.train_samples(n=1000000)

        Args:
            * n (int or None): Number of samples; None uses default_n (default is None)

        Returns:
            * x (torch.Tensor of shape (n, *data_shape)): The biased/cheap samples
        """
        if n is None:
            n = self.default_n
        array = cached_local(self.train_cache_key(n),
                             lambda: self.generate_train_samples(n).cpu().numpy())
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    def train_cache_key(self, n):
        """Returns the cache name of the host-generated train split, keyed by spec() and n

        Targets whose RECIPE carries a knob the density spec does not cover must append it here,
        so two settings cannot collide on one cache entry.
        """
        return 'train_' + self.name + '_' + spec_id(self.spec()) + '_n' + str(n)

    def reference_key(self):
        """Returns the file stem of the generated reference set, its name and spec_id

        A reference set is versioned the way a train split is: the stem carries spec_id, so a
        density change is a cache miss and a new published file rather than the old set read back.
        """
        return self.name + '_' + spec_id(self.spec())

    def reference_path(self):
        """Returns the cache path of the generated reference set, whether or not it is there"""
        return cache_dir() / self.reference_folder / (self.reference_key() + '.npy')

    def generate_train_samples(self, n):
        """Runs this target's documented locally-correct-globally-short recipe"""
        raise NotImplementedError('generate_train_samples is not implemented.')

    def test_samples(self, n=None, seed=None):
        """Returns the bundled ground-truth sample set, if one exists for this target"""
        raise NotImplementedError('test_samples is not implemented.')

    def preprocess(self, x):
        """Maps samples into the representation metrics should compare in

        Identity by default; particle targets override it with remove_mean, because their
        energy is translation-invariant and a raw-coordinate metric would not be.
        """
        return x

    def mode_labels(self, x):
        """Returns an integer mode label per sample, for coverage metrics"""
        raise NotImplementedError('mode_labels is not implemented.')

    def compute_metrics(self, samples, weights=None, ref_samples=None,
                        compute_standard_metrics=True, skip_costly_metrics=True, generator=None,
                        return_hist=False):
        """Compute every metric this target reports

        Every metric here compares SAMPLES against a reference ensemble. Nothing scores a model's
        own density, and nothing is derived from the importance weights alone.

        This is the target-agnostic half, and it runs only where standard_panel says it means
        something. A subclass OVERRIDES this method, calls it through super() for whatever the
        base contributes, and adds the panel its own density is read on -- so the metrics a target
        reports are always one method, under one name, whatever tier it belongs to.

        WARNING: the target-agnostic half is the SYNTHETIC tier's panel. A sliced KS or a
        transport distance between raw coordinates presumes the coordinates are directly
        comparable, and for the structured targets they are not: a particle configuration is
        defined up to a translation, a glass up to a translation, a permutation and a periodic
        image, and a peptide up to a rigid motion. Those targets replace it entirely with their
        own invariant panel -- distances, energies, torsions, g(r) -- which is the number their
        literature quotes anyway. The machinery still lives here, so
        metrics.compute_standard_metrics remains callable directly on anything.

        Args:
            * samples (torch.Tensor of shape (batch_size, *data_shape)): Samples to compare against
            * weights (torch.Tensor of shape (batch_size,)): Weights of the samples, not
                necessarily normalized (default is None)
            * ref_samples (torch.Tensor of same shape as samples): Reference samples; None draws
                or reads this target's own (default is None)
            * compute_standard_metrics (bool): Whether to compute the target-agnostic panel, where
                this target has one (default is True)
            * skip_costly_metrics (bool): Whether to skip the quadratic metrics -- the MMD and the
                two OT solvers (default is True)
            * generator (torch.Generator or None): RNG of the reference draw and of the random
                projections (default is None)
            * return_hist (bool): Whether to also return the raw histograms this target's panel
                rests on (default is False)

        Returns:
            * metrics (dict): All the computed metrics
            if return_hist:
                * hist: The raw histograms, None for a target with no histogram panel
        """
        weights = None if weights is None else normalize_weights(weights)
        ret = {}

        if compute_standard_metrics and self.standard_panel:
            ref_samples = self.get_ref_samples(samples, ref_samples)
            ret.update(standard_metrics(
                self.preprocess(samples).view(samples.shape[0], -1),
                self.preprocess(ref_samples).view(ref_samples.shape[0], -1),
                weights=weights, skip_costly_metrics=skip_costly_metrics, generator=generator,
            ))
        if return_hist:
            return ret, None
        return ret

    def get_ref_samples(self, samples, ref_samples=None):
        """Returns ground-truth samples matching the batch size of samples

        NOTE: the drawn reference is seeded, so a reported metric is reproducible. Pass
        ref_samples explicitly to score against a different or larger reference set.

        Args:
            * samples (torch.Tensor of shape (batch_size, *data_shape)): The scored samples
            * ref_samples (torch.Tensor or None): Reference samples to reuse (default is None)

        Returns:
            * ref_samples (torch.Tensor of shape (batch_size, *data_shape)): Reference samples
        """
        if ref_samples is not None:
            return ref_samples.to(samples.device)
        if self.has_ground_truth is False:
            raise NotImplementedError(type(self).__name__ + ' has no ground-truth sample set.')
        return self.test_samples(n=samples.shape[0], seed=0).to(samples.device)

    def _apply(self, fn):
        """Builds the distribution again when using _apply"""
        new_self = super(Target, self)._apply(fn)
        if new_self.dist is not None:
            new_self.build_dist()
        return new_self


class ExactTarget(Target):
    """A target whose exact samples are free, so test_samples() is just sample()

    This is also exactly the tier the target-agnostic panel applies to: an exact draw in the same
    coordinates log_prob is written in, with no invariance standing between the two sample sets,
    which is what makes a raw-coordinate KS or transport distance meaningful (see
    Target.compute_metrics).
    """

    standard_panel = True

    def test_samples(self, n=None, seed=0):
        """Draws the ground-truth sample set

        NOTE: seed defaults to 0 rather than None so that a reported metric is reproducible;
        pass seed=None for fresh draws. Exact draws cost nothing to make, so any n works:

            x = target.test_samples(n=1000000)

        Args:
            * n (int or None): Number of samples; None uses default_n (default is None)
            * seed (int or None): RNG seed (default is 0)

        Returns:
            * x (torch.Tensor of shape (n, *data_shape)): Exact samples
        """
        if n is None:
            n = self.default_n
        return self.sample((n,), generator=generator(seed))


def generator(seed, device=None):
    """Returns a seeded torch.Generator, or None when seed is None"""
    if seed is None:
        return None
    gen = torch.Generator(device='cpu' if device is None else device)
    gen.manual_seed(seed)
    return gen


def spec_id(spec):
    """Returns the short digest of a density specification

    Args:
        * spec (dict): Density-defining values, as returned by Target.spec

    Returns:
        * digest (str): First eight hex digits of the sha256 of the specification
    """
    digest = hashlib.sha256()
    for name in sorted(spec):
        value = spec[name]
        digest.update(name.encode())
        if torch.is_tensor(value):
            digest.update(value.detach().cpu().to(torch.float64).numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()[:8]


def score_autograd(target, x):
    """Evaluates grad log gamma(x) by autograd, as a reference for the analytic score

    NOTE: this is a test utility. Nothing in the library calls it, and no sampler should --
    every target ships an analytic score.
    """
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        (grad,) = torch.autograd.grad(target.log_prob(x).sum(), x)
    return grad
