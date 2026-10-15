# Tier 1 many-particle targets, the calibration systems shared by both benchmark cultures

# Libraries
import math
import numpy as np
import torch
from .kernels import random_walk_mh
from ..data import fetch_reference, read_rows
from ..metrics.structure import compute_structure_metrics
from ..metrics.weights import normalize_weights
from .base import Target, generator

# Epsilon inside the sqrt of every pairwise distance, so the score stays finite at d_ij = 0
DISTANCE_EPS = 1e-12

# The working dtype is float32 as of 2026-08-25, having been float64 since the corpus was built.
# The reason is measured rather than assumed: against float64 on real configurations the energies
# agree to ~1e-4 kT and the forces to ~1e-5 of their own rms, while the integrator's own timestep
# bias is 0.08 kT -- two to three orders of magnitude larger. float64 stays available on every
# constructor and is what the OpenMM comparisons use, since they resolve ~1e-6 kT.

def remove_mean(x, n_particles, spatial_dim):
    """Subtracts the centre of mass of each configuration

    Args:
        * x (torch.Tensor of shape (n, n_particles * spatial_dim)): Configurations
        * n_particles (int): Number of particles
        * spatial_dim (int): Spatial dimension

    Returns:
        * x (torch.Tensor of the same shape): Mean-free configurations
    """
    x = x.reshape(len(x), n_particles, spatial_dim)
    return (x - x.mean(dim=1, keepdim=True)).reshape(len(x), n_particles * spatial_dim)


def lattice_configuration(n_particles, spatial_dim, spacing, dtype=torch.float32):
    """Builds the fixed starting configuration of every train-split chain

    Particles occupy the n_particles sites of a simple cubic lattice of the given spacing that
    are closest to the origin, ordered by distance then lexicographically, and the result is
    mean-free. Deterministic and documented, per the recipe's second knob.

    Args:
        * n_particles (int): Number of particles
        * spatial_dim (int): Spatial dimension
        * spacing (float): Lattice spacing
        * dtype (torch.dtype): Output dtype (default is torch.float32)

    Returns:
        * x (torch.Tensor of shape (n_particles * spatial_dim,)): The starting configuration
    """
    width = int(math.ceil(n_particles ** (1.0 / spatial_dim))) + 2
    axis = np.arange(width) - (width - 1) / 2.0
    sites = np.stack(np.meshgrid(*([axis] * spatial_dim), indexing='ij'), axis=-1)
    sites = sites.reshape(-1, spatial_dim)
    order = np.lexsort(tuple(sites.T[::-1]) + ((sites ** 2).sum(-1),))
    sites = sites[order][:n_particles] * spacing
    sites = sites - sites.mean(0)
    return torch.as_tensor(sites.reshape(-1), dtype=dtype)


class ParticleTarget(Target):
    """Base class for the translation/rotation/permutation-invariant particle systems

    The energy is a sum over unordered pairs of a radial pair potential, so both the energy and
    its analytic gradient are written once here: subclasses provide pair_energy(d) and
    pair_energy_grad(d), and any extra term (LJ's centre-of-mass oscillator) is added on top.

    WARNING: the energy is invariant to translation, so metrics must compare mean-free
    configurations -- preprocess() does that, and every metric goes through it.
    """

    truth_tier = 1
    reference_folder = 'pt'

    # Tag of the train recipe, in the train cache key: the split is returned centred
    train_recipe = 'single-chain-centred'

    # How many times the energy counts each unordered pair. DW-4 counts each once (bgflow's
    # compute_distances(remove_duplicates=True)); the Lennard-Jones systems count each TWICE,
    # because bgflow's LennardJonesPotential sums the full off-diagonal distance matrix and dem
    # leaves energy_factor at 1.0. Both were settled against the released data rather than read
    # off a formula -- see the class docstrings.
    pair_multiplicity = 1

    def __init__(self, dim, name, n_particles, spatial_dim, prior_scale, dtype=torch.float32):
        """Builds the particle target

        Args:
            * dim (int): n_particles * spatial_dim
            * name (str): Version tag
            * n_particles (int): Number of particles
            * spatial_dim (int): Spatial dimension
            * prior_scale (float): Scale of the isotropic Gaussian reference distribution
            * dtype (torch.dtype): Working dtype (default is torch.float32)
        """
        # Call the parent constructor
        super().__init__(dim, name, dtype=dtype)

        # 1. Geometry of the system
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.n_pairs = n_particles * (n_particles - 1) // 2
        self.prior_scale = prior_scale

        # 2. The upper-triangular pair indexes, matching jarridrb/DEM's interatomic_dist
        rows, cols = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer('pair_rows', rows)
        self.register_buffer('pair_cols', cols)

    def spec(self):
        """Adds the particle count and the spatial dimension, which dim alone does not separate"""
        spec = super().spec()
        spec.update({'n_particles': self.n_particles, 'spatial_dim': self.spatial_dim})
        return spec

    def pair_energy(self, distances):
        """Evaluates the pair potential at the given distances"""
        raise NotImplementedError('pair_energy is not implemented.')

    def pair_energy_grad(self, distances):
        """Evaluates the derivative of the pair potential with respect to the distance"""
        raise NotImplementedError('pair_energy_grad is not implemented.')

    def distances(self, x):
        """Returns the upper-triangular pairwise distances of each configuration

        Chunked like the energy: the displacement tensor is spatial_dim times the size of the
        distances it produces, and the metric panel hands this whole reference sets at a time.

        Args:
            * x (torch.Tensor of shape (n, dim)): Configurations

        Returns:
            * distances (torch.Tensor of shape (n, n_pairs)): Distances
        """
        return self.chunked(self.distances_chunk, x, cost=self.n_pairs * self.spatial_dim)

    def distances_chunk(self, x):
        """Returns the pair distances of a batch small enough to hold its displacements"""
        positions = x.reshape(len(x), self.n_particles, self.spatial_dim)
        delta = positions[:, self.pair_rows] - positions[:, self.pair_cols]
        return torch.sqrt((delta ** 2).sum(-1) + DISTANCE_EPS)

    def energy(self, x):
        """Evaluates E(x), the pair sum with this system's pair multiplicity"""
        return self.chunked(self.energy_chunk, x, cost=self.n_pairs * self.spatial_dim)

    def energy_chunk(self, x):
        """Evaluates E(x) on a batch small enough to hold its pair displacements"""
        return self.pair_multiplicity * self.pair_energy(self.distances_chunk(x)).sum(-1)

    def log_prob(self, x):
        """Evaluates log gamma(x) = -E(x)"""
        return -self.energy(x)

    def score(self, x):
        """Evaluates the score of the target at x, analytically

        The pair term contributes -sum_j f'(d_ij) (x_i - x_j) / d_ij to coordinate i, with the
        diagonal masked out so the epsilon inside the distance never reaches f'.

        NOTE: each pair force is f'(d) times the unit vector (x_i - x_j) / d, never f'(d) / d
        times the displacement. f'(d) is finite at every distance (LennardJones.sixth_power caps
        r / d for that), while f'(d) / d overflows float32 below d ~ 3e-5, and two such infinite
        forces on one particle sum to NaN. So the score is finite everywhere, particles on top of
        each other included.
        """
        return self.chunked(self.score_chunk, x, cost=self.n_particles ** 2 * self.spatial_dim)

    def score_chunk(self, x):
        """Evaluates the score on a batch small enough to hold its all-pairs tensor"""
        positions = x.reshape(len(x), self.n_particles, self.spatial_dim)
        delta = positions[:, :, None, :] - positions[:, None, :, :]
        distances = torch.sqrt((delta ** 2).sum(-1) + DISTANCE_EPS)
        mask = ~torch.eye(self.n_particles, dtype=torch.bool, device=x.device)
        pair_grad = torch.where(mask, self.pair_energy_grad(distances), torch.zeros_like(distances))
        force = pair_grad[..., None] * (delta / distances[..., None])
        grad = -self.pair_multiplicity * force.sum(dim=2)
        return grad.reshape(len(x), self.dim)

    def preprocess(self, x):
        """Removes the centre of mass, making a coordinate metric translation-invariant"""
        return remove_mean(x, self.n_particles, self.spatial_dim)

    def initial_configuration(self):
        """Returns the single fixed starting configuration of the train-split chain"""
        raise NotImplementedError('initial_configuration is not implemented.')

    def test_samples(self, n=None, seed=None):
        """Returns this library's own parallel-tempering reference set

        Generated by gen/parallel_tempering/generate.py and published in this library's dataset repo
        (see data.fetch_reference), so the first call downloads it and later calls read the cache.
        It replaces the split Klein et al. 2023 published, and the reason is DW-4: that set is
        measurably not stationary for the density it comes from. Running DW-4's own HMC kernel
        (200 moves of 10 leapfrog steps of size 0.18) from 10 000 published configurations moves
        the mean pair distance from 3.9700 to 4.0374 and leaves it there; the same kernel from the
        first 10 000 configurations of the set generated here moves it from 4.0410 to 4.0393, and
        afterwards the two agree to 0.002. The whole set reads 4.0472. A small bias, in the
        direction an under-relaxed chain would leave.

        NOTE: the Lennard-Jones sets are close to the published ones, and are generated here for
        consistency and for the size. Energies: LJ-13 -42.95 +- 6.91 here against -43.13 +- 7.02,
        LJ-55 -325.40 +- 13.76 against -325.15 +- 14.19. At 5 000 configurations a side, the
        distance TVD across the two sets is 0.0083 (LJ-13) and 0.0026 (LJ-55), against 0.0071 and
        0.0027 between two halves of the published set and 0.0084 and 0.0022 between two halves of
        ours. LJ-55's energy W1 across the sets is above both floors, 0.42 against 0.23 and 0.26;
        the two mean energies differ by 0.25, 1.8 standard errors of the published mean.

        The published splits are not shipped by this library. tests/test_particles_data.py fetches
        them, at a pinned commit and checksum, because they are what settled the pair conventions.

        Args:
            * n (int or None): Number of leading samples to read; None reads all (default is None)
            * seed (int or None): Unused, the reference set is a fixed file (default is None)

        Returns:
            * x (torch.Tensor of shape (n, dim)): Reference samples

        Raises:
            * FileNotFoundError: When the set is neither cached nor published, naming the command
                that generates it
            * ValueError: When n is more than the set holds, naming the command for a larger one
        """
        command = ('python gen/parallel_tempering/generate.py --target ' + self.name
                   + ' (from a clone of the repository)')
        path = fetch_reference(self.reference_folder, self.reference_key(),
                               'Generate one with: ' + command)
        array = read_rows(path, n, command + ', with more chains (--n_chains) or more rounds '
                          '(--n_rounds)')
        x = torch.as_tensor(array, dtype=self.dtype, device=self.device)
        return self.preprocess(x)

    def generate_train_samples(self, n):
        """Runs the historical single-chain recipe

        One Metropolis chain, including its own burn-in, from the single fixed lattice
        configuration, with the original step size of Kohler et al. 2020 and no burn-in discard --
        the literal historical artifact Klein et al. 2023 rejected, turned into a documented
        stress test. Locally the kernel samples the visited basin correctly; globally the
        bound/unbound ratio (DW-4) or the relative isomer population (LJ) comes out wrong. The
        chain is returned through preprocess(), centred like the reference set.

        Args:
            * n (int): Number of samples, i.e. the length of the retained chain

        Returns:
            * x (torch.Tensor of shape (n, dim)): The biased samples, mean-free
        """
        gen = generator(0)
        x0 = self.initial_configuration()[None].clone()
        trace = random_walk_mh(self.log_prob, x0, n_steps=n, step_size=self.train_step_size,
                               generator=gen, return_trace=True)
        return self.preprocess(trace[:, 0])

    def train_cache_key(self, n):
        """Appends the recipe tag, which the density spec does not cover"""
        return super().train_cache_key(n) + '_' + self.train_recipe

    def compute_metrics(self, samples, weights=None, ref_samples=None,
                        compute_standard_metrics=True, skip_costly_metrics=True, generator=None,
                        return_hist=False):
        """Replaces the panel with the interatomic-distance and energy histograms

        WARNING: this is where the binning matters, and the convention is inherited rather than
        invented. The distance histogram uses jarridrb/DEM's protocol -- 200 equal-width bins
        spanning the REFERENCE distances' own min and max -- which drops whatever the scored
        samples put outside that range and renormalises the rest, so a diffuse sampler can score
        a deceptively small TVD. The energy histogram instead uses 100 bins over the range that
        same implementation hardcodes per system (DW-4 (-26, 0), LJ-13 (-60, 0), LJ-55
        (-380, -180)), because the energy has a heavy high-side tail that would otherwise set the
        range from a handful of outliers. Both dropped masses are reported next to their TVD.

        Arguments and returns are Target.compute_metrics'; the histogram dict holds 'dist' and
        'energy', each with 'edges', 'hist' and 'ref_hist'.
        """
        samples, weights = self.cast(samples), self.cast(weights)
        ref_samples = self.cast(ref_samples)
        ref_samples = self.get_ref_samples(samples, ref_samples)
        ret = super().compute_metrics(samples, weights=weights, ref_samples=ref_samples,
                                      compute_standard_metrics=compute_standard_metrics,
                                      skip_costly_metrics=skip_costly_metrics, generator=generator)
        weights = None if weights is None else normalize_weights(weights)

        metrics, hist = compute_structure_metrics(
            self.distances(samples).detach(), self.distances(ref_samples).detach(),
            self.energy(samples).detach(), self.energy(ref_samples).detach(),
            weights=weights, energy_range=self.energy_histogram_range, return_hist=True,
        )
        ret.update(metrics)
        if return_hist:
            return ret, hist
        return ret


class MultiDoubleWell(ParticleTarget):
    """The multi-particle double well of Kohler et al. 2020, as implemented in bgflow / dem

    E(x) = sum_{i<j} [ a*(d_ij - d0)^4 + b*(d_ij - d0)^2 + c ] / tau

    NOTE: unlike the Lennard-Jones systems, this energy counts each unordered pair ONCE
    (pair_multiplicity = 1), which is what bgflow's MultiDoubleWellPotential does via
    compute_distances(remove_duplicates=True). Checked the same way: the released frames are
    stationary under this convention (drift +0.20 sigma) and not under the doubled one (-0.75
    sigma, too cold).

    WARNING: two live conventions exist for the (a, b, c) triple -- (quartic, quadratic,
    constant) here, (linear, quadratic, quartic) in Kohler et al. 2020 and Havens et al. 2026.
    Transcribing one into the other's slots drops the quartic term, so E -> -inf as particles
    separate and the density is improper. This is the single most likely reimplementation error
    in the corpus, and test_target_specs.py asserts against it.
    """

    train_step_size = 0.5
    energy_histogram_range = (-26.0, 0.0)

    def __init__(self, n_particles=4, spatial_dim=2, a=0.9, b=-4.0, c=0.0, offset=4.0, tau=1.0,
                 name='dw4', prior_scale=2.0, dtype=torch.float32):
        """Builds the multi-particle double well

        Args:
            * n_particles (int): Number of particles (default is 4)
            * spatial_dim (int): Spatial dimension (default is 2)
            * a (float): Quartic coefficient (default is 0.9)
            * b (float): Quadratic coefficient (default is -4.0)
            * c (float): Constant (default is 0.0)
            * offset (float): d0, the distance the quartic is centred on (default is 4.0)
            * tau (float): Temperature the energy is divided by (default is 1.0)
            * name (str): Short identifier (default is 'dw4')
            * prior_scale (float): Reference-distribution scale (default is 2.0)
            * dtype (torch.dtype): Working dtype (default is torch.float32)
        """
        super().__init__(n_particles * spatial_dim, name, n_particles, spatial_dim, prior_scale,
                         dtype=dtype)
        self.a, self.b, self.c = a, b, c
        self.offset = offset
        self.tau = tau

    def spec(self):
        """Adds the quartic's coefficients, its centre and the temperature it is divided by"""
        spec = super().spec()
        spec.update({'a': self.a, 'b': self.b, 'c': self.c, 'offset': self.offset,
                     'tau': self.tau, 'pair_multiplicity': self.pair_multiplicity})
        return spec

    def pair_energy(self, distances):
        """Evaluates the per-pair energy at the given distances"""
        delta = distances - self.offset
        return (self.a * delta ** 4 + self.b * delta ** 2 + self.c) / self.tau

    def pair_energy_grad(self, distances):
        """Evaluates the derivative of the per-pair energy with respect to the distance"""
        delta = distances - self.offset
        return (4 * self.a * delta ** 3 + 2 * self.b * delta) / self.tau

    def initial_configuration(self):
        """Returns particles on a lattice of spacing d0, so every pair starts near the well"""
        return lattice_configuration(self.n_particles, self.spatial_dim, self.offset,
                                     dtype=self.dtype).to(self.device)


class DW4(MultiDoubleWell):
    """DW-4: 4 particles in 2D, dim 8 -- the pinned bgflow / dem constants, which are the parent's
    own defaults"""


class LennardJones(ParticleTarget):
    """The Lennard-Jones cluster of Kohler et al. 2020, as implemented in bgflow / dem

    E(x) = (eps / tau) * sum_{i != j} [ (r/d_ij)^12 - 2*(r/d_ij)^6 ]
         + c_harm * sum_i ||x_i - mean(x)||^2

    i.e. every unordered pair counted TWICE (pair_multiplicity = 2), plus the centre-of-mass
    oscillator whose coefficient 0.5 traces to vgsatorras/en_flows and not to any paper.

    WARNING: the published formula writes this energy as eps/(2*tau) over ordered pairs, i.e.
    eps/tau over unordered pairs, and that is NOT what the released data follows. bgflow's
    LennardJonesPotential sums the full off-diagonal distance matrix and dem leaves its
    energy_factor at 1.0, so the reference implementation never applies that 1/2 -- its own comment
    says to set energy_factor=0.5 "to match the eacf energy", meaning the halved convention belongs
    to a different lineage than the EFM data shipped here.

    Settled against the data rather than by reading, twice over. Running Metropolis-Hastings at the
    released step size from released frames, the pair energy is stationary under this convention
    (drift +0.01 sigma on LJ-13, -0.04 on LJ-55) and drifts badly under the halved one (+3.27 and
    +2.16 sigma -- too hot by a factor of two). Independently, only this convention puts the
    energies inside the ranges dem hardcodes for its own histograms of these very files: LJ-13
    lands at -41.6 in (-60, 0) and LJ-55 at -323 in (-380, -180), where the halved convention gives
    -16.2 and -120.7, the latter entirely outside its own range.

    That factor of two is a factor of two in log_prob, i.e. a different temperature and a different
    target, which is why pair_multiplicity is part of spec() and moves the spec_id.
    """

    train_step_size = 0.025
    energy_histogram_range = (-60.0, 0.0)
    pair_multiplicity = 2

    def __init__(self, n_particles=13, spatial_dim=3, eps=1.0, tau=1.0, r=1.0, c_harm=0.5,
                 name='lj13', prior_scale=1.0, dtype=torch.float32):
        """Builds the Lennard-Jones cluster

        Args:
            * n_particles (int): Number of particles (default is 13)
            * spatial_dim (int): Spatial dimension (default is 3)
            * eps (float): Energy scale (default is 1.0)
            * tau (float): Temperature the pair sum is divided by (default is 1.0)
            * r (float): Distance at the pair minimum (default is 1.0)
            * c_harm (float): Coefficient of the centre-of-mass oscillator (default is 0.5)
            * name (str): Short identifier (default is 'lj13')
            * prior_scale (float): Reference-distribution scale (default is 1.0)
            * dtype (torch.dtype): Working dtype (default is torch.float32)
        """
        super().__init__(n_particles * spatial_dim, name, n_particles, spatial_dim, prior_scale,
                         dtype=dtype)
        self.eps = eps
        self.tau = tau
        self.r = r
        self.c_harm = c_harm

    def spec(self):
        """Adds the pair constants and the centre-of-mass oscillator

        pair_multiplicity is in here because it is a factor of two in log_prob, i.e. a different
        temperature and a different target, as the class docstring says.
        """
        spec = super().spec()
        spec.update({'eps': self.eps, 'tau': self.tau, 'r': self.r, 'c_harm': self.c_harm,
                     'pair_multiplicity': self.pair_multiplicity})
        return spec

    def pair_energy(self, distances):
        """Evaluates the per-pair energy at the given distances"""
        sixth = self.sixth_power(distances)
        return self.eps / self.tau * (sixth ** 2 - 2 * sixth)

    def pair_energy_grad(self, distances):
        """Evaluates the derivative of the per-pair energy with respect to the distance"""
        sixth = self.sixth_power(distances)
        return 12 * self.eps / self.tau * (sixth - sixth ** 2) / distances

    def sixth_power(self, distances):
        """Evaluates (r / d)^6, capped so the twelfth power stays representable

        DISTANCE_EPS floors d at 1e-6, where (r / d)^12 reaches 1e72: fine in float64 and far past
        float32's 3.4e38. The cap sits on the BASE, and it is set by the GRADIENT rather than the
        energy because the derivative carries a further 12 / d. The bound is 230 in float32 and
        1.3e23 in float64, so it never binds in double precision, and it never binds on a real
        configuration either, where the closest approach puts r / d near 2.
        """
        base = (self.r / distances).clamp(
            max=0.25 * torch.finfo(distances.dtype).max ** (1.0 / 13.0))
        return base ** 6

    def energy_chunk(self, x):
        """Evaluates E(x) on a batch: the doubly-counted pair sum plus the harmonic oscillator"""
        return super().energy_chunk(x) + self.c_harm * (self.preprocess(x) ** 2).sum(-1)

    def score(self, x):
        """Evaluates the score of the target at x, analytically

        The oscillator's gradient is 2 c_harm (x_i - mean), the mean-free projection being
        idempotent, so it adds -2 c_harm times the mean-free coordinates to the pair term.
        """
        return super().score(x) - 2 * self.c_harm * self.preprocess(x)

    def initial_configuration(self):
        """Returns particles on a lattice of spacing r, the pair-minimum distance"""
        return lattice_configuration(self.n_particles, self.spatial_dim, self.r,
                                     dtype=self.dtype).to(self.device)


class LJ13(LennardJones):
    """LJ-13: 13 particles in 3D, dim 39 -- the pinned bgflow / dem constants, which are the
    parent's own defaults"""


class LJ55(LennardJones):
    """LJ-55: 55 particles in 3D, dim 165"""

    energy_histogram_range = (-380.0, -180.0)

    def __init__(self, dtype=torch.float32):
        """Builds LJ-55 with the pinned bgflow / dem constants"""
        super().__init__(n_particles=55, name='lj55', dtype=dtype)
