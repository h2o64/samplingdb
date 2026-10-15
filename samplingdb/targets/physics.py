# Tier 4 glass formers: species-dependent pair mixtures on a 2D torus

# Libraries
import math
import numpy as np
import torch
from ..data import cache_dir, fetch_zenodo, read_rows
from ..metrics.structure import compute_structure_metrics
from ..metrics.weights import normalize_weights
from .base import Target, generator

# Zenodo record holding the released equilibrium configurations, CC-BY-4.0
ZENODO_RECORD = '17966995'

# md5 digests published with that record, per file
ZENODO_CHECKSUMS = {
    'ka44_T1.0_positions.pt': 'ac101e50627ba29818fd51f3ac9c32ba',
    'ka44_T1.0_species.pt': 'f6185b47f64a63b73a526efe7b1b1eee',
    'ka44_T0.32_positions.pt': '8da994d0a5cbacf0108e66f9a3634cfa',
    'ka44_T0.32_species.pt': '5f04ac2dedfa43007fb7c4429a9d9521',
    'ipl10_T0.1_positions.pt': '5da839776c4af788abe2b2e8dae87c30',
    'ipl10_T0.1_species.pt': 'eb8c763b51bb09f1c34f4054010e44ce',
    'ipl10_T0.07_positions.pt': '28d5bafe850281fd3e4f7eab15be04a2',
    'ipl10_T0.07_species.pt': 'eb8c763b51bb09f1c34f4054010e44ce',
    'ipl10_T0.04_positions.pt': '76f8e63384c8315a1f7a5e3cfb061895',
    'ipl10_T0.04_species.pt': 'eb8c763b51bb09f1c34f4054010e44ce',
    'ipl44_T0.1_positions.pt': 'ffba05df28c8da0ceee529bcc7bb7b82',
    'ipl44_T0.1_species.pt': '16172fc67ad39431f131747a61f26276',
    'ipl44_T0.07_positions.pt': 'fa25a397449e70acd0415baffe426c58',
    'ipl44_T0.07_species.pt': '4f00c550751ac00c51bec30009c90192',
}


def minimum_image(delta, box):
    """Wraps a displacement into the nearest periodic image

    Args:
        * delta (torch.Tensor of shape (..., spatial_dim)): Displacements, of any magnitude
        * box (float): Side of the cubic box

    Returns:
        * delta (torch.Tensor of the same shape): Displacements in (-box/2, box/2]
    """
    return delta - box * torch.round(delta / box)


class PeriodicPairMixture(Target):
    """Base class for a species-dependent pair mixture on a periodic square box

    log_gamma(x) = -E(x) / T, with E a sum over unordered pairs of a radial, species-dependent,
    cut-off pair potential. Both the energy and its analytic gradient are written once here:
    subclasses provide pair_potential(r, sigma, epsilon) and pair_potential_grad(...).

    WARNING: the species labels are fixed and canonically ordered (all of species 0, then all of
    species 1,...). The released configurations carry their own per-configuration labels, which
    swap moves have permuted, so the loader sorts each configuration into canonical order and
    applies the same permutation to the positions -- the energy is invariant under that, since
    labels travel with their particles.

    WARNING: on a torus, coordinates are defined only up to a translation, a particle permutation
    and a periodic image, so a coordinate-space metric is meaningless. preprocess() therefore maps
    a configuration to its SORTED minimum-image pair distances, which is invariant under all
    three, and that is what the universal panel compares.
    """

    truth_tier = 4
    has_ground_truth = True
    spatial_dim = 2
    swap_channels = ()

    # First peak of the structure factor, which is the wavevector F_s(k, t) is read at. 7.30 for
    # the ternary mixture, measured from the released ensemble as the maximum of
    # S(k) = 1 + (2/N) sum_{i<j} J_0(k r_ij).
    wavevector_peak = 7.30

    # Structural relaxation time in time units, per temperature, MEASURED as the time at which
    # F_s(k, t) crosses 1/e under this system's own reference kernel. It is what sets the
    # equilibration and the spacing of a reference run, so it is not guessed.
    relaxation_times = {}

    def __init__(self, name, species, sigma, epsilon, density, temperature, cutoff_factor=2.5,
                 prior_scale=1.0, dtype=torch.float32):
        """Builds the mixture

        Args:
            * name (str): Version tag
            * species (torch.Tensor of shape (n_particles,)): Canonically ordered species labels
            * sigma (torch.Tensor of shape (n_species, n_species)): Pair diameters
            * epsilon (torch.Tensor of shape (n_species, n_species)): Pair energy scales
            * density (float): N / L^2, which fixes the box side
            * temperature (float): T in log_gamma = -E / T
            * cutoff_factor (float): r_cut = cutoff_factor * sigma per pair (default is 2.5)
            * prior_scale (float): Scale a Gaussian reference should use (default is 1.0)
            * dtype (torch.dtype): Working dtype (default is torch.float32)
        """
        # Call the parent constructor
        n_particles = len(species)
        super().__init__(n_particles * self.spatial_dim, name, temperature=temperature,
                         dtype=dtype)

        # 1. Geometry of the system
        self.n_particles = n_particles
        self.n_pairs = n_particles * (n_particles - 1) // 2
        self.density = density
        self.box = math.sqrt(n_particles / density)
        self.cutoff_factor = cutoff_factor
        self.prior_scale = prior_scale

        # 2. Species labels and the pair parameters they select, as (n_particles, n_particles)
        self.register_buffer('species', species.to(torch.int64))
        self.register_buffer('sigma', sigma.to(dtype))
        self.register_buffer('epsilon', epsilon.to(dtype))
        self.register_buffer('pair_sigma', sigma.to(dtype)[species][:, species])
        self.register_buffer('pair_epsilon', epsilon.to(dtype)[species][:, species])

        # 3. The upper-triangular pair indexes, and the off-diagonal mask the score needs
        rows, cols = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer('pair_rows', rows)
        self.register_buffer('pair_cols', cols)
        self.register_buffer('off_diagonal', ~torch.eye(n_particles, dtype=torch.bool))

    def pair_parameters(self, full=False):
        """Returns the per-pair diameters and energy scales

        Args:
            * full (bool): Whether to return the full (n_particles, n_particles) matrices the
                score needs rather than the upper-triangular vectors the energy needs
                (default is False)

        Returns:
            * parameters (tuple of torch.Tensor): sigma and epsilon, of shape
                (n_particles, n_particles) if full else (n_pairs,)
        """
        if full:
            return self.pair_sigma, self.pair_epsilon
        return (self.pair_sigma[self.pair_rows, self.pair_cols],
                self.pair_epsilon[self.pair_rows, self.pair_cols])

    def pair_potential(self, distances, *parameters):
        """Evaluates the pair potential, already zero beyond its cutoff

        Args:
            * distances (torch.Tensor): Pair distances, of any shape
            * parameters (tuple of torch.Tensor): Per-pair parameters, broadcasting against
                distances, as pair_parameters returns them

        Returns:
            * value (torch.Tensor of the same shape as distances): The pair potential
        """
        raise NotImplementedError('pair_potential is not implemented.')

    def pair_potential_grad(self, distances, *parameters):
        """Evaluates the derivative of the pair potential with respect to the distance"""
        raise NotImplementedError('pair_potential_grad is not implemented.')

    def positions(self, x):
        """Reshapes flat coordinates into (n, n_particles, spatial_dim)"""
        return x.reshape(len(x), self.n_particles, self.spatial_dim)

    def distances(self, x):
        """Returns the upper-triangular minimum-image pair distances of each configuration

        Chunked like the energy: the displacement tensor is spatial_dim times the size of the
        distances it produces, and the metric panel hands this whole reference sets at a time.

        Args:
            * x (torch.Tensor of shape (n, dim)): Configurations, wrapped or not

        Returns:
            * distances (torch.Tensor of shape (n, n_pairs)): Distances
        """
        return self.chunked(self.distances_chunk, x, cost=self.n_pairs * self.spatial_dim)

    def distances_chunk(self, x):
        """Returns the pair distances of a batch small enough to hold its displacements"""
        positions = self.positions(x)
        delta = minimum_image(positions[:, self.pair_rows] - positions[:, self.pair_cols], self.box)
        return torch.linalg.norm(delta, dim=-1)

    def energy(self, x):
        """Evaluates E(x), the pair sum over unordered pairs"""
        return self.chunked(self.energy_chunk, x, cost=self.n_pairs)

    def energy_chunk(self, x):
        """Evaluates E(x) on a batch small enough to hold its pair tensor"""
        return self.pair_potential(self.distances_chunk(x), *self.pair_parameters()).sum(-1)

    def local_energy(self, positions, index):
        """Evaluates the energy of one particle per chain against every other particle

        This is ParticlesMC's compute_energy_particle: the inner loop of every Monte-Carlo move
        needs only the rows the move touches, which is n_particles terms instead of n_pairs.

        Args:
            * positions (torch.Tensor of shape (n_chains, n_particles, spatial_dim)): Configurations
            * index (torch.Tensor of shape (n_chains,)): Particle index, one per chain

        Returns:
            * energy (torch.Tensor of shape (n_chains,)): Sum of the pair potentials involving it
        """
        chains = torch.arange(len(positions), device=positions.device)
        delta = minimum_image(positions[chains, index][:, None, :] - positions, self.box)
        distances = torch.linalg.norm(delta, dim=-1)
        parameters = tuple(value[index] for value in self.pair_parameters(full=True))
        value = self.pair_potential(distances.clamp_min(1e-12), *parameters)
        zero = torch.zeros_like(index[:, None], dtype=value.dtype)
        return value.scatter(1, index[:, None], zero).sum(-1)

    def local_log_prob(self, positions, index):
        """Evaluates the log_prob contribution of one particle per chain

        This is the decomposition every Monte-Carlo move scores itself with, and it is only
        log_prob's local piece as long as log_prob is -E / T. swap_monte_carlo asserts that
        agreement before it runs, so a subclass that changes the law cannot be sampled with a
        kernel that no longer matches it.
        """
        return -self.local_energy(positions, index) / self.temperature

    def species_indexes(self, label):
        """Returns the particle indexes carrying the given species label"""
        return (self.species == label).nonzero(as_tuple=True)[0]

    def log_prob(self, x):
        """Evaluates log gamma(x) = -E(x) / T, with E the TOTAL energy

        WARNING: this normalisation is a real fork, and it was settled by experiment rather than
        by reading. h2o64/learndiffeq_wip's KobAndersen defines U = E / n_particles and its
        experiments use -U / temp, i.e. exp(-E / (N T)). Running the released protocol's own
        Metropolis kernel from the released T = 1.0 configurations leaves E/N at -1.930 +- 0.146
        under exp(-E / T) -- stationary, so that is the law they were drawn from -- while
        exp(-E / (N T)) drives E/N to +14.4 within 200 time units. tests/test_physics.py records
        this as a stationarity check.
        """
        return -self.energy(x) / self.temperature

    def score(self, x):
        """Evaluates the score of the target at x, analytically"""
        return self.chunked(self.score_chunk, x, cost=self.n_particles ** 2 * self.spatial_dim)

    def score_chunk(self, x):
        """Evaluates the score on a batch small enough to hold its all-pairs tensor"""
        positions = self.positions(x)
        delta = minimum_image(positions[:, :, None, :] - positions[:, None, :, :], self.box)
        distances = torch.linalg.norm(delta, dim=-1)
        safe = torch.where(self.off_diagonal, distances, torch.ones_like(distances))
        radial = self.pair_potential_grad(safe, *self.pair_parameters(full=True)) / safe
        radial = torch.where(self.off_diagonal, radial, torch.zeros_like(radial))
        grad = -(radial[..., None] * delta).sum(dim=2) / self.temperature
        return grad.reshape(len(x), self.dim)

    def preprocess(self, x):
        """Returns the sorted minimum-image pair distances, the torus-invariant representation"""
        return self.chunked(self.preprocess_chunk, x, cost=2 * self.n_pairs * self.spatial_dim)

    def preprocess_chunk(self, x):
        """Sorts the pair distances of a batch small enough to hold them and their sort"""
        return torch.sort(self.distances_chunk(x), dim=-1).values

    def distance_edges(self, n_bins=200):
        """Returns the pinned histogram edges, spanning every distance the box can produce

        NOTE: the largest minimum-image distance in a square box is box / sqrt(2), so these edges
        cover the whole support and nothing is ever dropped. The radial distribution function is
        conventionally read only up to box / 2, beyond which the shell is no longer a full
        annulus.
        """
        return torch.linspace(0.0, self.box / math.sqrt(2.0), n_bins + 1, dtype=self.dtype,
                              device=self.device)

    def self_intermediate_scattering(self, x0, xt, wavevector=None):
        """Evaluates F_s(k, t), the observable that measures how far a glass has relaxed

        F_s(k, t) = < J_0(k |r_i(t) - r_i(0)|) >, averaged over particles and chains, with the
        minimum-image displacement and the Bessel function of the first kind that is the isotropic
        average of exp(i k. dr) in TWO dimensions. It decays from 1 to 0, and the time at which it
        crosses 1/e is the structural relaxation time tau_alpha -- which is what sets how long a
        reference run has to be, and is therefore measured rather than assumed.

        Args:
            * x0 (torch.Tensor of shape (n_chains, n_particles, spatial_dim)): Configurations at
                the start of the interval
            * xt (torch.Tensor of the same shape): Configurations at the end of it
            * wavevector (float or None): k, in inverse length; None uses the first peak of the
                structure factor, wavevector_peak (default is None)

        Returns:
            * value (float): F_s(k, t), in [-1, 1]
        """
        if wavevector is None:
            wavevector = self.wavevector_peak
        displacement = minimum_image(xt - x0, self.box)
        distance = torch.linalg.norm(displacement, dim=-1)
        return float(torch.special.bessel_j0(wavevector * distance).mean())

    def specific_heat(self, x):
        """Evaluates c_V = Var(E) / (N T^2), the observable that says whether an ensemble is one

        Jung, Biroli and Berthier 2024 (Mach. Learn.: Sci. Technol. 5 035053) make this the
        sampling task their whole benchmark is scored on, and the reason is that it needs
        INDEPENDENT configurations rather than merely equilibrated ones. The mean energy plateaus
        long before c_V does: an ensemble trapped in one metabasin sees only vibrations and reads
        the equipartition value, ~1 in two dimensions, while the true value adds the fluctuation
        between basins on top. They report having tried g(r), energy histograms and the density of
        states instead, and found all three blind to exactly the deviations that matter.

        Measured on the released configurations: 0.9013 at T = 1.0 and 1.1683 at T = 0.32. The
        second sits only 17% above the vibrational floor, so it is a sensitive test rather than a
        loose one, and an under-decorrelated ensemble reads LOW.

        Args:
            * x (torch.Tensor of shape (n, dim)): Configurations

        Returns:
            * value (float): c_V per particle
        """
        energy = self.energy(x)
        return float(energy.var(unbiased=True)) / (self.n_particles * self.temperature ** 2)

    def rdf(self, x, edges=None, weights=None):
        """Estimates the radial distribution function g(r)

        Args:
            * x (torch.Tensor of shape (n, dim)): Configurations
            * edges (torch.Tensor of shape (n_bins + 1,) or None): Bin edges; None uses
                distance_edges() (default is None)
            * weights (torch.Tensor of shape (n,)): Weights of the configurations
                (default is None)

        Returns:
            * g (torch.Tensor of shape (n_bins,)): g(r) per bin
            * centres (torch.Tensor of shape (n_bins,)): Bin centres
        """
        if edges is None:
            edges = self.distance_edges()
        distances = self.distances(x).detach()
        mass = normalize_weights(weights, n=len(x), dtype=self.dtype, device=distances.device)

        # Mean number of pairs per configuration in each bin
        index = torch.bucketize(distances.contiguous(), edges, right=True) - 1
        inside = (index >= 0) & (index < len(edges) - 1)
        counts = torch.zeros(len(edges) - 1, dtype=self.dtype, device=distances.device)
        pair_mass = mass[:, None].expand_as(distances)
        counts.scatter_add_(0, index[inside], pair_mass[inside])

        # Divide by the ideal-gas count in the same annulus
        area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        ideal = self.n_pairs * area / self.box ** 2
        return counts / ideal, 0.5 * (edges[1:] + edges[:-1])

    def _zenodo_name(self, kind):
        """Returns the released file name for this system and temperature"""
        return (self.data_name + str(self.n_particles) + '_T' + self.temperature_tag
                + '_' + kind + '.pt')

    def test_samples(self, n=None, seed=None):
        """Returns this library's own swap-Monte-Carlo reference set

        Generated here rather than downloaded, by gen/swap_mc/generate.py: independent chains from
        a uniform start, then one configuration per chain every spacing_taus relaxation times. The
        run length is read off the MEASURED tau_alpha of THIS temperature rather than off the
        released protocol's 5 x 10^4 / 10^4 time units, which were sized for temperatures far
        below it -- so do not read those two numbers as this generator's settings.

        WARNING: the acceptance criterion is the specific heat, not the mean energy. c_V needs
        independent configurations rather than merely equilibrated ones and plateaus much later,
        which is exactly what an ensemble stored faster than it decorrelates gets wrong; the
        generator asserts on it at --merge. See specific_heat.

        NOTE: unlike the Tier 1 particle systems, this is a REPRODUCTION rather than a correction.
        The released configurations already pass this library's stationarity test (swap Monte Carlo
        started from them holds E/N at -2.6357 against their own -2.6273 +- 0.052), so generating
        our own buys size and independence, not a fix. released_samples() below reads theirs, and
        tests/test_physics.py scores the two against each other.

        Args:
            * n (int or None): Number of leading configurations to read (default is None)
            * seed (int or None): Unused, the reference set is a fixed file (default is None)

        Returns:
            * x (torch.Tensor of shape (n, dim)): Reference configurations, in canonical
                species order

        Raises:
            * FileNotFoundError: When the reference set has not been generated yet, naming the
                command and its cost
            * ValueError: When n is more than the set holds, naming the command for a larger one
        """
        path = cache_dir() / 'swapmc' / (self.name + '.npy')
        if not path.exists():
            raise FileNotFoundError(
                'no reference set for ' + self.name + ' at ' + str(path) + '. Generate one with: '
                './venv/bin/python gen/swap_mc/generate.py --temperature '
                + self.temperature_tag + ' --shard <i> --n_shards <k>, then --merge. 100 000 '
                'configurations is ~1.05e9 chain-time-units, i.e. ~90 core-hours at the measured '
                '3.2e3 chain-time-units per second per core -- one shard per core. '
                'released_samples() reads the published Zenodo configurations instead.'
            )
        array = read_rows(path, n, './venv/bin/python gen/swap_mc/generate.py --temperature '
                          + self.temperature_tag + ' --shard <i> --n_shards <k> over more '
                          'shards or --n_configurations, then --merge')
        x = torch.as_tensor(array, dtype=self.dtype, device=self.device)
        return x

    def released_samples(self, n=None):
        """Returns the released equilibrium configurations for this system and temperature

        The swap-Monte-Carlo (or, at the high temperature, displacement-only) protocol behind
        them is the released one: 100 independent chains, equilibration checked by both the
        potential energy and the self-intermediate scattering function, then one configuration
        stored every 10^4 (or 5 x 10^4) time units. Downloaded from the paper's own Zenodo record
        under CC-BY-4.0.

        Args:
            * n (int or None): Number of leading configurations to read (default is None)

        Returns:
            * x (torch.Tensor of shape (n, dim)): Reference configurations, in canonical
                species order
        """
        name = self._zenodo_name('positions')
        labels_name = self._zenodo_name('species')
        positions = torch.load(fetch_zenodo(ZENODO_RECORD, name, ZENODO_CHECKSUMS.get(name)),
                               weights_only=True)
        labels = torch.load(
            fetch_zenodo(ZENODO_RECORD, labels_name, ZENODO_CHECKSUMS.get(labels_name)),
            weights_only=True,
        )
        if n is not None:
            positions, labels = positions[:n], labels[:n]

        # Sort every configuration into canonical species order, positions following their labels
        order = torch.argsort(labels.to(torch.int64), dim=-1, stable=True)
        positions = torch.gather(positions, 1, order[..., None].expand_as(positions))
        return positions.reshape(len(positions), self.dim).to(dtype=self.dtype, device=self.device)

    def generate_train_samples(self, n):
        """Runs the displacement-only Metropolis recipe

        This is the one train split in the catalog that is literature-native rather than
        host-designed: displacement-only Metropolis-Hastings is exactly the run the paper made to
        establish that swap moves were necessary at the low temperature, where the relaxation
        time is ~300x the high-temperature one. One move picks a single particle uniformly and
        proposes a N(0, 0.065^2) displacement (Frenkel & Smit's standard scale); one time unit is
        n_particles attempted moves; the chains run train_time_units time units, far short of the
        5 x 10^4 the released protocol needs.

        NOTE: chains start from a jittered square lattice rather than the protocol's uniformly
        random configuration. A uniform start at this density overlaps particles, so its energy is
        ~1e25 and thousands of sweeps go into unjamming rather than into the composition ordering
        this split is meant to get wrong. The lattice is deterministic and fixed, which is the
        recipe's second knob.

        Args:
            * n (int): Number of configurations, one per chain

        Returns:
            * x (torch.Tensor of shape (n, dim)): The non-equilibrium samples
        """
        gen = generator(0)
        x = self.lattice_configuration(n, jitter=0.05, generator=gen)
        return swap_monte_carlo(self, x, n_time_units=self.train_time_units, step_size=0.065,
                                p_swap=0.0, generator=gen)

    def lattice_configuration(self, n, jitter=0.0, generator=None):
        """Returns n copies of a square lattice filling the box, optionally jittered

        Args:
            * n (int): Number of configurations
            * jitter (float): Scale of the Gaussian noise added to every coordinate
                (default is 0.0)
            * generator (torch.Generator or None): RNG of the jitter (default is None)

        Returns:
            * x (torch.Tensor of shape (n, n_particles, spatial_dim)): Configurations
        """
        side = int(math.ceil(self.n_particles ** (1.0 / self.spatial_dim)))
        axis = torch.arange(side, dtype=self.dtype) * self.box / side
        grid = torch.stack(torch.meshgrid(*([axis] * self.spatial_dim), indexing='ij'), dim=-1)
        sites = grid.reshape(-1, self.spatial_dim)[:self.n_particles]
        x = sites[None].expand(n, self.n_particles, self.spatial_dim).clone()
        if jitter > 0.0:
            x = x + jitter * torch.randn(x.shape, generator=generator, dtype=self.dtype)
        return x

    def spec(self):
        """Adds the pair parameters, the composition, the density and the cutoff"""
        spec = super().spec()
        spec.update({'sigma': self.sigma, 'epsilon': self.epsilon, 'species': self.species,
                     'density': self.density, 'cutoff_factor': self.cutoff_factor,
                     'temperature': self.temperature})
        return spec

    def train_cache_key(self, n):
        """Appends the chain length and the recipe tag, which the density spec does not cover"""
        return (super().train_cache_key(n) + '_' + self.train_recipe
                + '_t' + str(self.train_time_units))

    def compute_metrics(self, samples, weights=None, ref_samples=None,
                        compute_standard_metrics=True, skip_costly_metrics=True, generator=None,
                        return_hist=False, n_bins=200):
        """Replaces the panel with the pair distances, the energies and g(r)

        NOTE: unlike the Lennard-Jones clusters, the distance range here is pinned by the box
        rather than derived from the reference set (see distance_edges), so nothing is dropped and
        dist_out_of_range is 0 by construction. The energy range is derived from the reference
        energies, because it depends on the temperature; whatever falls outside is reported.

        NOTE: the energies are PER PARTICLE, which is how a glass former is quoted, so
        'energy_w1' and 'energy_total_var' are per-particle numbers here.

        Args are Target.compute_metrics', plus:
            * n_bins (int): Number of bins of both histograms (default is 200)

        Returns:
            * metrics (dict): The six structural numbers, plus 'rdf_max_abs_err'
            if return_hist:
                * hist (dict): 'dist' and 'energy' histogram dicts, plus 'rdf' holding the centres
                    and both g(r)
        """
        ref_samples = self.get_ref_samples(samples, ref_samples)
        ret = super().compute_metrics(samples, weights=weights, ref_samples=ref_samples,
                                      compute_standard_metrics=compute_standard_metrics,
                                      skip_costly_metrics=skip_costly_metrics, generator=generator)
        weights = None if weights is None else normalize_weights(weights)

        edges = self.distance_edges(n_bins=n_bins)
        metrics, hist = compute_structure_metrics(
            self.distances(samples).detach(), self.distances(ref_samples).detach(),
            self.energy(samples).detach() / self.n_particles,
            self.energy(ref_samples).detach() / self.n_particles,
            weights=weights, distance_edges=edges, n_energy_bins=n_bins, return_hist=True,
        )

        # The same histograms as g(r), which is what the field reads
        g, centres = self.rdf(samples, edges=edges, weights=weights)
        ref_g, _ = self.rdf(ref_samples, edges=edges)
        metrics['rdf_max_abs_err'] = float((g - ref_g).abs().max())
        hist['rdf'] = {'centres': centres, 'g': g, 'ref_g': ref_g}
        ret.update(metrics)
        if return_hist:
            return ret, hist
        return ret


class KobAndersen2D(PeriodicPairMixture):
    """The swap-friendly ternary Kob-Andersen-Lennard-Jones mixture of Jung et al. 2023

    W(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6 + C0 + C2 (r/sigma)^2 + C4 (r/sigma)^4],  r < 2.5 sigma

    with (C0, C2, C4) chosen so that W and its first TWO derivatives vanish at the cutoff, which is
    what keeps the score continuous. They are SPECIFIC to the cutoff: PINNED_SMOOTHING holds the
    values h2o64/learndiffeq_wip's KobAndersen uses at the released 2.5 -- the implementation the
    released data came from -- and any other cutoff_factor solves for its own with
    smoothing_coefficients(), because a triple carried over from 2.5 would leave the potential
    discontinuous where it is actually truncated.

    NOTE: because the pinned triple is quoted to 8 decimals rather than solved, W leaves a residual
    of ~1e-9 at the cutoff instead of ~1e-17. That is the specification, since it is what generated
    the released configurations; pass smoothing=smoothing_coefficients(2.5) for the exact one.

    WARNING: this is NOT the classic 80:20 binary Kob-Andersen glass. That
    composition is deliberately not swap-friendly, needs non-reversible parallel tempering to
    equilibrate, and has no released reference data; it is a different target with its own
    name, deferred. This one -- ternary 5/11 : 3/11 : 3/11, N = 44, rho = 1.192075 -- was
    designed so swap Monte Carlo works, which is why its ground truth already exists.
    """

    data_name = 'ka'
    train_time_units = 2000
    train_recipe = 'displacement-local-v1'
    swap_channels = ((0, 2), (1, 2))

    # The smoothing triple the released data was generated with, at the cutoff it belongs to
    PINNED_SMOOTHING = {2.5: (0.04049023795, -0.00970155098, 0.00062012616)}

    # Measured with self_intermediate_scattering from the released ensembles, 64 chains, at
    # p_swap = 0.2. At T = 0.32 the displacement-only chain gives 2674 instead of 509, so swap
    # moves are worth 5.3x HERE -- this temperature is the onset regime of the phase diagram of
    # Jung et al. 2024 (T_g near 0.15), not the deeply supercooled one where the same moves
    # are worth many orders of magnitude.
    relaxation_times = {1.0: 40.0, 0.32: 509.0}

    def __init__(self, temperature=0.32, n_particles=44, density=1.192075, cutoff_factor=2.5,
                 smoothing=None, dtype=torch.float32):
        """Builds the ternary Kob-Andersen mixture

        Args:
            * temperature (float): 0.32 (deeply supercooled) or 1.0 (onset of glassy dynamics)
                (default is 0.32)
            * n_particles (int): Number of particles, 44 in the released data (default is 44)
            * density (float): N / L^2 (default is 1.192075)
            * cutoff_factor (float): r_cut / sigma per pair (default is 2.5)
            * smoothing (tuple of three floats or None): (C0, C2, C4); None takes the released
                triple at the released cutoff and solves for any other (default is None)
            * dtype (torch.dtype): Working dtype (default is torch.float32)
        """
        assert temperature in [0.32, 1.0]

        # 1. The pinned pair parameters
        epsilon = torch.tensor([[1.0, 1.5, 0.75], [1.5, 0.5, 1.5], [0.75, 1.5, 0.75]],
                               dtype=dtype)
        sigma = torch.tensor([[1.0, 0.8, 0.9], [0.8, 0.88, 0.8], [0.9, 0.8, 0.94]], dtype=dtype)

        # 2. The 5/11 : 3/11 : 3/11 composition, in canonical order
        counts = [5 * n_particles // 11, 3 * n_particles // 11, 3 * n_particles // 11]
        species = torch.cat([torch.full((count,), label) for label, count in enumerate(counts)])
        assert len(species) == n_particles

        # 3. Call the parent constructor
        self.temperature_tag = str(temperature)
        super().__init__('ka-ternary-N' + str(n_particles) + '-T' + self.temperature_tag,
                         species, sigma, epsilon, density, temperature,
                         cutoff_factor=cutoff_factor, dtype=dtype)

        # 4. The smoothing, pinned where the released data pinned it and solved anywhere else
        if smoothing is None:
            smoothing = self.PINNED_SMOOTHING.get(cutoff_factor)
        if smoothing is None:
            smoothing = smoothing_coefficients(cutoff_factor)
        self.smoothing = smoothing

    def spec(self):
        """Adds the C0/C2/C4 smoothing triple, which is part of the potential"""
        spec = super().spec()
        spec['smoothing'] = self.smoothing
        return spec

    def pair_potential(self, distances, sigma, epsilon):
        """Evaluates the smoothed Lennard-Jones pair potential"""
        c0, c2, c4 = self.smoothing
        ratio = sigma / distances
        scaled = distances / sigma
        value = 4 * epsilon * (ratio ** 12 - ratio ** 6 + c0 + c2 * scaled ** 2 + c4 * scaled ** 4)
        return torch.where(distances < self.cutoff_factor * sigma, value, torch.zeros_like(value))

    def pair_potential_grad(self, distances, sigma, epsilon):
        """Evaluates the derivative of the smoothed pair potential

        WARNING: the smoothing terms differentiate to +2 C2 r / sigma^2 + 4 C4 r^3 / sigma^4.
        learndiffeq's own grad_Wpp instead carries them as negative powers of r, which agrees
        with autograd only where the Lennard-Jones terms dominate -- it is 1.8x off at r = 2.0
        and 78x off at r = 2.45. tests/test_physics.py checks this derivative against autograd.
        """
        _, c2, c4 = self.smoothing
        ratio = sigma / distances
        scaled = distances / sigma
        value = 4 * epsilon * (-12 * ratio ** 12 + 6 * ratio ** 6) / distances
        value = value + 4 * epsilon * (2 * c2 * scaled + 4 * c4 * scaled ** 3) / sigma
        return torch.where(distances < self.cutoff_factor * sigma, value, torch.zeros_like(value))


class KobAndersenBinary(PeriodicPairMixture):
    """The classic 80:20 binary Kob-Andersen mixture in 2D (Cheng et al. 2026)

    W(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6] - W(r_cut),  r < 2.5 sigma

    with (eps_AA, sigma_AA) = (1.0, 1.0), (eps_AB, sigma_AB) = (1.5, 0.8) and
    (eps_BB, sigma_BB) = (0.5, 0.88), N = 64 at rho = 1.0 so the box is exactly 8, and
    n_A = 51, n_B = 13.

    NOTE: only the VALUE is shifted to zero at the cutoff, not the derivative -- that is what the
    paper specifies, so the force has a small jump there. The ternary KobAndersen2D above instead
    carries C0/C2/C4 smoothing, which is one of several reasons the two are different targets.

    WARNING: this composition is deliberately NOT swap-friendly, which is the whole point of it:
    conventional Monte Carlo does not equilibrate it below the glass transition and swap moves do
    not help, so swap_channels is empty and the reference algorithm is parallel tempering
    (gen/parallel_tempering/generate.py). Nothing is downloaded for this target -- unlike the
    ternary one, no reference ensemble was published with the paper.

    WARNING: at T = 0.2 the equilibrium state is partly CRYSTALLINE, not a glass. In 62
    parallel-tempering chains (20 levels, beta_min 0.1, 8192 collection scans; 2026-10-09) the
    A particles form a hexagonal crystallite spanning the periodic box next to a B-rich amorphous
    band: 48% of A particles have |psi_6| > 0.8 with six neighbours, and the more crystalline
    configurations have the lower energy (E/N -3.216 against -3.195). That is the known behaviour
    of the 2D 80:20 mixture (Bruening et al., J. Phys.: Condens. Matter 21, 035117, 2009), which is
    why 2D glass studies use 65:35; the paper calls T = 0.2 "the glass" and validates its own
    reference on free energies alone. The density is kept as the paper defines it, so numbers stay
    comparable, but what a sampler has to get right here is the crystallite and the band around it:
    along the parallel-tempering chains the B-B contacts decorrelate in >= 1257 scans and the
    crystallite's orientation in >= 772, against 137 for the energy. Swap moves do not reach it
    either -- A<->B swaps are accepted at 6.4e-6 at T = 0.2, and neither Jung et al.'s third
    species nor Parmar, Ozawa and Berthier's polydisperse defect, coupled in exactly through an
    expanded ensemble, decorrelated it faster than displacements.
    """

    train_time_units = 2000
    train_recipe = 'displacement-local'
    swap_channels = ()

    def __init__(self, temperature=0.2, n_particles=64, density=1.0, dtype=torch.float32):
        """Builds the binary Kob-Andersen mixture

        Args:
            * temperature (float): 2.0 (liquid), 0.5, or 0.2 (partly crystalline, see above) --
                the three the paper reports (default is 0.2)
            * n_particles (int): Number of particles; 64 is the trained size, and 128, 256 and 512
                are the transfer sizes (default is 64)
            * density (float): N / L^2 (default is 1.0)
            * dtype (torch.dtype): Working dtype (default is torch.float32)
        """
        # 1. The pinned pair parameters (Cheng et al. 2026, Methods)
        epsilon = torch.tensor([[1.0, 1.5], [1.5, 0.5]], dtype=dtype)
        sigma = torch.tensor([[1.0, 0.8], [0.8, 0.88]], dtype=dtype)

        # 2. The 80:20 composition, in canonical order: 51 A and 13 B at N = 64
        n_a = round(0.8 * n_particles)
        species = torch.cat([torch.zeros(n_a, dtype=torch.int64),
                             torch.ones(n_particles - n_a, dtype=torch.int64)])

        # 3. Call the parent constructor
        self.temperature_tag = str(temperature)
        super().__init__('ka-binary-N' + str(n_particles) + '-T' + self.temperature_tag,
                         species, sigma, epsilon, density, temperature, dtype=dtype)

        # 4. The cutoff shift, one constant per pair type, so W vanishes at r_cut
        ratio = 1.0 / self.cutoff_factor
        self.register_buffer('pair_shift', 4 * self.pair_epsilon * (ratio ** 12 - ratio ** 6))

    def pair_parameters(self, full=False):
        """Returns (sigma, epsilon, shift) per pair, the shift being the cutoff offset"""
        sigma, epsilon = super().pair_parameters(full=full)
        shift = self.pair_shift if full else self.pair_shift[self.pair_rows, self.pair_cols]
        return sigma, epsilon, shift

    def pair_potential(self, distances, sigma, epsilon, shift):
        """Evaluates the truncated and shifted Lennard-Jones pair potential"""
        ratio = sigma / distances
        value = 4 * epsilon * (ratio ** 12 - ratio ** 6) - shift
        return torch.where(distances < self.cutoff_factor * sigma, value, torch.zeros_like(value))

    def pair_potential_grad(self, distances, sigma, epsilon, shift):
        """Evaluates the derivative of the pair potential, the shift contributing nothing"""
        ratio = sigma / distances
        value = 4 * epsilon * (-12 * ratio ** 12 + 6 * ratio ** 6) / distances
        return torch.where(distances < self.cutoff_factor * sigma, value, torch.zeros_like(value))

    def test_samples(self, n=None, seed=None):
        """Returns the parallel-tempering reference set, once it has been generated"""
        path = cache_dir() / 'pt' / (self.name + '.npy')
        if not path.exists():
            raise FileNotFoundError(
                'no reference set for ' + self.name + ' at ' + str(path) + '. Cheng et al. 2026 '
                'published none, so generate one with: ./venv/bin/python '
                'gen/parallel_tempering/generate.py --target ' + self.name
            )
        array = read_rows(path, n, './venv/bin/python gen/parallel_tempering/generate.py --target '
                          + self.name + ' --shard <k> --n_shards <m> for every k < m, then '
                          './venv/bin/python gen/merge_shards.py --target ' + self.name
                          + ' --min_samples <n>')
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)


def swap_monte_carlo(target, x, n_time_units, step_size=0.065, p_swap=0.0, swap_channels=None,
                     generator=None, return_acceptance=False):
    """Runs swap Monte Carlo, one chain per row of x

    The move set is TheDisorderedOrganization/ParticlesMC's, which is the field's reference
    implementation of this model: with probability 1 - p_swap a Displacement move picks one
    particle uniformly and proposes an isotropic Gaussian of scale step_size, and otherwise one
    of the DiscreteSwap channels is chosen uniformly and exchanges two particles of the two
    species it names. One time unit is n_particles attempted moves (their sweepstep = N), and
    every move is accepted by the plain Metropolis criterion on the total-energy change, both
    proposals being symmetric.

    NOTE: ParticlesMC swaps the SPECIES LABELS of two particles. Here the labels are fixed and
    canonically ordered, so the same move is realised by exchanging the two particles' POSITIONS
    -- identical energy change, identical chain, no label bookkeeping.

    NOTE: p_swap = 0 recovers the displacement-only chain, which is what the train splits use and
    what fails to equilibrate the deeply supercooled state.

    WARNING: every move is scored with local_log_prob, which is only log_prob's local piece while
    log_prob is -E / T. A subclass that changes that law has to override local_log_prob too, or
    this kernel would silently sample a different density than the one it was handed -- the exact
    failure the exp(-E / (N T)) fork was, so the agreement is CHECKED once before the run rather
    than trusted. One energy pass ahead of thousands of sweeps is not a cost worth saving.

    Args:
        * target (PeriodicPairMixture): The target, providing the energy and the temperature
        * x (torch.Tensor of shape (n_chains, n_particles, spatial_dim)): Starting configurations
        * n_time_units (int): Number of sweeps, each of n_particles attempted moves
        * step_size (float): Scale of the Gaussian displacement; 0.065 is Frenkel & Smit's
            standard scale, as used by the released protocol, and ParticlesMC uses 0.05
            (default is 0.065)
        * p_swap (float): Total probability of attempting a swap rather than a displacement;
            ParticlesMC's KA2D setup uses 0.2 (default is 0.0)
        * swap_channels (list of tuple of two ints or None): Species pairs a swap may exchange;
            None uses target.swap_channels, which for the ternary mixture is [(0, 2), (1, 2)] --
            A<->C and B<->C, never A<->B, exactly as ParticlesMC pools them (default is None)
        * generator (torch.Generator or None): RNG (default is None)
        * return_acceptance (bool): Whether to also return the mean acceptance rate per move type
            (default is False)

    Returns:
        * x (torch.Tensor of shape (n_chains, dim)): The final configurations
        if return_acceptance:
            * acceptance (dict): 'displacement' and 'swap' acceptance rates, and the realised
                'swap_fraction' of attempted moves
    """
    if swap_channels is None:
        swap_channels = target.swap_channels
    n_chains, n_particles, spatial_dim = x.shape

    flat = x.reshape(n_chains, -1)
    assert torch.allclose(target.log_prob(flat), -target.energy(flat) / target.temperature), \
        'swap_monte_carlo needs log_prob = -energy / temperature; override local_log_prob too.'

    x = x.clone()
    chains = torch.arange(n_chains, device=x.device)
    channels = [(target.species_indexes(a), target.species_indexes(b)) for a, b in swap_channels]
    counts = {'displacement': [0.0, 0.0], 'swap': [0.0, 0.0]}

    # One time unit is n_particles attempted moves, ParticlesMC's sweepstep
    for _ in range(n_time_units * n_particles):
        # 1. Pick the move type, the particle it acts on, and a swap partner where it is a swap.
        # Every random draw is made on the CPU with the caller's generator and then moved, so a
        # seeded run gives bit-identical chains on the CPU and on the GPU
        is_swap = (torch.rand(n_chains, generator=generator, dtype=x.dtype) < p_swap).to(x.device)
        index = torch.randint(n_particles, (n_chains,), generator=generator).to(x.device)
        partner = index.clone()
        channel = torch.randint(max(len(channels), 1), (n_chains,), generator=generator)
        channel = channel.to(x.device)
        for number, (first, second) in enumerate(channels):
            picked = (channel == number) & is_swap
            drawn = int(picked.sum())
            if drawn > 0:
                index[picked] = first[torch.randint(len(first), (drawn,),
                                                    generator=generator).to(x.device)]
                partner[picked] = second[torch.randint(len(second), (drawn,),
                                                       generator=generator).to(x.device)]

        # 2. Propose, scoring only the rows the move touches
        noise = step_size * torch.randn(n_chains, spatial_dim, generator=generator,
                                        dtype=x.dtype).to(x.device)
        zero = torch.zeros(n_chains, dtype=x.dtype, device=x.device)
        before = target.local_log_prob(x, index) + torch.where(
            is_swap, target.local_log_prob(x, partner), zero)
        y = x.clone()
        y[chains, index] = torch.where(is_swap[:, None], x[chains, partner],
                                       x[chains, index] + noise)
        swapping = is_swap.nonzero(as_tuple=True)[0]
        y[swapping, partner[swapping]] = x[swapping, index[swapping]]
        after = target.local_log_prob(y, index) + torch.where(
            is_swap, target.local_log_prob(y, partner), zero)

        # 3. Metropolis, both proposals being symmetric
        u = torch.rand(n_chains, generator=generator, dtype=x.dtype).to(x.device)
        accept = torch.log(u) < after - before
        x = torch.where(accept[:, None, None], y, x)
        for name, mask in [('displacement', ~is_swap), ('swap', is_swap)]:
            counts[name][0] += float((accept & mask).sum())
            counts[name][1] += float(mask.sum())

    out = x.reshape(n_chains, -1)
    if return_acceptance:
        attempted = counts['displacement'][1] + counts['swap'][1]
        acceptance = {name: value[0] / value[1] if value[1] > 0 else float('nan')
                      for name, value in counts.items()}
        acceptance['swap_fraction'] = counts['swap'][1] / attempted
        return out, acceptance
    return out


def smoothing_coefficients(cutoff_factor=2.5):
    """Solves for (C0, C2, C4) so that W, W' and W'' all vanish at the cutoff

    With u = r / sigma, g(u) = u^-12 - u^-6 and uc = cutoff_factor, the three conditions
    W(uc) = W'(uc) = W''(uc) = 0 on W / (4 eps) = g(u) + C0 + C2 u^2 + C4 u^4 give a triangular
    system whose solution is

        C4 = (g'(uc) - uc g''(uc)) / (8 uc^3)
        C2 = -(g''(uc) + 12 C4 uc^2) / 2
        C0 = -(g(uc) + C2 uc^2 + C4 uc^4)

    Both are independent of sigma and epsilon, which is why the pinned triple is a single set of
    three numbers rather than one per species pair.

    Args:
        * cutoff_factor (float): r_cut / sigma (default is 2.5)

    Returns:
        * c0, c2, c4 (float): Smoothing coefficients
    """
    uc = cutoff_factor
    g = uc ** -12 - uc ** -6
    g_prime = -12 * uc ** -13 + 6 * uc ** -7
    g_second = 156 * uc ** -14 - 42 * uc ** -8
    c4 = (g_prime - uc * g_second) / (8 * uc ** 3)
    c2 = -(g_second + 12 * c4 * uc ** 2) / 2
    c0 = -(g + c2 * uc ** 2 + c4 * uc ** 4)
    return c0, c2, c4
