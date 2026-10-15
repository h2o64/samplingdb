# Tier 4 glass formers: species-dependent pair mixtures on a 2D torus

# Libraries
import math
import torch
from ..data import fetch_reference, read_rows
from ..metrics.structure import compute_structure_metrics
from ..metrics.weights import normalize_weights
from .base import Target, generator


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
    species 1,...). A swap move therefore exchanges the POSITIONS of two particles of different
    species rather than their labels; the energy is the same either way.

    WARNING: on a torus, coordinates are defined only up to a translation, a particle permutation
    and a periodic image, so a coordinate-space metric is meaningless. preprocess() therefore maps
    a configuration to its SORTED minimum-image pair distances, which is invariant under all
    three, and that is what the universal panel compares.
    """

    truth_tier = 4
    reference_folder = 'swapmc'
    spatial_dim = 2
    swap_channels = ()

    def __init__(self, name, species, sigma, epsilon, density, temperature, cutoff_factor=2.5,
                 dtype=torch.float32):
        """Builds the mixture

        Args:
            * name (str): Version tag
            * species (torch.Tensor of shape (n_particles,)): Canonically ordered species labels
            * sigma (torch.Tensor of shape (n_species, n_species)): Pair diameters
            * epsilon (torch.Tensor of shape (n_species, n_species)): Pair energy scales
            * density (float): N / L^2, which fixes the box side
            * temperature (float): T in log_gamma = -E / T
            * cutoff_factor (float): r_cut = cutoff_factor * sigma per pair (default is 2.5)
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
        return self.chunked(self.energy_chunk, x, cost=self.n_pairs * self.spatial_dim)

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
        by reading. The implementation the released configurations of Jung et al. 2023 came from
        defines U = E / n_particles and samples exp(-U / T), i.e. exp(-E / (N T)). Running the
        released protocol's own Metropolis kernel from the released T = 1.0 configurations leaves
        E/N at -1.930 +- 0.146 under exp(-E / T) -- stationary, so that is the law they were drawn
        from -- while exp(-E / (N T)) drives E/N to +14.4 within 200 time units.
        tests/test_physics.py records this as a stationarity check.
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
        mass = mass.to(distances.dtype)

        # Mean number of pairs per configuration in each bin
        index = torch.bucketize(distances.contiguous(), edges, right=True) - 1
        inside = (index >= 0) & (index < len(edges) - 1)
        counts = torch.zeros(len(edges) - 1, dtype=distances.dtype, device=distances.device)
        pair_mass = mass[:, None].expand_as(distances)
        counts.scatter_add_(0, index[inside], pair_mass[inside])

        # Divide by the ideal-gas count in the same annulus
        area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        ideal = self.n_pairs * area / self.box ** 2
        return counts / ideal, 0.5 * (edges[1:] + edges[:-1])

    def test_samples(self, n=None, seed=None):
        """Returns this library's own reference set

        Generated by the command in generator_command and published in this library's dataset
        repo, so the first call downloads it (see data.fetch_reference) and later calls read the
        cache. See docs/reference-sets.md for how each set was generated and checked.

        Args:
            * n (int or None): Number of leading configurations to read (default is None)
            * seed (int or None): Unused, the reference set is a fixed file (default is None)

        Returns:
            * x (torch.Tensor of shape (n, dim)): Reference configurations, in canonical
                species order

        Raises:
            * FileNotFoundError: When the set is neither cached nor published, naming the command
            * ValueError: When n is more than the set holds, naming the command for a larger one
        """
        command = 'python gen/' + self.generator_command + ' (from a clone of the repository)'
        path = fetch_reference(self.reference_folder, self.reference_key(),
                               'Generate one with: ' + command)
        array = read_rows(path, n, command + ', with more chains or a longer run')
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    def generate_train_samples(self, n):
        """Runs the displacement-only Metropolis recipe

        This is the one train split in the catalog that is literature-native rather than
        host-designed: displacement-only Metropolis-Hastings is exactly the run the paper made to
        establish that swap moves were necessary at the low temperature. One move picks a single
        particle uniformly and proposes a N(0, 0.065^2) displacement (Frenkel & Smit's standard
        scale); one time unit is n_particles attempted moves; the chains run train_time_units time
        units, far short of the 5 x 10^4 the released protocol needs.

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
        x = self.lattice_configuration(n, jitter=0.05, generator=gen).to(self.device)
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
        """Adds the pair parameters, the composition, the density and the cutoff

        The pair parameters enter at float32, the default working dtype, so a target built in
        float64 keeps the spec_id of the default one and reads the same reference set.
        """
        spec = super().spec()
        spec.update({'sigma': self.sigma.float(), 'epsilon': self.epsilon.float(),
                     'species': self.species, 'density': self.density,
                     'cutoff_factor': self.cutoff_factor, 'temperature': self.temperature})
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
        samples, weights = self.cast(samples), self.cast(weights)
        ref_samples = self.cast(ref_samples)
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
    values the released configurations of Jung et al. 2023 were generated with at the cutoff 2.5,
    which are also those of ParticlesMC's SmoothLennardJones, and any other cutoff_factor solves
    for its own with
    smoothing_coefficients(), because a triple carried over from 2.5 would leave the potential
    discontinuous where it is actually truncated.

    NOTE: because the pinned triple is quoted to 8 decimals rather than solved, W leaves a residual
    of ~1e-9 at the cutoff instead of ~1e-17. That is the specification, since it is what generated
    the released configurations; pass smoothing=smoothing_coefficients(2.5) for the exact one.

    WARNING: this is NOT the classic 80:20 binary Kob-Andersen glass, which is KobAndersenBinary
    below. This one -- ternary 5/11 : 3/11 : 3/11, N = 44, rho = 1.192075 -- was designed so swap
    Monte Carlo works, which is how its reference set is generated.
    """

    generator_command = 'swap_mc/generate.py --temperature <T>'
    train_time_units = 2000
    train_recipe = 'displacement-local-v1'
    swap_channels = ((0, 2), (1, 2))

    # The smoothing triple the released data was generated with, at the cutoff it belongs to
    PINNED_SMOOTHING = {2.5: (0.04049023795, -0.00970155098, 0.00062012616)}

    def __init__(self, temperature=0.32, n_particles=44, density=1.192075, cutoff_factor=2.5,
                 smoothing=None, dtype=torch.float32):
        """Builds the ternary Kob-Andersen mixture

        Args:
            * temperature (float): 0.32 (supercooled) or 1.0 (onset of glassy dynamics)
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
        self.temperature_tag = str(float(temperature))
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
        The released implementation's gradient instead carries them as negative powers of r, which
        agrees with autograd only where the Lennard-Jones terms dominate -- it is 1.8x off at r = 2.0
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
    paper specifies, so the force has a small jump there. The paper also mentions a logarithmic
    core below 0.8 sigma_AA without giving its form; it is not implemented, and it does not touch
    the equilibrium region (0.28% of T = 0.2 configurations hold a pair below 0.8). The ternary
    KobAndersen2D above instead carries C0/C2/C4 smoothing, which is one of several reasons the
    two are different targets.

    WARNING: this composition is deliberately NOT swap-friendly, which is the whole point of it:
    conventional Monte Carlo does not equilibrate it below the glass transition and swap moves do
    not help, so swap_channels is empty and the reference algorithm is parallel tempering
    (gen/parallel_tempering/generate.py). No reference ensemble was published with the paper. This
    library's set, published as pt/ka-binary-N64-T0.2_<spec_id>.npy and downloaded by
    test_samples on first use, holds 30 720 configurations at N = 64 and T = 0.2, 91% of them
    independent. It comes from non-reversible parallel tempering on ten levels geometric in T from
    0.6 down to 0.2, with half the chains started from deep configurations and half from the melt,
    and the two starts agree on every audited observable. No other temperature or size has a set.

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

    reference_folder = 'pt'
    generator_command = 'parallel_tempering/generate.py --target <name>'
    train_time_units = 2000
    train_recipe = 'displacement-local'
    swap_channels = ()

    def __init__(self, temperature=0.2, n_particles=64, density=1.0, dtype=torch.float32):
        """Builds the binary Kob-Andersen mixture

        Args:
            * temperature (float): 2.0 (liquid), 0.5, or 0.2 (partly crystalline, see above) --
                the three the paper reports; only 0.2 has a reference set (default is 0.2)
            * n_particles (int): Number of particles; 64 is the trained size and the one with a
                reference set, and 128, 256 and 512 are the transfer sizes (default is 64)
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
        self.temperature_tag = str(float(temperature))
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


def swap_move(target, x, is_swap, index, partner, noise, u):
    """Applies one Metropolis move per chain from pre-drawn randomness

    A displacement has partner = index and moves that particle by noise; a swap exchanges the
    positions of index and partner. Only the rows the move touches are scored.

    Args:
        * target (PeriodicPairMixture): The target
        * x (torch.Tensor of shape (n_chains, n_particles, spatial_dim)): Current configurations
        * is_swap (torch.Tensor of shape (n_chains,)): Whether the move is a swap
        * index (torch.Tensor of shape (n_chains,)): Particle moved, or first swapped particle
        * partner (torch.Tensor of shape (n_chains,)): Second swapped particle, index otherwise
        * noise (torch.Tensor of shape (n_chains, spatial_dim)): Proposed displacement
        * u (torch.Tensor of shape (n_chains,)): Uniform draws of the Metropolis test

    Returns:
        * x (torch.Tensor of the same shape): Configurations after the move
        * accept (torch.Tensor of shape (n_chains,)): Whether each move was accepted
    """
    chains = torch.arange(len(x), device=x.device)
    zero = torch.zeros(len(x), dtype=x.dtype, device=x.device)
    before = target.local_log_prob(x, index) + torch.where(
        is_swap, target.local_log_prob(x, partner), zero)
    y = x.clone()
    y[chains, index] = torch.where(is_swap[:, None], x[chains, partner], x[chains, index] + noise)
    y[chains, partner] = torch.where(is_swap[:, None], x[chains, index], y[chains, partner])
    after = target.local_log_prob(y, index) + torch.where(
        is_swap, target.local_log_prob(y, partner), zero)
    accept = torch.log(u) < after - before
    return torch.where(accept[:, None, None], y, x), accept


def swap_monte_carlo(target, x, n_time_units, step_size=0.065, p_swap=0.0, swap_channels=None,
                     generator=None, return_acceptance=False, compile=False):
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
    -- identical energy change, identical chain, no label bookkeeping. A per-index displacement
    therefore jumps on every accepted swap, so a relaxation time read from it is not the
    structural one.

    NOTE: p_swap = 0 recovers the displacement-only chain, which is what the train splits use and
    what fails to equilibrate the deeply supercooled state.

    NOTE: every random number is drawn on the generator's device (the CPU when generator is None)
    and the loop never synchronises with the host, so a CUDA generator runs the whole chain on the
    GPU. With p_swap = 0 the draws are the same, in the same order, as for any p_swap, so a CPU
    generator gives bit-identical train splits on every device.

    WARNING: every move is scored with local_log_prob, which is only log_prob's local piece while
    log_prob is -E / T. A subclass that changes that law has to override local_log_prob too, or
    this kernel would silently sample a different density than the one it was handed -- the exact
    failure the exp(-E / (N T)) fork was, so the agreement is CHECKED once before the run rather
    than trusted.

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
        * generator (torch.Generator or None): RNG; its device is where the draws are made
            (default is None)
        * return_acceptance (bool): Whether to also return the mean acceptance rate per move type
            (default is False)
        * compile (bool): Whether to run swap_move through torch.compile, about 9x faster on a GPU
            with a CUDA generator; the draws are unchanged (default is False)

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

    # Draws are made where the generator lives and moved once, which is free when that is x's device
    device = x.device
    draws = torch.device('cpu') if generator is None else generator.device
    x = x.clone()
    move = torch.compile(swap_move, dynamic=False) if compile else swap_move
    channels = [(target.species_indexes(a).to(device), target.species_indexes(b).to(device))
                for a, b in swap_channels]
    counts = torch.zeros(4, dtype=torch.float64, device=device)

    # One time unit is n_particles attempted moves, ParticlesMC's sweepstep
    for _ in range(n_time_units * n_particles):
        # 1. Pick the move type, the particle it acts on, and a swap partner where it is a swap
        is_swap = torch.rand(n_chains, generator=generator, dtype=x.dtype, device=draws) < p_swap
        is_swap = is_swap.to(device)
        index = torch.randint(n_particles, (n_chains,), generator=generator, device=draws)
        index = index.to(device)
        channel = torch.randint(max(len(channels), 1), (n_chains,), generator=generator,
                                device=draws).to(device)
        partner = index
        if p_swap > 0:
            for number, (first, second) in enumerate(channels):
                picked = is_swap & (channel == number)
                drawn_first = torch.randint(len(first), (n_chains,), generator=generator,
                                            device=draws).to(device)
                drawn_second = torch.randint(len(second), (n_chains,), generator=generator,
                                             device=draws).to(device)
                index = torch.where(picked, first[drawn_first], index)
                partner = torch.where(picked, second[drawn_second], partner)
        if not channels:
            is_swap = torch.zeros_like(is_swap)

        # 2. Propose and accept by Metropolis, both proposals being symmetric
        noise = step_size * torch.randn(n_chains, spatial_dim, generator=generator, dtype=x.dtype,
                                        device=draws).to(device)
        u = torch.rand(n_chains, generator=generator, dtype=x.dtype, device=draws).to(device)
        x, accept = move(target, x, is_swap, index, partner, noise, u)
        counts += torch.stack([(accept & ~is_swap).sum(), (~is_swap).sum(),
                               (accept & is_swap).sum(), is_swap.sum()]).to(counts.dtype)

    out = x.reshape(n_chains, -1)
    if return_acceptance:
        counts = counts.tolist()
        acceptance = {'displacement': counts[0] / counts[1] if counts[1] > 0 else float('nan'),
                      'swap': counts[2] / counts[3] if counts[3] > 0 else float('nan'),
                      'swap_fraction': counts[3] / (counts[1] + counts[3])}
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
