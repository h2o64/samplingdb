# Tier 3 peptide targets, whose energy and forces come from OpenMM

# Libraries
import math
import numpy as np
import torch
from ..data import cache_dir, fetch_hf, fetch_reference, read_rows
from ..metrics.histogram import (
    histogram_2d, histogram_kl, jensen_shannon, shared_edges, weighted_histogram,
)
from ..metrics.molecular import N_BINS, TicaModel, fit_tica, kmeans_jsd, wrap_angles
from ..metrics.structure import compute_structure_metrics
from ..metrics.wasserstein import (
    compute_wasserstein_distance, torus_wasserstein, wasserstein_distance_1d,
)
from ..metrics.weights import normalize_weights
from .base import Target, generator
from .forcefield import GAS_CONSTANT, TorchForceField

# Hugging Face dataset holding the Sequential Boltzmann Generators trajectories, MIT
SBG_REPO = 'transferable-samplers/sequential-boltzmann-generators-data'

# Ordering of the TICA input features: C/N/S upper-triangular distances, then sin and cos of phi,
# of psi and of omega. It versions the cached projections, which a reordering would silently break.
TICA_LAYOUT = 'cns-dist-sincos'

# The two force-field families the five released systems split into. They differ in force field,
# solvent model, nonbonded treatment AND temperature, so a number from one never compares with a
# number from the other -- hence one name tag each.
FF99SBILDN_OBC = {
    'tag': 'ff99sbildn-obc',
    'force_field': ('amber99sbildn.xml', 'tip3p.xml', 'amber99_obc.xml'),
    'nonbonded_method': 'NoCutoff',
    'nonbonded_cutoff': 0.9,
}
FF14SB_OBC1 = {
    'tag': 'ff14sb-obc1',
    'force_field': ('amber14-all.xml', 'implicit/obc1.xml'),
    'nonbonded_method': 'CutoffNonPeriodic',
    'nonbonded_cutoff': 2.0,
}

# The five released systems, with everything that defines their density. Atom counts are asserted
# against the built system, so a force-field or topology mismatch fails loudly.
PEPTIDE_SYSTEMS = {
    'Ace-A-Nme': {'temperature': 300.0, 'family': FF99SBILDN_OBC, 'friction': 1.0, 'n_atoms': 22,
                  'source': 'Kohler et al. 2021'},
    'Ace-AAA-Nme': {'temperature': 300.0, 'family': FF99SBILDN_OBC, 'friction': 0.3, 'n_atoms': 42,
                    'source': 'Dibak et al. 2022'},
    'AAA': {'temperature': 310.0, 'family': FF14SB_OBC1, 'friction': 0.3, 'n_atoms': 33,
            'source': 'Tan et al. 2025'},
    'AAAAAA': {'temperature': 310.0, 'family': FF14SB_OBC1, 'friction': 0.3, 'n_atoms': 63,
               'source': 'Tan et al. 2025'},
    'GYDPETGTWG': {'temperature': 310.0, 'family': FF14SB_OBC1, 'friction': 0.3, 'n_atoms': 138,
                   'source': 'Tan et al. 2025'},
}


def dihedral(x, indexes):
    """Evaluates torsion angles from four atom indexes each, in radians

    Args:
        * x (torch.Tensor of shape (n, n_atoms, 3)): Positions
        * indexes (torch.Tensor of shape (n_torsions, 4)): Atom quadruples

    Returns:
        * angles (torch.Tensor of shape (n, n_torsions)): Angles in (-pi, pi]
    """
    p0, p1, p2, p3 = (x[:, indexes[:, k]] for k in range(4))
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / torch.linalg.norm(b1, dim=-1, keepdim=True)
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    return torch.atan2((torch.cross(b1, v, dim=-1) * w).sum(-1), (v * w).sum(-1))


def center(x, n_atoms):
    """Subtracts the centre of geometry of every frame

    Args:
        * x (torch.Tensor of shape (n, n_atoms * 3)): Positions
        * n_atoms (int): Number of atoms

    Returns:
        * x (torch.Tensor of the same shape): Centred positions
    """
    positions = x.reshape(len(x), n_atoms, 3)
    return (positions - positions.mean(dim=1, keepdim=True)).reshape(len(x), -1)


class OpenMMTarget(Target):
    """Base class for a molecular target whose force field is built by OpenMM

    log_gamma(x) = -E(x) / kT, with x the Cartesian coordinates in nanometres and E the Amber
    potential energy, softened by linlogcut (forcefield) above energy_cut.

    OpenMM builds the system and owns every parameter; TorchForceField evaluates it. The
    difference that matters is that a whole batch is now one set of tensor operations rather than
    a Python loop driving one Context over one configuration at a time, which is worth 20x to 150x
    (see TorchForceField). log_prob_and_grad returns both quantities from a SINGLE pass and is
    what a sampler should call.

    NOTE: the energy is differentiable whenever x requires grad. The SCORE can be too, so a loss
    may be built on it and differentiated through -- the OpenMM bridge this replaces could not do
    that, having injected the forces through a custom autograd Function with no double backward --
    but it costs a retained backward graph, so it is off until differentiable_score is set.

    NOTE: the score is autograd through a closed-form energy rather than the hand-written algebra
    the rest of the library uses. It is exact to machine precision, not a finite difference, and
    it is what keeps the score itself differentiable.

    WARNING: openmm and mdtraj are imported lazily, inside the constructor, so importing this
    module and the package costs nothing without the [molecular] extra installed.

    WARNING: platform_name reaches run_dynamics only, which is still OpenMM. The
    energy no longer has a platform: it is deterministic on CPU and on GPU, which is what the
    'Reference' default used to buy at a 20x price.
    """

    truth_tier = 3
    reference_folder = 'md'
    discrete = False
    friction = 1.0

    # No published energy range exists for these targets, unlike DW-4 and LJ, so the edges are
    # derived from the reference energies and the dropped mass is reported next to the TVD
    energy_histogram_range = None

    def __init__(self, name, temperature, platform_name='Reference',
                 energy_cut=1e8, energy_max=1e20, chunk_size=None, compile=True,
                 differentiable_score=False, dtype=torch.float32):
        """Builds the molecular target

        Args:
            * name (str): Version tag, which must name the force field and solvent model
            * temperature (float): Temperature in kelvin
            * platform_name (str): OpenMM platform for run_dynamics, which is the only thing
                that still runs through OpenMM. 'Reference' is the default because it is the one
                bit-reproducible platform (default is 'Reference')
            * energy_cut (float): linlogcut's high_val, in kT (default is 1e8)
            * energy_max (float): linlogcut's max_val, in kT (default is 1e20)
            * chunk_size (int or None): Configurations evaluated at once; None derives one from
                the atom count (default is None)
            * compile (bool): Whether to torch.compile the potential, worth ~2.2x. Turn it off for
                short or interactive work, for a batch shape that keeps changing, or to read a
                traceback: compilation costs ~15 s per shape and only repays over a long run
                (default is True)
            * differentiable_score (bool): Whether the score is kept differentiable, for a loss
                built on it. Off by default so nothing retains a backward graph for nothing; flip
                self.bridge.differentiable_score to turn it on later (default is False)
            * dtype (torch.dtype): Working dtype (default is torch.float32)
        """
        # 1. Build the OpenMM system, which subclasses define
        self.topology, self.system, self.reference_positions = self.build_system()
        n_atoms = self.system.getNumParticles()

        # Call the parent constructor
        super().__init__(3 * n_atoms, name, temperature=temperature, dtype=dtype)

        # 2. The geometry, and the force field that evaluates energies and forces
        self.n_atoms = n_atoms
        self.kT = GAS_CONSTANT * temperature
        self.platform_name = platform_name
        self.bridge = TorchForceField(self.system, temperature, energy_cut=energy_cut,
                                      energy_max=energy_max, chunk_size=chunk_size,
                                      compile=compile,
                                      differentiable_score=differentiable_score, dtype=dtype)

        # 3. Particle masses in amu, which only the batched integrator needs
        masses = [self.system.getParticleMass(index)._value for index in range(n_atoms)]
        self.register_buffer('masses', torch.as_tensor(masses, dtype=dtype))

        # 4. The torsions the metric panel reports, from the topology
        self.register_buffer('torsion_indexes', self.build_torsion_indexes())
        self.build_tica_geometry()
        rows, cols = torch.triu_indices(n_atoms, n_atoms, offset=1)
        self.register_buffer('pair_rows', rows)
        self.register_buffer('pair_cols', cols)

    def spec(self):
        """Adds the temperature and the two linlogcut constants, which soften log_gamma"""
        spec = super().spec()
        spec.update({'temperature': self.temperature, 'energy_cut': self.bridge.energy_cut,
                     'energy_max': self.bridge.energy_max})
        return spec

    def build_system(self):
        """Returns the (topology, system, reference_positions) triple defining this molecule"""
        raise NotImplementedError('build_system is not implemented.')

    def build_torsion_indexes(self):
        """Returns the backbone torsion atom quadruples, as a (n_torsions, 4) long tensor

        Uses mdtraj's phi/psi/omega definitions on the OpenMM topology rather than hardcoded
        indexes, so the same code works for every peptide in the catalog.

        WARNING: phi and psi columns are NOT residue-aligned in general. mdtraj's compute_phi
        drops the first residue and compute_psi drops the last, so for an uncapped peptide the two
        are offset by one: on AAA, phi covers residues [1, 2] and psi covers [0, 1]. Pairing them
        by column position would put one residue's phi next to another's psi and still call it a
        Ramachandran plot. ramachandran_pairs holds the columns matched by residue through the atom
        indices mdtraj returns, which for the capped systems is the identity and for the uncapped
        ones is not.
        """
        import mdtraj

        topology = mdtraj.Topology.from_openmm(self.topology)
        frame = mdtraj.Trajectory(np.zeros((1, self.system.getNumParticles(), 3)), topology)
        phi_indexes, _ = mdtraj.compute_phi(frame)
        psi_indexes, _ = mdtraj.compute_psi(frame)
        omega_indexes, _ = mdtraj.compute_omega(frame)
        self.n_phi, self.n_psi = len(phi_indexes), len(psi_indexes)

        # phi is C(i-1)-N(i)-CA(i)-C(i), so its N atom names the residue; psi is
        # N(i)-CA(i)-C(i)-N(i+1), so its first atom does
        phi_residues = [topology.atom(index).residue.index for index in phi_indexes[:, 1]]
        psi_residues = [topology.atom(index).residue.index for index in psi_indexes[:, 0]]
        shared = [residue for residue in phi_residues if residue in psi_residues]
        pairs = [[phi_residues.index(residue), self.n_phi + psi_residues.index(residue)]
                 for residue in shared]
        self.register_buffer('ramachandran_pairs', torch.as_tensor(pairs, dtype=torch.int64))
        self.ramachandran_residues = shared
        return torch.as_tensor(np.concatenate([phi_indexes, psi_indexes, omega_indexes]),
                               dtype=torch.int64)

    @property
    def torsion_names(self):
        """Returns one name per torsion: every phi, then every psi, then every omega"""
        n_omega = len(self.torsion_indexes) - self.n_phi - self.n_psi
        return ['phi' + str(i + 1) for i in range(self.n_phi)] \
            + ['psi' + str(i + 1) for i in range(self.n_psi)] \
            + ['omega' + str(i + 1) for i in range(n_omega)]

    @property
    def backbone_angles_slice(self):
        """Returns the slice of torsions() holding phi and psi, which the torus metrics use"""
        return slice(0, self.n_phi + self.n_psi)

    def positions(self, x):
        """Reshapes flat coordinates into (n, n_atoms, 3)"""
        return x.reshape(len(x), self.n_atoms, 3)

    def energy(self, x):
        """Evaluates the potential energy in kT"""
        return self.bridge.energy(x)

    def log_prob(self, x):
        """Evaluates log gamma(x) = -E(x) / kT"""
        return -self.bridge.energy(x)

    def score(self, x):
        """Evaluates the score, which is the force field's force divided by kT"""
        _, force = self.bridge.evaluate(x)
        return force

    def harmonic_log_prob(self, x):
        """Evaluates the log-density of the bond and angle terms alone, in kT

        This is the beta = 0 end of a SOLUTE-TEMPERING path: pair it with
        baselines.PotentialReference and the linear path becomes
        -(E_harmonic + beta * E_soft) / kT, which scales the torsions, the nonbonded terms and the
        Born solvation and leaves the stiff internal coordinates alone. See
        TorchForceField.harmonic for why that is the split worth making.
        """
        return -self.bridge.harmonic_energy(x)

    def harmonic_score(self, x):
        """Evaluates the gradient of harmonic_log_prob at x"""
        _, force = self.bridge.harmonic_evaluate(x)
        return force

    def log_prob_and_grad(self, x):
        """Evaluates the log-likelihood and the score from one pass

        This is the call that matters for a molecular target: the score is the gradient of the
        energy that was just computed, so asking for them together costs half of asking
        separately.
        """
        energy, force = self.bridge.evaluate(x)
        return -energy, force

    def preprocess(self, x):
        """Returns the pairwise interatomic distances, in atom order

        The energy is invariant to translation and rotation but NOT to permuting atoms -- they are
        labelled, unlike the particles of Tier 1 and Tier 4 -- so the invariant representation is
        the ordered upper-triangular distance matrix, which is also what the corpus's
        interatomic-distance histograms are built from. It is also immune to the centring
        difference between the released splits (see Peptide.load_split).

        Chunked like the energy: chignolin has 9453 atom pairs, so the displacements of a whole
        reference split are three times that again.
        """
        return self.chunked(self.preprocess_chunk, x, cost=len(self.pair_rows) * 3)

    def preprocess_chunk(self, x):
        """Returns the pair distances of a batch small enough to hold its displacements"""
        positions = self.positions(x)
        delta = positions[:, self.pair_rows] - positions[:, self.pair_cols]
        return torch.linalg.norm(delta, dim=-1)

    def torsions(self, x):
        """Evaluates the backbone torsions of each configuration, in radians"""
        return dihedral(self.positions(x), self.torsion_indexes)

    def generated_samples(self, n=None):
        """Returns this library's own replica-exchange ensemble, once it has been generated

        NOTE: this reads the same file test_samples() does and returns the same thing. It predates
        the generated ensemble becoming the reference and is kept because callers name it; there is
        no longer a distinction to draw between them.

        gen/md/generate.py is replica-exchange molecular dynamics with solute tempering. The
        integrator was checked against OpenMM's own LangevinMiddleIntegrator on the same starting
        frames: 2 ps from 64 released frames of Ace-A-Nme gives -24.23 kT here against -25.15 kT
        there, a 0.17 sigma difference where the standard error on the difference is 0.18 sigma.

        Args:
            * n (int or None): Number of leading frames; None reads all (default is None)

        Returns:
            * x (torch.Tensor of shape (n, dim)): Structures in nanometres, centred
        """
        return self.test_samples(n=n)

    def minimised_positions(self, x0=None):
        """Returns the energy-minimised structure, which is where a trajectory has to start

        WARNING: reference_positions is the shipped pdb verbatim, and a pdb is not a minimum of
        this force field. On chignolin it sits at E/kT = 9.0e5 against an ensemble at -453 +- 12,
        and run_batched_dynamics from there does NOT raise -- linlogcut caps the energy at
        energy_max, so the molecule flies apart to |x| ~ 2.6e4 nm inside 200 fs while every number
        stays finite and the run writes a full file of it. run_dynamics never had the problem
        because OpenMM's Simulation minimises for it; this is the same call for the torch path.

        Args:
            * x0 (numpy.ndarray of shape (n_atoms, 3) or None): Starting positions in nanometres;
                None uses reference_positions (default is None)

        Returns:
            * x (torch.Tensor of shape (dim,)): The minimised structure, in nanometres
        """
        import openmm

        if x0 is None:
            x0 = self.reference_positions
        simulation = openmm.app.Simulation(
            self.topology, self.system, openmm.VerletIntegrator(0.001),
            openmm.Platform.getPlatformByName(self.platform_name),
            {'Threads': '1'} if self.platform_name == 'CPU' else None,
        )
        simulation.context.setPositions(x0)
        simulation.minimizeEnergy()
        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)._value.reshape(-1)
        return torch.as_tensor(positions, dtype=self.dtype, device=self.device)

    def run_batched_dynamics(self, x0, n_steps, save_interval=None, step_size=0.001,
                             generator=None, return_velocities=False, score=None,
                             velocities=None):
        """Runs Langevin dynamics on a whole batch of replicas at once, in torch

        The integrator is BAOAB -- position update, Ornstein-Uhlenbeck velocity update, position
        update, both halves of the force kick around them -- which is what OpenMM's
        LangevinMiddleIntegrator is, so this samples the same law as run_dynamics above at the same
        friction and timestep. The difference is that a batch of replicas advances in one set of
        tensor operations instead of one Context per replica, which is what makes generating a
        reference ensemble affordable: on Ace-A-Nme the force field does ~4.7e5 configurations per
        second compiled, so 1024 replicas advance at ~0.46 ps of their own time per second.

        Units are OpenMM's, in which a force in kJ/mol/nm divided by a mass in amu is directly an
        acceleration in nm/ps^2: positions nm, velocities nm/ps, masses amu, friction 1/ps,
        step_size ps. The force is kT times the score, the score being the gradient of -E/kT.

        WARNING: this is dynamics, not a Metropolis chain, so a timestep too large biases the
        distribution silently rather than showing up as a rejection rate. 1 fs is the released
        protocol's own timestep and is what the reference generator uses; check any change against
        the energy of the released frames.

        Args:
            * x0 (torch.Tensor of shape (n_replicas, dim)): Starting positions, in nanometres
            * n_steps (int): Number of integration steps
            * save_interval (int or None): Steps between kept frames; None keeps only the last
                state (default is None)
            * step_size (float): Timestep in picoseconds (default is 0.001, i.e. 1 fs)
            * generator (torch.Generator or None): RNG of the velocities and the noise
                (default is None)
            * return_velocities (bool): Whether to also return the final velocities, so a run can
                be continued exactly where it stopped (default is False)
            * velocities (torch.Tensor of the shape of x0, or None): Starting velocities; None
                draws them from the Maxwell distribution at this system's temperature. Passing
                the previous call's velocities back in is what makes a sequence of short calls
                the same dynamics as one long one -- redrawing them every call is a momentum
                refresh, i.e. an Andersen thermostat at 1 / (n_steps * step_size), which at 100 fs
                per call is 10/ps against this system's own 0.3/ps (default is None)
            * score (function or None): Score to integrate against; None uses this target's own.
                A TEMPERED score turns this into the local kernel of a replica-exchange run: the
                force becomes kT * beta * score, which is the gradient of the potential beta * E,
                and the thermostat stays at kT, so the stationary law is exp(-beta E / kT) --
                the same annealed density parallel tempering asks for. Scaling the potential
                rather than the thermostat is what lets one call serve every rung
                (default is None)

        Returns:
            * x (torch.Tensor of shape (n_replicas, dim), or (n_frames, n_replicas, dim) when
                save_interval is given): The trajectory
            if return_velocities:
                * velocities (torch.Tensor of shape (n_replicas, dim)): Final velocities
        """
        score = self.score if score is None else score
        mass = self.masses.repeat_interleave(3)
        thermal = torch.sqrt(self.kT / mass)
        friction = math.exp(-self.friction * step_size)
        noise_scale = math.sqrt(1.0 - friction ** 2)

        def draw(shape):
            values = torch.randn(shape, generator=generator, dtype=self.dtype)
            return values.to(x0.device)

        x = x0.clone()
        velocity = thermal * draw(x.shape) if velocities is None else velocities.clone()
        force = self.kT * score(x).detach()
        frames = []

        for step in range(n_steps):
            velocity = velocity + 0.5 * step_size * force / mass
            x = x + 0.5 * step_size * velocity
            velocity = friction * velocity + noise_scale * thermal * draw(x.shape)
            x = x + 0.5 * step_size * velocity
            force = self.kT * score(x).detach()
            velocity = velocity + 0.5 * step_size * force / mass
            if save_interval is not None and (step + 1) % save_interval == 0:
                frames.append(center(x, self.n_atoms))

        out = torch.stack(frames) if save_interval is not None else center(x, self.n_atoms)
        if return_velocities:
            return out, velocity
        return out

    def run_dynamics(self, x0, n_steps, n_save, step_size=0.001, seed=1):
        """Runs Langevin dynamics from one starting structure and returns the saved frames

        The integrator matches the released protocol: LangevinMiddleIntegrator at this system's
        friction with a 1 fs timestep. A separate Simulation is built here rather than reusing the
        force field, so the energy evaluator is untouched by the MD settings.

        WARNING: the seed must not be zero. OpenMM reads a seed of 0 as an instruction to draw a
        fresh random one, for both setRandomNumberSeed and setVelocitiesToTemperature, so seed=0
        returns a different trajectory on every call -- 0.17 nm apart after 200 steps on
        Ace-A-Nme. Every other seed is bit-reproducible.

        Args:
            * x0 (numpy.ndarray of shape (n_atoms, 3)): Starting positions, in nanometres
            * n_steps (int): Number of integration steps
            * n_save (int): Number of frames to keep, evenly spaced
            * step_size (float): Timestep in picoseconds (default is 0.001, i.e. 1 fs)
            * seed (int): Integrator and velocity seed, which must be nonzero (default is 1)

        Returns:
            * x (torch.Tensor of shape (n_save, dim)): The saved frames, centred
        """
        import openmm

        integrator = openmm.LangevinMiddleIntegrator(
            self.temperature * openmm.unit.kelvin, self.friction / openmm.unit.picosecond,
            step_size * openmm.unit.picosecond,
        )
        integrator.setRandomNumberSeed(seed)
        # One thread, because a peptide this small is slower on more: 0.22 ms per step on one
        # thread against 0.62 on sixteen, the threads spending their time synchronising
        simulation = openmm.app.Simulation(
            self.topology, self.system, integrator,
            openmm.Platform.getPlatformByName(self.platform_name),
            {'Threads': '1'} if self.platform_name == 'CPU' else None,
        )
        simulation.context.setPositions(x0)
        simulation.minimizeEnergy()
        simulation.context.setVelocitiesToTemperature(self.temperature * openmm.unit.kelvin, seed)

        frames = []
        for _ in range(n_save):
            simulation.step(max(n_steps // n_save, 1))
            state = simulation.context.getState(getPositions=True)
            frames.append(state.getPositions(asNumpy=True)._value.reshape(-1).copy())
        x = torch.as_tensor(np.stack(frames), dtype=self.dtype, device=self.device)
        return center(x, self.n_atoms)

    def build_tica_geometry(self):
        """Records the atom subset and dihedrals TICA features are built from

        transferable-samplers slices the trajectory to its C, N and S atoms and then takes both the
        upper-triangular distances of that subset and its sin/cos-wrapped phi, psi and omega. The
        dihedrals are recomputed on the SLICED topology, so their atom indexes are mapped back to
        the full molecule here and everything downstream is one torch gather.

        The wrapped angles are laid out one torsion family at a time -- sin phi, cos phi, then psi,
        then omega -- because that is the column order upstream's fitted projections expect.
        """
        import mdtraj

        topology = mdtraj.Topology.from_openmm(self.topology)
        selection = topology.select('symbol == C or symbol == N or symbol == S')
        sliced = topology.subset(selection)
        frame = mdtraj.Trajectory(np.zeros((1, len(selection), 3)), sliced)
        blocks = [mdtraj.compute_phi(frame)[0], mdtraj.compute_psi(frame)[0],
                  mdtraj.compute_omega(frame)[0]]
        blocks = [block for block in blocks if len(block)]
        self.tica_angle_splits = tuple(len(block) for block in blocks)
        indexes = np.concatenate(blocks)
        rows, cols = torch.triu_indices(len(selection), len(selection), offset=1)
        self.register_buffer('tica_selection', torch.as_tensor(np.asarray(selection),
                                                              dtype=torch.int64))
        self.register_buffer('tica_torsions', torch.as_tensor(selection[indexes],
                                                             dtype=torch.int64))
        self.register_buffer('tica_pair_rows', rows)
        self.register_buffer('tica_pair_cols', cols)

    def tica_features(self, x):
        """Evaluates the TICA input features of a batch, on device

        Args:
            * x (torch.Tensor of shape (n, dim)): Configurations

        Returns:
            * features (torch.Tensor of shape (n, n_features)): Distances then wrapped dihedrals
        """
        positions = self.positions(x)
        subset = positions[:, self.tica_selection]
        delta = subset[:, self.tica_pair_rows] - subset[:, self.tica_pair_cols]
        distances = torch.linalg.norm(delta, dim=-1)
        angles = dihedral(positions, self.tica_torsions)
        wrapped = [wrap_angles(block) for block in angles.split(self.tica_angle_splits, dim=-1)]
        return torch.cat([distances] + wrapped, dim=-1)

    def tica_model(self, n_frames=20000, lagtime=100):
        """Returns this system's TICA projection, fitting it once on the reference trajectory

        The fit needs TIME-ORDERED frames, so it uses the released train split in file order
        rather than any shuffled or scored set, and it is cached next to the other derived data.
        The cache name carries TICA_LAYOUT because a projection is only meaningful for the exact
        feature ordering it was fitted on, and a reordering keeps the shape.

        Args:
            * n_frames (int): Frames to fit on (default is 20000)
            * lagtime (int): Lag of the covariance, in frames (default is 100)

        Returns:
            * model (TicaModel): The fitted projection, on this target's device
        """
        if getattr(self, '_tica_model', None) is None:
            path = cache_dir() / 'tica' / (self.name + '-' + TICA_LAYOUT + '.npz')
            if path.exists():
                stored = np.load(path)
                model = TicaModel(torch.as_tensor(stored['mean'], dtype=self.dtype),
                                  torch.as_tensor(stored['projection'], dtype=self.dtype))
            else:
                features = self.tica_features(self.load_split('train', n=n_frames))
                model = fit_tica(features, lagtime=lagtime)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, mean=model.mean.cpu().numpy(),
                         projection=model.projection.cpu().numpy())
            self._tica_model = model.to(self.device)
        return self._tica_model

    def compute_upstream_metrics(self, samples, ref_samples, weights=None,
                                 max_ot_samples=2000, n_clusters=20, energy=None, ref_energy=None,
                                 return_hist=False):
        """The molecular panel of transferable-samplers, in batched torch

        Their five scalars, under their names: energy-w1 and energy-w2 (exact 1D transport on the
        energies), torus-w2 (exact transport on all phi and psi at once, with the per-coordinate
        distance min(|d|, 2 pi - |d|)), tica-w2 (exact transport on the first two TICA components)
        and the two k-means Jensen-Shannon divergences, on TICA coordinates and on the
        sin/cos-wrapped dihedrals. Every feature is computed on device; only the transport linear
        programs and the one-off TICA fit leave it.

        NOTE: exact optimal transport is O(n^2) in memory, so both sides are truncated to
        max_ot_samples and the count used is reported as n_ot_samples rather than left implicit.

        Args:
            * samples (torch.Tensor of shape (n, dim)): Samples to score
            * ref_samples (torch.Tensor of shape (m, dim)): Reference samples
            * weights (torch.Tensor of shape (n,)): Weights of the samples (default is None)
            * max_ot_samples (int): Cap on the transport problems (default is 2000)
            * n_clusters (int): Clusters for the two JSD metrics (default is 20)
            * energy (torch.Tensor of shape (n,) or None): Energies of samples in kT, so a caller
                that already has them does not pay for a second force-field pass
                (default is None)
            * ref_energy (torch.Tensor of shape (m,) or None): Energies of ref_samples
                (default is None)
            * return_hist (bool): Whether to also return every histogram (default is False)

        Returns:
            * metrics (dict): 'energy_w1', 'energy_w2', 'torus_w2', 'tica_w2', 'tica_k_jsd',
                'torus_k_jsd' and 'n_ot_samples'
            if return_hist:
                * hist (dict): 'energy', 'distances', 'com_norm', 'ramachandran' and 'tica', each
                    holding 'edges' and the two histograms 'hist' and 'ref_hist'. Every one of
                    them is a probability summing to one, the 1D ones over their bins and the 2D
                    ones over the plane; 'ramachandran' holds one array per residue, in the order
                    of ramachandran_residues
        """
        n = min(len(samples), len(ref_samples), max_ot_samples)
        cut, ref_cut = samples[:n], ref_samples[:n]
        cut_weights = None if weights is None else weights[:n]

        # Energies, exactly as upstream's energy_wasserstein
        if energy is None:
            energy = self.energy(samples).detach()
        if ref_energy is None:
            ref_energy = self.energy(ref_samples).detach()
        metrics = {
            'energy_w1': float(wasserstein_distance_1d(ref_energy, energy, v_weights=weights)),
            'energy_w2': float(wasserstein_distance_1d(ref_energy, energy, v_weights=weights,
                                                       power=2)),
        }

        # All phi and psi at once, on the torus
        angles = self.torsions(cut)[:, self.backbone_angles_slice].detach()
        ref_angles = self.torsions(ref_cut)[:, self.backbone_angles_slice].detach()
        metrics['torus_w2'] = torus_wasserstein(ref_angles, angles, power=2, weights=cut_weights)

        # The first two TICA components, and the two k-means divergences
        model = self.tica_model()
        tics = model(self.tica_features(cut).detach())
        ref_tics = model(self.tica_features(ref_cut).detach())
        metrics['tica_w2'] = compute_wasserstein_distance(ref_tics, tics, power=2,
                                                 weights1=cut_weights)
        metrics['tica_k_jsd'] = kmeans_jsd(ref_tics, tics, n_clusters=n_clusters,
                                           weights=cut_weights)[0]
        all_angles = self.torsions(samples).detach()
        ref_all_angles = self.torsions(ref_samples).detach()
        metrics['torus_k_jsd'] = kmeans_jsd(wrap_angles(ref_all_angles), wrap_angles(all_angles),
                                            n_clusters=n_clusters, weights=weights)[0]
        metrics['n_ot_samples'] = float(n)

        if not return_hist:
            return metrics

        # The five histograms their plots are drawn from, as arrays
        hist = {}
        edges = shared_edges(ref_energy, energy)
        hist['energy'] = {'edges': edges, 'ref_hist': weighted_histogram(ref_energy, edges),
                          'hist': weighted_histogram(energy, edges, weights=weights)}
        distances = self.preprocess(samples).detach()
        ref_distances = self.preprocess(ref_samples).detach()
        edges = shared_edges(ref_distances, distances)
        hist['distances'] = {'edges': edges,
                             'ref_hist': weighted_histogram(ref_distances, edges),
                             'hist': weighted_histogram(distances, edges, weights=weights)}
        com = self.positions(samples).mean(1).norm(dim=-1).detach()
        ref_com = self.positions(ref_samples).mean(1).norm(dim=-1).detach()
        edges = shared_edges(ref_com, com)
        hist['com_norm'] = {'edges': edges, 'ref_hist': weighted_histogram(ref_com, edges),
                            'hist': weighted_histogram(com, edges, weights=weights)}
        angle_edges = _angle_edges(N_BINS, all_angles)
        hist['ramachandran'] = {'edges': angle_edges, 'residues': self.ramachandran_residues,
                                'hist': [], 'ref_hist': []}
        for pair in self.ramachandran_pairs.tolist():
            hist['ramachandran']['hist'].append(
                histogram_2d(all_angles[:, pair], angle_edges, weights=weights))
            hist['ramachandran']['ref_hist'].append(
                histogram_2d(ref_all_angles[:, pair], angle_edges))
        edges = shared_edges(ref_tics.reshape(-1), tics.reshape(-1))
        hist['tica'] = {'edges': edges, 'hist': histogram_2d(tics, edges, weights=cut_weights),
                        'ref_hist': histogram_2d(ref_tics, edges)}
        return metrics, hist

    def compute_torsion_metrics(self, samples, ref_samples, weights=None, n_bins=64,
                                n_bins_joint=32, n_bins_plot=200, return_hist=False):
        """Compare the backbone torsions against a reference set

        The panel the molecular literature reports (Havens et al. 2026, OuYang et al. 2026): a 1D
        KL and Wasserstein distance per torsion marginal, and the Jensen-Shannon divergence of the
        2D Ramachandran joint. Torsion metrics come first and the energy ones second, because a
        handful of high-energy structures dominate the energy comparison while telling you little
        about the conformational ensemble.

        WARNING: a histogram divergence is only a distance when the bins are populated, and the
        corpus's 200 x 200 Ramachandran grid is 40 000 cells -- built for PLOTS, at which it is
        the right convention. Scored on 500 samples it is 98% empty, and js_ramachandran then
        reads ~0.61 for two halves of the SAME ensemble, indistinguishable from a genuinely
        collapsed sampler. So the divergences default to a coarser grid (64 bins per marginal,
        32 x 32 joint), the plot-grade 200 x 200 histograms are still what return_hist gives back,
        and 'ramachandran_occupancy' reports the fraction of reference cells that carry any mass
        at all -- read a divergence next to it, or prefer the binning-free w1_ entries.

        Args:
            * samples (torch.Tensor of shape (batch_size, dim)): Samples to score
            * ref_samples (torch.Tensor of shape (n_ref, dim)): Reference samples
            * weights (torch.Tensor of shape (batch_size,)): Weights of the samples
                (default is None)
            * n_bins (int): Bins per marginal for the divergences (default is 64)
            * n_bins_joint (int): Bins per axis of the joint for the divergence (default is 32)
            * n_bins_plot (int): Bins per axis of the returned histograms, the corpus's
                plot-grade convention (default is 200)
            * return_hist (bool): Whether to also return the raw histograms (default is False)

        Returns:
            * metrics (dict): 'kl_<name>' and 'w1_<name>' per torsion, plus 'js_ramachandran'
                and 'ramachandran_occupancy'
            if return_hist:
                * hist (dict): 'edges', one 1D histogram pair per torsion, and the two 2D
                    Ramachandran histograms, all on the plot-grade grid
        """
        angles = self.torsions(samples).detach()
        ref_angles = self.torsions(ref_samples).detach()
        edges = _angle_edges(n_bins, angles)
        plot_edges = _angle_edges(n_bins_plot, angles)

        # One marginal KL and Wasserstein per torsion
        metrics, hist = {}, {'edges': plot_edges}
        for index, name in enumerate(self.torsion_names):
            marginal = weighted_histogram(angles[:, index], edges, weights=weights)
            ref_marginal = weighted_histogram(ref_angles[:, index], edges)
            metrics['kl_' + name] = histogram_kl(marginal, ref_marginal)
            metrics['w1_' + name] = float(wasserstein_distance_1d(ref_angles[:, index],
                                                                 angles[:, index],
                                                                 v_weights=weights))
            hist[name] = {'hist': weighted_histogram(angles[:, index], plot_edges,
                                                     weights=weights),
                          'ref_hist': weighted_histogram(ref_angles[:, index], plot_edges)}

        # The Ramachandran joint, on the first RESIDUE-PAIRED phi/psi (see build_torsion_indexes)
        pair = angles[:, self.ramachandran_pairs[0]]
        ref_pair = ref_angles[:, self.ramachandran_pairs[0]]
        joint_edges = _angle_edges(n_bins_joint, angles)
        joint = histogram_2d(pair, joint_edges, weights=weights)
        ref_joint = histogram_2d(ref_pair, joint_edges)
        metrics['js_ramachandran'] = jensen_shannon(joint.reshape(-1), ref_joint.reshape(-1))
        metrics['ramachandran_occupancy'] = float((ref_joint > 0).to(joint.dtype).mean())
        hist['ramachandran'] = {'hist': histogram_2d(pair, plot_edges, weights=weights),
                                'ref_hist': histogram_2d(ref_pair, plot_edges)}

        if return_hist:
            return metrics, hist
        return metrics

    def compute_metrics(self, samples, weights=None, ref_samples=None,
                        compute_standard_metrics=True, skip_costly_metrics=True, generator=None,
                        return_hist=False):
        """Replaces the panel with the torsion, structural and upstream ones

        Order matters in the reading, not in the computing: the torsions describe the
        conformational ensemble, while a handful of high-energy structures dominate the energy
        comparison. The energies are evaluated ONCE and shared by both panels that need them.

        Arguments and returns are Target.compute_metrics'; the histogram dict holds 'torsions',
        'dist', 'energy' and every upstream histogram.
        """
        ref_samples = self.get_ref_samples(samples, ref_samples)
        ret = super().compute_metrics(samples, weights=weights, ref_samples=ref_samples,
                                      compute_standard_metrics=compute_standard_metrics,
                                      skip_costly_metrics=skip_costly_metrics, generator=generator)
        weights = None if weights is None else normalize_weights(weights)

        metrics, torsion_hist = self.compute_torsion_metrics(samples, ref_samples,
                                                             weights=weights, return_hist=True)

        # The distance and energy histograms, on the same panel the other three families report
        energy = self.energy(samples).detach()
        ref_energy = self.energy(ref_samples).detach()
        structure, hist = compute_structure_metrics(
            self.preprocess(samples).detach(), self.preprocess(ref_samples).detach(),
            energy, ref_energy, weights=weights,
            energy_range=self.energy_histogram_range, return_hist=True,
        )

        # 'energy_w1' is upstream's 'energy-w1', so the structural panel's copy of it goes
        del structure['energy_w1']
        metrics.update(structure)

        # The upstream panel, which is what a peptide result is actually compared on
        upstream, upstream_hist = self.compute_upstream_metrics(samples, ref_samples,
                                                               weights=weights, energy=energy,
                                                               ref_energy=ref_energy,
                                                               return_hist=True)
        metrics.update(upstream)
        hist['torsions'] = torsion_hist
        hist.update(upstream_hist)
        ret.update(metrics)
        if return_hist:
            return ret, hist
        return ret


class Peptide(OpenMMTarget):
    """One of the five peptides released with the Sequential Boltzmann Generators data

    | sequence      | atoms | T     | force field                                    |
    | ---           | ---   | ---   | ---                                            |
    | Ace-A-Nme     |  22   | 300 K | amber99sbildn + tip3p + amber99_obc, NoCutoff  |
    | Ace-AAA-Nme   |  42   | 300 K | same                                           |
    | AAA           |  33   | 310 K | amber14-all + implicit/obc1, 2 nm cutoff       |
    | AAAAAA        |  63   | 310 K | same                                           |
    | GYDPETGTWG    | 138   | 310 K | same (chignolin)                               |

    WARNING: those are TWO force-field families at two temperatures, so the five are five different
    densities and a number from one family never compares with a number from the other. The name
    carries the family tag and the temperature for exactly that reason. Only the AAA / AAAAAA pair
    forms a chain-length ladder at fixed spec.

    WARNING: the released splits are NOT consistently preprocessed. The train split is mean-centred
    (per-frame centre of geometry exactly 0) while val and test are not (centre of geometry up to
    ~1.8 nm, raw box coordinates). The conformations are equivalent -- radius of gyration 0.282 vs
    0.288 nm on Ace-A-Nme -- so this is purely a frame of reference, but a coordinate-space metric
    across splits would be dominated by a spurious 1.5 nm translation. load_split centres
    everything, and preprocess()'s pairwise distances are immune either way.

    NOTE: the dataset card names each system's source paper but does NOT state how train, val and
    test were partitioned, so this library does not claim they are equal-quality partitions. Their
    energies do agree closely on Ace-A-Nme (-22.18, -22.55, -23.11 kT for train, val, test), which
    is consistent with one ensemble without being proof of it.
    """

    prior_scale = 0.05

    def __init__(self, sequence='Ace-A-Nme', reference_split='test', platform_name='Reference',
                 chunk_size=None, compile=True, differentiable_score=False,
                 dtype=torch.float32):
        """Builds the peptide

        Args:
            * sequence (str): One of PEPTIDE_SYSTEMS' keys (default is 'Ace-A-Nme')
            * reference_split (str): Which released split test_samples() reads, 'train', 'val' or
                'test' (default is 'test')
            * platform_name (str): OpenMM platform for run_dynamics (default is 'Reference')
            * chunk_size (int or None): Configurations evaluated at once (default is None)
            * compile (bool): Whether to torch.compile the potential; turn it off for short or
                interactive work (default is True)
            * differentiable_score (bool): Whether the score is kept differentiable, for a loss
                built on it (default is False)
            * dtype (torch.dtype): Working dtype (default is torch.float32)
        """
        assert sequence in PEPTIDE_SYSTEMS
        assert reference_split in ['train', 'val', 'test']

        # 1. Everything that defines this system
        system = PEPTIDE_SYSTEMS[sequence]
        self.sequence = sequence
        self.family = system['family']
        self.friction = system['friction']
        self.expected_n_atoms = system['n_atoms']
        self.reference_split = reference_split

        # The released files are named '<sequence>_<T>K'. This is a plain attribute rather than a
        # property because build_system() runs before Target.__init__, and until that has run a
        # property reading self.temperature raises through torch.nn.Module.__getattr__.
        self.trajectory_name = sequence + '_' + str(int(system['temperature'])) + 'K'

        # 2. Call the parent constructor. The name carries the temperature, because the two
        # force-field families sit at two of them and a number from one never compares with a
        # number from the other
        name = sequence.lower() + '-T' + str(int(system['temperature']))
        super().__init__(name, system['temperature'], platform_name=platform_name,
                         chunk_size=chunk_size, compile=compile,
                         differentiable_score=differentiable_score, dtype=dtype)
        assert self.n_atoms == self.expected_n_atoms

    def spec(self):
        """Adds the sequence and its force-field family tag, the two force fields being two laws"""
        spec = super().spec()
        spec.update({'sequence': self.sequence, 'family': self.family['tag']})
        return spec

    def build_system(self):
        """Builds the OpenMM system from the released PDB and this family's force field"""
        from openmm import app, unit

        family = PEPTIDE_SYSTEMS[self.sequence]['family']
        path = fetch_hf(SBG_REPO, self.trajectory_name + '/' + self.trajectory_name + '.pdb')
        pdb = app.PDBFile(str(path))
        force_field = app.ForceField(*family['force_field'])
        built = force_field.createSystem(
            pdb.topology,
            nonbondedMethod=getattr(app, family['nonbonded_method']),
            nonbondedCutoff=family['nonbonded_cutoff'] * unit.nanometer,
            constraints=None,
        )
        return pdb.topology, built, pdb.getPositions(asNumpy=True)._value

    def load_split(self, split, n=None):
        """Reads one of the three released splits, centred

        WARNING: chignolin's released val and test splits are the SAME file -- both resolve to
        the hub blob e286745f50fa... -- so a validation number and a test number read from them
        are not two measurements. The file is the validation set: Tan et al. 2025 describe it as
        the 2 x 10^4 contiguous frames after the 10^5-frame train stretch, and their test set as
        10^4 frames strided over the rest of one 4.5 us run. The shipped file has 2 x 10^4
        frames, and its compact fraction (Calpha 3-8 below 0.6 nm) has an integrated
        autocorrelation time of ~1300 frames, which only a contiguous stretch can have.

        Args:
            * split (str): 'train', 'val' or 'test'
            * n (int or None): Number of leading frames; None reads all (default is None)

        Returns:
            * x (torch.Tensor of shape (n, dim)): Structures in nanometres, centred
        """
        assert split in ['train', 'val', 'test']
        name = self.trajectory_name + '/' + self.trajectory_name + '_' + split + '.npy'
        array = np.load(fetch_hf(SBG_REPO, name), mmap_mode='r')
        if n is not None:
            array = array[:n]
        x = torch.as_tensor(np.array(array).reshape(len(array), -1), dtype=self.dtype,
                            device=self.device)
        return center(x, self.n_atoms)

    def test_samples(self, n=None, seed=None):
        """Returns this library's own replica-exchange ensemble

        Generated by gen/md/generate.py and published in this library's dataset repo, so the first
        call downloads it (see data.fetch_reference) and later calls read the cache. It replaces the
        released splits, which are not equilibrium reference sets at the sizes they advertise.
        Measured on the observables the upstream panel is built from, their integrated
        autocorrelation times are 4.3 stored frames for AAA, 17.7 for AAAAAA, 38.3 for Ace-A-Nme,
        60.7 for Ace-AAA-Nme -- and 1115 for chignolin, whose 20 000-frame test split is therefore
        worth about 18 independent structures. No metric read against that can separate two
        samplers.

        load_split() still reads the released files, which is what a comparison against a number
        from the machine-learning literature has to use.

        NOTE: chignolin's set is planned at 10 000 structures, the others' at 100 000, and it is
        not published yet. Its fold decorrelates in ~7 ns of plain MD at 310 K (measured on the
        released train split, which is contiguous), so one independent structure costs ~11-16 ns
        of MD and 100 000 of them 300-500 A6000-days. 10 000 is also the size of the chignolin test
        set of Tan et al. 2025.

        Args:
            * n (int or None): Number of leading frames; None reads all (default is None)
            * seed (int or None): Unused, the reference set is a fixed file (default is None)

        Returns:
            * x (torch.Tensor of shape (n, dim)): Reference structures, in nanometres

        Raises:
            * FileNotFoundError: When the ensemble is neither cached nor published, naming the
                command
            * ValueError: When n is more than the set holds, naming the command for a larger one
        """
        path = fetch_reference(
            self.reference_folder, self.reference_key(),
            'Generate one with: ./venv/bin/python gen/md/generate.py --sequence ' + self.sequence
            + '.  load_split("' + self.reference_split + '") reads the released split instead, '
            'which is what the literature reports against.'
        )
        array = read_rows(path, n, './venv/bin/python gen/md/generate.py --sequence '
                          + self.sequence + ' --shard <k> --n_shards <m> for every '
                          'k < m, then ./venv/bin/python gen/merge_shards.py --folder md '
                          '--target ' + self.sequence + ' --min_samples <n>')
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    def generate_train_samples(self, n):
        """Runs the short single-trajectory MD recipe

        The pretraining protocol of Nam et al. 2025: 100 ps of Langevin dynamics at this system's
        temperature from the energy-minimised released structure, with the released integrator's
        friction and a 1 fs timestep.

        NOTE: what this gets wrong is the basin POPULATIONS, not the basin coverage. 100 ps is long
        enough for brief excursions between basins but nowhere near long enough for their relative
        weights to mean anything, so the signal is in the marginals rather than in any claim that
        some basin is empty.

        NOTE: this is ONE trajectory, so n changes the stride between kept frames (100 000 steps
        divided by n) and therefore the split itself, not just its size. The cache keys n
        separately.

        Args:
            * n (int): Number of frames to keep

        Returns:
            * x (torch.Tensor of shape (n, dim)): The non-equilibrium samples
        """
        return self.run_dynamics(self.reference_positions, n_steps=100000, n_save=n,
                                 step_size=0.001, seed=1)


def _angle_edges(n_bins, like):
    """Returns n_bins + 1 edges spanning the full angular range, matching a tensor's dtype"""
    return torch.linspace(-math.pi, math.pi, n_bins + 1, dtype=like.dtype, device=like.device)
