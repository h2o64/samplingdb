# Tier 3 peptide targets, whose energy and forces come from an OpenMM-built force field

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
    circular_wasserstein_1d, compute_wasserstein_distance, torus_wasserstein,
    wasserstein_distance_1d,
)
from ..metrics.weights import normalize_weights, subsample
from .base import Target, chain_layout, cycle, generator, interleave
from .forcefield import GAS_CONSTANT, TorchForceField
from .kernels import langevin, steepest_descent, temperature_ladder

# Hugging Face dataset holding the Sequential Boltzmann Generators trajectories, MIT, read at a
# pinned commit with every file checked against its sha256: the PDB defines the density (topology
# and atom order), and the test split is what the TICA projection is fitted on
SBG_REPO = 'transferable-samplers/sequential-boltzmann-generators-data'
SBG_REVISION = '61037092c603e47010988a02f281843ead2d3536'
SBG_CHECKSUMS = {
    'AAAAAA_310K/AAAAAA_310K.pdb':
        'd774bcdb830805b92058f682d74f5a4aca042d3e780c90ac8ac135437651723e',
    'AAAAAA_310K/AAAAAA_310K_test.npy':
        '894f9a92a028d4dab74bf31bf050ff1337cc251375333d5aaf449bebf896b3aa',
    'AAAAAA_310K/AAAAAA_310K_train.npy':
        '86a3193ff69faf37e144df637acbe405285d2f0f00e88f0841df14e67c39e537',
    'AAAAAA_310K/AAAAAA_310K_val.npy':
        '65e1a7434ac7fcba90ea8ecd8f865457b17d32d2ff61b818680352f2cd5e3535',
    'AAA_310K/AAA_310K.pdb':
        '6dad9622155c87a486c8a404c4cb6f22d72425e1701670fab77884d49ddf9713',
    'AAA_310K/AAA_310K_test.npy':
        'd6001247d1c250497a37ab506b6f6ae41f57bf760bde25bafa567f18d98df709',
    'AAA_310K/AAA_310K_train.npy':
        '34e78ff5dd886087e2008f89621a1dfdd72381bd0e330f403ed3e214b30c677b',
    'AAA_310K/AAA_310K_val.npy':
        '69fd765b00d6392b89d786e62a6323edc1453f61b1dbf3aaa5faf3dac5833447',
    'Ace-A-Nme_300K/Ace-A-Nme_300K.pdb':
        '548f42f837995c2cd70fc0396d28178afcccc6cb93d7fc29a080bae98575bdf2',
    'Ace-A-Nme_300K/Ace-A-Nme_300K_test.npy':
        '311221d230ba3e280be2ccb0aca3db69acf9101c91d507f6194db05cb1939019',
    'Ace-A-Nme_300K/Ace-A-Nme_300K_train.npy':
        'dd4a442bc0b4e4227c480a2314fd0513549614a4093ef6e5e08e5b1dfe8c55e3',
    'Ace-A-Nme_300K/Ace-A-Nme_300K_val.npy':
        '965f33d4f783b611cd5635ee72a1ecba384a17149818581a155e923e609fd249',
    'Ace-AAA-Nme_300K/Ace-AAA-Nme_300K.pdb':
        'e4f0d9fe243241fb68e28ff12e58ac91c9e019c502b83f68bc08f8e36c6b882a',
    'Ace-AAA-Nme_300K/Ace-AAA-Nme_300K_test.npy':
        '14040f220b1b44abf3c8be6048d47511e160be8a104fc3a5c5d8a4400b8c1c0f',
    'Ace-AAA-Nme_300K/Ace-AAA-Nme_300K_train.npy':
        '04b412b02ed5930e7af9b0ac5145e3d540cd9c00a7d2502705b1aed42293d318',
    'Ace-AAA-Nme_300K/Ace-AAA-Nme_300K_val.npy':
        '35d33035a46257bbc82d7add790f9593553bcad0fd5346e1f3f4486285fd3cf8',
}

# Ordering of the TICA input features: C/N/S upper-triangular distances, then sin and cos of phi,
# of psi and of omega. It versions the cached projections, which a reordering would silently break.
TICA_LAYOUT = 'cns-dist-sincos'

# Lag of the TICA fit, in frames of the released test split, as upstream fits it
TICA_LAGTIME = 100

# The two force-field families the four systems split into. They differ in force field,
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

# The four systems, with everything that defines their density. Atom counts are asserted
# against the built system, so a force-field or topology mismatch fails loudly.
PEPTIDE_SYSTEMS = {
    'Ace-A-Nme': {'temperature': 300.0, 'family': FF99SBILDN_OBC, 'friction': 1.0, 'n_atoms': 22},
    'Ace-AAA-Nme': {'temperature': 300.0, 'family': FF99SBILDN_OBC, 'friction': 0.3,
                    'n_atoms': 42},
    'AAA': {'temperature': 310.0, 'family': FF14SB_OBC1, 'friction': 0.3, 'n_atoms': 33},
    'AAAAAA': {'temperature': 310.0, 'family': FF14SB_OBC1, 'friction': 0.3, 'n_atoms': 63},
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


class Peptide(Target):
    """One of four peptides released with the Sequential Boltzmann Generators data

    | sequence      | atoms | T     | force field                                    |
    | ---           | ---   | ---   | ---                                            |
    | Ace-A-Nme     |  22   | 300 K | amber99sbildn + tip3p + amber99_obc, NoCutoff  |
    | Ace-AAA-Nme   |  42   | 300 K | same                                           |
    | AAA           |  33   | 310 K | amber14-all + implicit/obc1, 2 nm cutoff       |
    | AAAAAA        |  63   | 310 K | same                                           |

    log_gamma(x) = -E(x) / kT, with x the Cartesian coordinates in nanometres and E the Amber
    potential energy, softened by linlogcut (forcefield) above ENERGY_CUT. OpenMM builds the system
    and owns every parameter; TorchForceField evaluates it on a whole batch at once.
    log_prob_and_grad returns both quantities from a SINGLE pass and is what a sampler should call.

    WARNING: those are TWO force-field families at two temperatures, so the four are four different
    densities and a number from one family never compares with a number from the other. The name
    carries the sequence and the temperature for exactly that reason. Only the AAA / AAAAAA pair
    forms a chain-length ladder at fixed spec.

    WARNING: the potential is exactly inversion-symmetric (no term distinguishes a structure from
    its mirror image), so the D enantiomer of every structure carries the same weight as the L
    one, and the reference sets hold only the L enantiomer, which is what molecular dynamics from
    an L start reaches. compute_metrics therefore maps every sample whose chirality differs from
    the reference's onto its mirror image before scoring, as transferable-samplers does, and
    reports the two rates.

    NOTE: the energy is differentiable whenever x requires grad. The SCORE can be too, so a loss
    may be built on it, at the cost of a retained backward graph; it is off until
    differentiable_score is set. The score is autograd through the closed-form energy, the one
    target in the library whose score is not written out by hand, and it is exact to machine
    precision.

    NOTE: openmm, mdtraj and huggingface_hub are imported lazily, inside the constructor, so
    importing the package costs nothing without the [molecular] extra installed.
    """

    truth_tier = 3
    reference_folder = 'md'
    prior_scale = 0.05
    train_recipe = 'basin-md'

    # Backbone basins the train recipe places every residue in, as (phi, psi) centres in degrees --
    # beta/PII, alpha-R, alpha-L -- and their shares, geometric from 1 to 10 against their usual
    # order of weight, so that alpha-L, the lightest, gets the most. Klein et al. 2023 bias their
    # alanine dipeptide training set towards the same positive-phi basin
    train_basins = ((-75.0, 150.0), (-70.0, -25.0), (55.0, 35.0))
    train_basin_shares = (1.0, 10 ** 0.5, 10.0)

    # Temperature factors of the train trajectories, lowest and highest. Bonds and angles carry the
    # whole spread of the energy and relax within a picosecond, so the spread is what carries the
    # train energies past both ends of the test ones. The low end sits further from 1 because
    # alpha-L, which the basins overrepresent, is the highest basin
    train_temperatures = (0.6, 1.25)

    # The two linlogcut constants, in kT. They are part of the specification: energies above
    # ENERGY_CUT are softened logarithmically, so an atomic overlap gives a large finite number
    ENERGY_CUT = 1e8
    ENERGY_MAX = 1e20

    # No published energy range exists for these targets, unlike DW-4 and LJ, so the edges are
    # derived from the reference energies and the dropped mass is reported next to the TVD
    energy_histogram_range = None

    def __init__(self, sequence='Ace-A-Nme', chunk_size=None, compile=True,
                 differentiable_score=False, dtype=torch.float32):
        """Builds the peptide

        Args:
            * sequence (str): One of PEPTIDE_SYSTEMS' keys (default is 'Ace-A-Nme')
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
        assert sequence in PEPTIDE_SYSTEMS

        # 1. Everything that defines this system. These are plain attributes set before the parent
        # constructor, because build_system() needs them and torch.nn.Module only resolves
        # properties once it is initialised
        system = PEPTIDE_SYSTEMS[sequence]
        self.sequence = sequence
        self.family = system['family']
        self.friction = system['friction']
        self.trajectory_name = sequence + '_' + str(int(system['temperature'])) + 'K'
        self.topology, self.system, self.reference_positions = self.build_system()
        n_atoms = self.system.getNumParticles()
        assert n_atoms == system['n_atoms']

        # 2. Call the parent constructor. The name carries the temperature, because the two
        # force-field families sit at two of them
        name = sequence.lower() + '-T' + str(int(system['temperature']))
        super().__init__(3 * n_atoms, name, temperature=system['temperature'], dtype=dtype)

        # 3. The force field that evaluates energies and forces, and the masses the dynamics of
        # the train recipe need, in amu
        self.n_atoms = n_atoms
        self.register_buffer('masses', torch.as_tensor(
            [self.system.getParticleMass(index)._value for index in range(n_atoms)], dtype=dtype))
        self.kT = GAS_CONSTANT * self.temperature
        self.bridge = TorchForceField(self.system, self.temperature, energy_cut=self.ENERGY_CUT,
                                      energy_max=self.ENERGY_MAX, chunk_size=chunk_size,
                                      compile=compile,
                                      differentiable_score=differentiable_score, dtype=dtype)

        # 4. The torsions, TICA features and chirality centres the metric panel reads
        self.register_buffer('torsion_indexes', self.build_torsion_indexes())
        self.register_buffer('torsion_sides', self.build_torsion_sides())
        self.build_tica_geometry()
        self.register_buffer('chirality_centres', self.build_chirality_centres())
        rows, cols = torch.triu_indices(n_atoms, n_atoms, offset=1)
        self.register_buffer('pair_rows', rows)
        self.register_buffer('pair_cols', cols)

    def spec(self):
        """Adds the temperature, the two linlogcut constants, the sequence and the family tag"""
        spec = super().spec()
        spec.update({'temperature': self.temperature, 'energy_cut': self.bridge.energy_cut,
                     'energy_max': self.bridge.energy_max, 'sequence': self.sequence,
                     'family': self.family['tag']})
        return spec

    def build_system(self):
        """Builds the OpenMM system from the released PDB and this family's force field"""
        from openmm import app, unit

        family = PEPTIDE_SYSTEMS[self.sequence]['family']
        name = self.trajectory_name + '/' + self.trajectory_name + '.pdb'
        path = fetch_hf(SBG_REPO, name, SBG_REVISION, SBG_CHECKSUMS[name])
        pdb = app.PDBFile(str(path))
        force_field = app.ForceField(*family['force_field'])
        built = force_field.createSystem(
            pdb.topology,
            nonbondedMethod=getattr(app, family['nonbonded_method']),
            nonbondedCutoff=family['nonbonded_cutoff'] * unit.nanometer,
            constraints=None,
        )
        return pdb.topology, built, pdb.getPositions(asNumpy=True)._value

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
        self.backbone_residues = phi_residues + psi_residues
        return torch.as_tensor(np.concatenate([phi_indexes, psi_indexes, omega_indexes]),
                               dtype=torch.int64)

    def build_torsion_sides(self):
        """Returns, per torsion, the atoms a rotation about its central bond carries

        For a torsion (a, b, c, d) these are the atoms on c's side of the bond b-c, which for
        phi, psi and omega alike is the C-terminal side, found by walking the bonds from c without
        crossing b. A bond inside a ring has no side, which only a proline would bring.

        Returns:
            * sides (torch.Tensor of shape (n_torsions, n_atoms)): Boolean masks
        """
        neighbours = {atom.index: [] for atom in self.topology.atoms()}
        for bond in self.topology.bonds():
            neighbours[bond[0].index].append(bond[1].index)
            neighbours[bond[1].index].append(bond[0].index)
        sides = torch.zeros(len(self.torsion_indexes), self.n_atoms, dtype=torch.bool)
        for k, (_, b, c, _) in enumerate(self.torsion_indexes.tolist()):
            side, stack = {c}, [c]
            while stack:
                for atom in neighbours[stack.pop()]:
                    if atom != b and atom not in side:
                        side.add(atom)
                        stack.append(atom)
            assert b not in side
            sides[k, list(side)] = True
        return sides

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

    def build_chirality_centres(self):
        """Returns the chirality centres, as (centre, three bonded atoms) index quadruples

        transferable-samplers' rule: an atom with four bonds of which at least three go to heavy
        atoms, so a centre carrying two hydrogens (whose mirror image is a relabelling) is left out.
        """
        neighbours = {atom.index: [] for atom in self.topology.atoms()}
        for bond in self.topology.bonds():
            neighbours[bond[0].index].append(bond[1])
            neighbours[bond[1].index].append(bond[0])
        centres = []
        for index, bonded in neighbours.items():
            heavy = [atom for atom in bonded if atom.element.symbol != 'H']
            if len(bonded) == 4 and len(heavy) > 2:
                centres.append([index] + [atom.index for atom in sorted(bonded,
                                                                         key=lambda a: a.index)[:3]])
        return torch.as_tensor(centres, dtype=torch.int64)

    def chirality_signs(self, x):
        """Evaluates the handedness of every chirality centre, +1 or -1 per centre

        Args:
            * x (torch.Tensor of shape (n, dim)): Configurations

        Returns:
            * signs (torch.Tensor of shape (n, n_centres)): Sign of the triple product of the
                three bond vectors at each centre
        """
        positions = self.positions(x)
        bonds = positions[:, self.chirality_centres[:, 1:]] \
            - positions[:, self.chirality_centres[:, :1]]
        volume = (bonds[:, :, 0] * torch.cross(bonds[:, :, 1], bonds[:, :, 2], dim=-1)).sum(-1)
        return torch.sign(volume)

    def fix_chirality(self, samples, ref_samples, weights=None):
        """Maps every sample of the wrong handedness onto its mirror image, as upstream does

        The energy is invariant under x -> -x, so a sampler that visits the mirror basin is exact
        and must not be scored as wrong; a sample whose centres disagree with each other has no
        mirror image in the reference and stays as it is.

        Args:
            * samples (torch.Tensor of shape (n, dim)): Samples to score
            * ref_samples (torch.Tensor of shape (m, dim)): Reference samples
            * weights (torch.Tensor of shape (n,) or None): Weights of the samples
                (default is None)

        Returns:
            * samples (torch.Tensor of the same shape): Samples, mirrored where needed
            * metrics (dict): 'correct_chirality_rate', the (weighted) fraction already of the
                reference's handedness, and 'uncorrectable_chirality_rate', the fraction a mirror
                image does not fix
        """
        reference = torch.sign(self.chirality_signs(ref_samples[:100]).sum(0))
        wrong = (self.chirality_signs(samples) != reference).any(-1)
        mirrored = torch.where(wrong[:, None], -samples, samples)
        still_wrong = (self.chirality_signs(mirrored) != reference).any(-1)
        mass = normalize_weights(weights, n=len(samples), dtype=samples.dtype,
                                 device=samples.device).to(samples.dtype)
        return mirrored, {'correct_chirality_rate': 1.0 - float((mass * wrong).sum()),
                          'uncorrectable_chirality_rate': float((mass * still_wrong).sum())}

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

        Chunked like the energy: AAAAAA has 1953 atom pairs, so the displacements of a whole
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

    def set_torsions(self, x, columns, angles):
        """Rotates torsions to the given angles, carrying the atoms on the far side of each bond

        A rotation about a torsion's central bond is rigid, so it keeps every bond length, bond
        angle and chirality centre, and leaves every other torsion where it was.

        Args:
            * x (torch.Tensor of shape (n, dim)): Configurations
            * columns (list of int): Torsions to set, as rows of torsion_indexes
            * angles (torch.Tensor of shape (n, len(columns))): Their new values, in radians

        Returns:
            * x (torch.Tensor of shape (n, dim)): The rotated configurations
        """
        positions = self.positions(x).clone()
        for k, column in enumerate(columns):
            _, b, c, _ = self.torsion_indexes[column].tolist()
            side = self.torsion_sides[column]
            delta = angles[:, k] - dihedral(positions, self.torsion_indexes[column][None])[:, 0]

            # Rodrigues' rotation of the side about the unit bond axis, around atom b
            axis = positions[:, c] - positions[:, b]
            axis = (axis / torch.linalg.norm(axis, dim=-1, keepdim=True))[:, None]
            v = positions[:, side] - positions[:, b, None]
            cos, sin = torch.cos(delta)[:, None, None], torch.sin(delta)[:, None, None]
            v = v * cos + torch.cross(axis.expand_as(v), v, dim=-1) * sin \
                + axis * (axis * v).sum(-1, keepdim=True) * (1 - cos)
            positions[:, side] = positions[:, b, None] + v
        return positions.reshape(len(x), -1)

    def load_split(self, split, n=None):
        """Reads one of the three released splits, centred

        Args:
            * split (str): 'train', 'val' or 'test'
            * n (int or None): Number of leading frames; None reads all (default is None)

        Returns:
            * x (torch.Tensor of shape (n, dim)): Structures in nanometres, centred
        """
        assert split in ['train', 'val', 'test']
        name = self.trajectory_name + '/' + self.trajectory_name + '_' + split + '.npy'
        array = np.load(fetch_hf(SBG_REPO, name, SBG_REVISION, SBG_CHECKSUMS[name]), mmap_mode='r')
        if n is not None:
            array = array[:n]
        x = torch.as_tensor(np.array(array).reshape(len(array), -1), dtype=self.dtype,
                            device=self.device)
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

    def tica_model(self):
        """Returns this system's TICA projection, fitting it once on the released test split

        transferable-samplers fits its projection on the whole released test split in file order,
        which is time-ordered, before subsampling anything, and the fit here does the same so the
        TICA metrics are theirs. TICA is defined through a time-lagged covariance, so the frames
        have to be time-ordered: the released Ace-A-Nme train split, for one, is shuffled. The
        projection is cached; the cache name carries TICA_LAYOUT because a projection is only
        valid for the exact feature ordering it was fitted on, and a reordering keeps the
        shape.

        Returns:
            * model (TicaModel): The fitted projection, on this target's device
        """
        if getattr(self, '_tica_model', None) is None:
            path = cache_dir() / 'tica' / (self.name + '-' + TICA_LAYOUT + '-test.npz')
            if path.exists():
                stored = np.load(path)
                model = TicaModel(torch.as_tensor(stored['mean'], dtype=self.dtype),
                                  torch.as_tensor(stored['projection'], dtype=self.dtype))
            else:
                reference = self.load_split('test')
                features = torch.cat([self.tica_features(block)
                                      for block in reference.split(65536)])
                model = fit_tica(features, lagtime=TICA_LAGTIME)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, mean=model.mean.cpu().numpy(),
                         projection=model.projection.cpu().numpy())
            self._tica_model = model.to(self.device)
        return self._tica_model

    def compute_upstream_metrics(self, samples, ref_samples, weights=None,
                                 max_ot_samples=2000, n_clusters=20, energy=None, ref_energy=None,
                                 generator=None, return_hist=False):
        """The molecular panel of transferable-samplers, in batched torch

        Their five scalars, under their names: energy-w1 and energy-w2 (exact 1D transport on the
        energies), torus-w2 (exact transport on all phi and psi at once, with the per-coordinate
        distance min(|d|, 2 pi - |d|)), tica-w2 (exact transport on the first two TICA components)
        and the two k-means Jensen-Shannon divergences, on TICA coordinates and on the
        sin/cos-wrapped dihedrals. Every feature is computed on device; only the transport linear
        programs and the one-off TICA fit leave it.

        WARNING: tica_k_jsd barely sees the alpha-R / beta split, a blind spot inherited from
        upstream. k-means runs on the raw TICA coordinates, as upstream runs it, and TIC 1 spreads
        far wider than TIC 2, so the clusters are bands across TIC 1 while alpha-R and beta differ
        along TIC 2: a sample missing alpha-R entirely barely moves it. Read the basin populations
        off torus_k_jsd and the torsion panel.

        NOTE: exact optimal transport is O(n^2) in memory, so torus_w2, tica_w2 and tica_k_jsd run
        on max_ot_samples rows per side drawn by metrics.weights.subsample, the draw the standard
        panel makes: a uniform subset without weights, and a resample in proportion to the weights
        with them, so the rows that carry the weight are the ones scored wherever they sit. The
        count used is reported as n_ot_samples rather than left implicit. The two energy distances
        and torus_k_jsd use every sample with its weight.

        NOTE: samples of the other handedness are first mapped onto their mirror image (see
        fix_chirality), and the two chirality rates are reported with the panel.

        Args:
            * samples (torch.Tensor of shape (n, dim)): Samples to score
            * ref_samples (torch.Tensor of shape (m, dim)): Reference samples
            * weights (torch.Tensor of shape (n,)): Weights of the samples (default is None)
            * max_ot_samples (int): Rows per side of the transport problems (default is 2000)
            * n_clusters (int): Clusters for the two JSD metrics (default is 20)
            * energy (torch.Tensor of shape (n,) or None): Energies of samples in kT, so a caller
                that already has them does not pay for a second force-field pass
                (default is None)
            * ref_energy (torch.Tensor of shape (m,) or None): Energies of ref_samples
                (default is None)
            * generator (torch.Generator or None): CPU RNG of the subsample; None seeds one at 0,
                so a reported number is reproducible (default is None)
            * return_hist (bool): Whether to also return every histogram (default is False)

        Returns:
            * metrics (dict): 'energy_w1', 'energy_w2', 'torus_w2', 'tica_w2', 'tica_k_jsd',
                'torus_k_jsd', 'n_ot_samples', 'correct_chirality_rate' and
                'uncorrectable_chirality_rate'
            if return_hist:
                * hist (dict): 'energy', 'distances', 'com_norm', 'ramachandran' and 'tica', each
                    holding 'edges' and the two histograms 'hist' and 'ref_hist'. Every one of
                    them is a probability summing to one, the 1D ones over their bins and the 2D
                    ones over the plane; 'ramachandran' holds one array per residue, in the order
                    of ramachandran_residues, and 'tica' is binned from the n_ot_samples rows
        """
        samples, weights = self.cast(samples), self.cast(weights)
        ref_samples = self.cast(ref_samples)
        samples, chirality = self.fix_chirality(samples, ref_samples, weights=weights)

        # The unweighted rows the transport problems and the TICA clustering run on
        if generator is None:
            generator = torch.Generator().manual_seed(0)
        n = min(len(samples), len(ref_samples), max_ot_samples)
        cut = subsample(samples, n, weights=weights, generator=generator)
        ref_cut = subsample(ref_samples, n, generator=generator)

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
        metrics['torus_w2'] = torus_wasserstein(ref_angles, angles, power=2)

        # The first two TICA components, and the two k-means divergences
        model = self.tica_model()
        tics = model(self.tica_features(cut).detach())
        ref_tics = model(self.tica_features(ref_cut).detach())
        metrics['tica_w2'] = compute_wasserstein_distance(ref_tics, tics, power=2)
        metrics['tica_k_jsd'] = kmeans_jsd(ref_tics, tics, n_clusters=n_clusters)[0]
        all_angles = self.torsions(samples).detach()
        ref_all_angles = self.torsions(ref_samples).detach()
        metrics['torus_k_jsd'] = kmeans_jsd(wrap_angles(ref_all_angles), wrap_angles(all_angles),
                                            n_clusters=n_clusters, weights=weights)[0]
        metrics['n_ot_samples'] = float(n)
        metrics.update(chirality)

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
        hist['tica'] = {'edges': edges, 'hist': histogram_2d(tics, edges),
                        'ref_hist': histogram_2d(ref_tics, edges)}
        return metrics, hist

    def compute_torsion_metrics(self, samples, ref_samples, weights=None, n_bins=64,
                                n_bins_joint=32, n_bins_plot=200, return_hist=False):
        """Compare the backbone torsions against a reference set

        The panel the molecular literature reports (Havens et al. 2026, OuYang et al. 2026): a 1D
        KL and Wasserstein distance per torsion marginal, and the Jensen-Shannon divergence of the
        2D Ramachandran joint. Torsion metrics come first and the energy ones second, because a
        handful of high-energy structures dominate the energy comparison while telling you little
        about the conformational ensemble. The Wasserstein distance is the circular one: omega sits
        on the -pi / pi cut, which a linear W1 treats as the two ends of its range.

        WARNING: a histogram divergence is only a distance when the bins are populated. The
        corpus's 200 x 200 Ramachandran grid, its plot convention, is 98% empty on 500 samples,
        where js_ramachandran reads ~0.61 for two halves of one ensemble. The divergences
        therefore default to 64 bins per marginal and a 32 x 32 joint, return_hist gives the
        200 x 200 histograms back, and 'ramachandran_occupancy' reports the fraction of reference
        cells that carry any mass: read a divergence next to it, or prefer the binning-free w1_
        entries.

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
            metrics['w1_' + name] = float(circular_wasserstein_1d(ref_angles[:, index],
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

        Every sample of the reference's mirror handedness is first mapped onto its mirror image
        (see fix_chirality), and the two chirality rates are reported.

        Arguments and returns are Target.compute_metrics'. generator draws the rows of the
        upstream panel's transport problems (see compute_upstream_metrics). The histogram dict
        holds 'torsions', compute_torsion_metrics' histograms; 'dist' and 'energy', the structural
        panel's, on edges derived from the reference; and 'upstream', compute_upstream_metrics'
        own dict, whose 'energy' is binned on edges spanning both sets.
        """
        samples, weights = self.cast(samples), self.cast(weights)
        ref_samples = self.cast(ref_samples)
        ref_samples = self.get_ref_samples(samples, ref_samples)
        samples, chirality = self.fix_chirality(samples, ref_samples, weights=weights)
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

        # The upstream panel, which a peptide result is compared on
        upstream, upstream_hist = self.compute_upstream_metrics(samples, ref_samples,
                                                               weights=weights, energy=energy,
                                                               ref_energy=ref_energy,
                                                               generator=generator,
                                                               return_hist=True)
        metrics.update(upstream)

        # The upstream panel re-applies the fix to samples that are already fixed, so its rates
        # are replaced by the ones measured on the samples as given
        metrics.update(chirality)
        hist['torsions'] = torsion_hist
        hist['upstream'] = upstream_hist
        ret.update(metrics)
        if return_hist:
            return ret, hist
        return ret

    def test_samples(self, n=None, seed=None):
        """Returns this library's own replica-exchange ensemble

        Generated by gen/md/generate.py and published in this library's dataset repo, so the first
        call downloads it (see data.fetch_reference) and later calls read the cache.

        NOTE: load_split() reads the released files, on which the TICA projection is fitted. They
        are correlated trajectories, worth far fewer independent structures than their frame
        counts, and are not a reference set.

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
        command = ('python gen/md/generate.py --sequence ' + self.sequence
                   + ' (from a clone of the repository)')
        path = fetch_reference(
            self.reference_folder, self.reference_key(),
            'Generate one with: ' + command + '.'
        )
        array = read_rows(path, n, command + ' --shard <k> --n_shards <m> for every k < m, then '
                          'python gen/merge_shards.py --folder md --target ' + self.name
                          + ' --min_samples <n>')
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    def minimised_positions(self, x0=None):
        """Returns the energy-minimised structure, which is where a trajectory has to start

        WARNING: reference_positions is the released pdb verbatim, which is not a minimum of this
        force field. Dynamics started from it can blow the molecule apart without raising, because
        linlogcut keeps every energy above energy_cut finite. This is the minimisation OpenMM's
        Simulation runs before its own dynamics, on the Reference platform.

        Args:
            * x0 (numpy.ndarray of shape (n_atoms, 3) or None): Starting positions in nanometres;
                None uses reference_positions (default is None)

        Returns:
            * x (torch.Tensor of shape (dim,)): The minimised structure, in nanometres
        """
        import openmm

        if x0 is None:
            x0 = self.reference_positions
        simulation = openmm.app.Simulation(self.topology, self.system,
                                           openmm.VerletIntegrator(0.001),
                                           openmm.Platform.getPlatformByName('Reference'))
        simulation.context.setPositions(x0)
        simulation.minimizeEnergy()
        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)._value.reshape(-1)
        return torch.as_tensor(positions, dtype=self.dtype, device=self.device)

    def generate_train_samples(self, n, seed=0, x0=None, n_chains=1000, n_steps=10000, thin=50,
                               temperatures=None, restrain_steps=2000, restraint=20.0,
                               descent_steps=500):
        """Runs the basin recipe: short dynamics from every backbone basin, in the wrong proportions

        Trajectory i starts from the energy-minimised released structure (minimised_positions).
        Every residue with a phi or a psi draws one of train_basins with the shares
        train_basin_shares, and set_torsions rotates its phi and psi to that basin's centre, which
        places it in the basin without crossing a barrier. A steepest descent of descent_steps
        steps of at most 1e-3 nm, then restrain_steps of Langevin dynamics (BAOAB, the released
        friction, 1 fs), relax the rotation while a restraint of restraint * (1 - cos(deviation))
        kT holds every phi and psi at its centre and every omega trans. The restraint is released,
        and the trajectory runs n_steps at this system's temperature times factor_i, the factors
        running geometrically over train_temperatures with the trajectory index
        (temperature_ladder), before keeping one frame every thin steps until the trajectories hold
        n between them. The split is those frames frame by frame, centred.

        Every basin of every residue is visited, alpha-L far more often than under the target,
        and inside a basin the shape is the target's. The recipe takes seconds on a GPU once
        compiled, and minutes on the CPU: run it on a GPU, or with fewer trajectories.

        NOTE: the restraint differentiates the torsions with autograd, like the force field itself.

        Args:
            * n (int): Number of samples
            * seed (int): Seed of the recipe (default is 0)
            * x0 (torch.Tensor of shape (m, dim) or None): Starting structures, trajectory i
                starting from x0[i % m]; None uses the minimised released structure
                (default is None)
            * n_chains (int): Number of trajectories, at most n (default is 1000)
            * n_steps (int): Steps after the release before any frame is kept, 10 ps
                (default is 10000)
            * thin (int): Steps between kept frames (default is 50)
            * temperatures (tuple of two floats or None): Lowest and highest temperature factor;
                None uses train_temperatures (default is None)
            * restrain_steps (int): Steps of restrained dynamics, 2 ps; 0 skips the placement, the
                descent and the restraint, which is how a split continues from x0
                (default is 2000)
            * restraint (float): Strength of the torsion restraint, in kT (default is 20.0)
            * descent_steps (int): Steps of the restrained descent (default is 500)

        Returns:
            * x (torch.Tensor of shape (n, dim)): The biased samples, centred
        """
        gen = generator(seed, device=self.device)
        n_chains, n_frames = chain_layout(n, n_chains)
        if x0 is None:
            x0 = self.minimised_positions()[None]
        x = cycle(self.cast(x0), n_chains)
        if temperatures is None:
            temperatures = self.train_temperatures

        # The dynamics, whose force is kT times the score
        mass = self.masses.repeat_interleave(3)
        kT = self.kT * temperature_ladder(n_chains, temperatures, dtype=self.dtype,
                                          device=self.device)[:, None]
        force = lambda y: self.kT * self.score(y)
        run = lambda f, y, steps, **kwargs: langevin(f, y, steps, 0.001, mass, kT, self.friction,
                                                     generator=gen, **kwargs)
        velocities = None

        if restrain_steps > 0:
            # Every residue draws a basin, and its phi and psi take that basin's centre
            residues = sorted(set(self.backbone_residues))
            shares = torch.tensor(self.train_basin_shares, dtype=self.dtype, device=self.device)
            basins = torch.multinomial(shares / shares.sum(), n_chains * len(residues),
                                       replacement=True, generator=gen)
            basins = basins.reshape(n_chains, len(residues))
            centres = torch.deg2rad(torch.tensor(self.train_basins, dtype=self.dtype,
                                                 device=self.device))
            angles = torch.stack([centres[basins[:, residues.index(residue)], int(k >= self.n_phi)]
                                  for k, residue in enumerate(self.backbone_residues)], -1)
            x = self.set_torsions(x, list(range(len(self.backbone_residues))), angles)

            # Hold every phi and psi at its centre and every omega trans
            n_omega = len(self.torsion_indexes) - len(self.backbone_residues)
            targets = torch.cat([angles, torch.full((n_chains, n_omega), math.pi,
                                                    dtype=self.dtype, device=self.device)], -1)

            def restrained_force(y):
                with torch.enable_grad():
                    z = y.detach().requires_grad_(True)
                    penalty = restraint * (1 - torch.cos(self.torsions(z) - targets))
                    (gradient,) = torch.autograd.grad(penalty.sum(), z)
                return force(y) - self.kT * gradient

            # Relax the rotation, then thermalise with the torsions held
            x = steepest_descent(restrained_force, x, descent_steps, 1e-3)
            x, velocities = run(restrained_force, x, restrain_steps, return_velocities=True)

        # Release the torsions, then keep the frames
        x, velocities = run(force, x, n_steps, velocities=velocities, return_velocities=True)
        trace = run(force, x, n_frames * thin, velocities=velocities, return_trace=True, thin=thin)
        return center(interleave(trace, n), self.n_atoms)


def _angle_edges(n_bins, like):
    """Returns n_bins + 1 edges spanning the full angular range, matching a tensor's dtype"""
    return torch.linspace(-math.pi, math.pi, n_bins + 1, dtype=like.dtype, device=like.device)
