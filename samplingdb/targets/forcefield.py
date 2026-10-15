# Batched torch force field, whose parameters are read out of an OpenMM system

# Libraries
import re
import numpy as np
import torch

# Molar gas constant in kJ / (mol K), so kT can be formed without importing openmm.unit. Every
# digit of OpenMM's own MOLAR_GAS_CONSTANT_R, since a truncation here biases every energy in the
# library by the same relative amount
GAS_CONSTANT = 0.00831446261815324

# OpenMM's internal Coulomb constant, and the different literal its app-level GB expressions
# carry. They differ in the 8th digit, so using one for both leaves a 2e-7 relative error on the
# Coulomb sum -- of order 1e-4 kJ/mol on these systems, and the entire residual until it was found
ONE_4PI_EPS0 = 138.935456
GB_4PI_EPS0 = 138.935485

# Squared length added under every square root. At any real interatomic distance this sits far
# below double-precision resolution, so it changes no energy this library reports; it exists so a
# coincident pair gives a huge finite number instead of a nan gradient.
MIN_SQUARE = 1e-24

# Dielectric offset and probe radius of the OBC models, in nm
OBC_OFFSET = 0.009
OBC_PROBE = 0.14

# Live (chunk, n_atoms, n_atoms) matrices one chunk holds for an energy AND gradient pass, counted
# in elements rather than bytes: the COUNT is dtype-independent and only the width changes, so
# float32 fits twice the chunk. Assuming 8 bytes here silently halved the chunk in single precision.
# The count bounds the memory of a large batch. torch.compile is the dominant variable, not
# the system: inductor fuses the elementwise chain and holds ~8x fewer, and only once compiled
# does the nonbonded branch separate, a NoCutoff system keeping about twice what a cutoff one does
# because none of its pair terms are masked away. Measured unchunked at batch 1024 over all five
# peptides -- 28.3-30.1 eager, 7.1-7.2 compiled without a cutoff, 3.3-3.5 compiled with one -- and
# rounded up. Only an UNCHUNKED call measures this, since dividing a chunked call's peak by its
# batch measures the chunk size instead
MATRICES_EAGER = 31.0
MATRICES_COMPILED_NO_CUTOFF = 7.5
MATRICES_COMPILED_CUTOFF = 3.6

# Working memory one chunk may hold, which is what keeps a 10^5-frame reference set from
# allocating the whole GPU. There is no throughput reason to go higher: cost per configuration is
# flat from batch 512 to 8192 on both branches, chignolin at 57.6 / 56.2 / 56.1 us and Ace-A-Nme
# at 2.71 / 2.00 / 1.89 us for 512 / 2048 / 8192, so past ~2048 a larger chunk buys memory
# pressure and nothing else. 3 GiB puts chignolin's chunk at ~5900 configurations
MEMORY_BUDGET = 3 * 1024 ** 3

# The forces this module knows how to evaluate. CMMotionRemover carries no potential energy.
KNOWN_FORCES = ['HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce',
                'NonbondedForce', 'GBSAOBCForce', 'CustomGBForce', 'CMMotionRemover']

# The OBC1 rescaling the Amber app writes into its CustomGBForce, checked so that a change in
# OpenMM's expression is a loud failure rather than a silently different density
OBC1_EXPRESSION = 'tanh(0.8*psi+2.909125*psi^3)'


def linlogcut(values, high_val=1e8, max_val=1e20):
    """Softens energies above high_val logarithmically and clamps them at max_val

    This is bgflow's linlogcut, which FAB applies to the ALDP energy with
    (energy_cut, energy_max) = (1e8, 1e20); those two constants are part of the target's
    specification, not of a training loop. Physical configurations never reach 1e8 kT, so the
    transform is the identity on anything the reference data contains -- it exists so a sampler
    that walks into an atomic overlap gets a large finite number instead of an inf.

    NOTE: the excess is clamped before the logarithm rather than after, so the branch that is not
    taken evaluates log1p(0) instead of the logarithm of a negative number. Both branches of a
    torch.where are always evaluated, and a nan in the dead one would poison the gradient.

    Args:
        * values (torch.Tensor): Energies, in kT
        * high_val (float): Value above which the energy is logarithmically scaled
            (default is 1e8)
        * max_val (float): Value the result is clamped to (default is 1e20)

    Returns:
        * values (torch.Tensor of the same shape): The softened energies
    """
    excess = (values - high_val).clamp(min=0.0)
    cut = torch.where(values >= high_val, high_val + torch.log1p(excess), values)
    return cut.clamp(max=max_val)


def norm(delta):
    """Evaluates the Euclidean norm along the last axis, twice differentiably at zero

    torch.linalg.norm defines its first derivative at the origin to be zero but its second one is
    a nan, and the diagonal of a distance matrix is exactly zero, so every distance in this module
    goes through here instead.

    Args:
        * delta (torch.Tensor of shape (..., 3)): Displacement vectors

    Returns:
        * r (torch.Tensor of shape (...)): Their lengths
    """
    return torch.sqrt((delta ** 2).sum(-1) + MIN_SQUARE)


def _constant(expression, name):
    """Reads a '; name=value' constant out of a CustomGBForce energy expression"""
    match = re.search(name + r'=(-?[0-9.eE+-]+)', expression)
    assert match is not None, name + ' is not pinned in the OpenMM expression: ' + expression
    return float(match.group(1))


def extract_parameters(system):
    """Reads every force's parameters out of an openmm.System into numpy arrays

    OpenMM stays the authority on force-field files, atom typing and topology: nothing here
    re-derives a parameter, it only copies what createSystem already built. Anything this module
    cannot evaluate raises rather than being silently dropped from the energy.

    Args:
        * system (openmm.System): The system to read

    Returns:
        * parameters (dict): Arrays and scalars naming every term of the potential
    """
    import openmm

    assert system.getNumConstraints() == 0, 'constrained systems are not supported'

    # Defaults for a system with no implicit solvent at all -- a vacuum peptide is a legitimate
    # target, and gb_model None is what tells the potential to skip the Born term entirely. The
    # two dielectrics are equal so that gb_factor is exactly zero even if one is read
    parameters = {'gb_model': None, 'cutoff': None, 'gb_cutoff': None, 'gb_surface': 0.0,
                  'gb_constant': ONE_4PI_EPS0, 'gb_solute': 1.0, 'gb_solvent': 1.0}

    for index in range(system.getNumForces()):
        force = system.getForce(index)
        name = force.__class__.__name__
        assert name in KNOWN_FORCES, 'no torch implementation of ' + name

        if name == 'HarmonicBondForce':
            rows = [force.getBondParameters(i) for i in range(force.getNumBonds())]
            parameters['bond_index'] = np.array([[r[0], r[1]] for r in rows])
            parameters['bond_length'] = np.array([r[2]._value for r in rows])
            parameters['bond_k'] = np.array([r[3]._value for r in rows])

        elif name == 'HarmonicAngleForce':
            rows = [force.getAngleParameters(i) for i in range(force.getNumAngles())]
            parameters['angle_index'] = np.array([[r[0], r[1], r[2]] for r in rows])
            parameters['angle_theta'] = np.array([r[3]._value for r in rows])
            parameters['angle_k'] = np.array([r[4]._value for r in rows])

        elif name == 'PeriodicTorsionForce':
            rows = [force.getTorsionParameters(i) for i in range(force.getNumTorsions())]
            parameters['torsion_index'] = np.array([r[:4] for r in rows])
            parameters['torsion_n'] = np.array([r[4] for r in rows], dtype=np.float64)
            parameters['torsion_phase'] = np.array([r[5]._value for r in rows])
            parameters['torsion_k'] = np.array([r[6]._value for r in rows])

        elif name == 'NonbondedForce':
            method = force.getNonbondedMethod()
            assert method in [openmm.NonbondedForce.NoCutoff,
                              openmm.NonbondedForce.CutoffNonPeriodic], 'periodic box unsupported'
            assert not force.getUseSwitchingFunction(), 'switching functions are not supported'
            assert force.getNumParticleParameterOffsets() == 0, 'parameter offsets are unsupported'
            assert force.getNumExceptionParameterOffsets() == 0, 'parameter offsets are unsupported'
            if method == openmm.NonbondedForce.CutoffNonPeriodic:
                parameters['cutoff'] = force.getCutoffDistance()._value
                parameters['rf_dielectric'] = force.getReactionFieldDielectric()
            rows = [force.getParticleParameters(i) for i in range(force.getNumParticles())]
            parameters['charge'] = np.array([r[0]._value for r in rows])
            parameters['sigma'] = np.array([r[1]._value for r in rows])
            parameters['epsilon'] = np.array([r[2]._value for r in rows])
            rows = [force.getExceptionParameters(i) for i in range(force.getNumExceptions())]
            # The dtype and the shape are explicit because np.array([]) is a float64 of shape (0,),
            # which would register exception_index as a float buffer and then fail as a tensor
            # index. Unreachable for any molecule with bonds, but it costs nothing to be right
            parameters['exception_index'] = np.array([[r[0], r[1]] for r in rows],
                                                     dtype=np.int64).reshape(-1, 2)
            parameters['exception_charge'] = np.array([r[2]._value for r in rows])
            parameters['exception_sigma'] = np.array([r[3]._value for r in rows])
            parameters['exception_epsilon'] = np.array([r[4]._value for r in rows])

        elif name == 'GBSAOBCForce':
            # OpenMM's built-in OBC is model II, whose rescaling is
            # tanh(psi - 0.8 psi^2 + 4.85 psi^3)
            parameters['gb_model'] = 'obc2'
            rows = [force.getParticleParameters(i) for i in range(force.getNumParticles())]
            parameters['gb_charge'] = np.array([r[0]._value for r in rows])
            radius = np.array([r[1]._value for r in rows])
            parameters['gb_or'] = radius - OBC_OFFSET
            parameters['gb_sr'] = np.array([r[2] for r in rows]) * (radius - OBC_OFFSET)
            parameters['gb_solvent'] = force.getSolventDielectric()
            parameters['gb_solute'] = force.getSoluteDielectric()
            parameters['gb_surface'] = force.getSurfaceAreaEnergy()._value * 4 * np.pi
            parameters['gb_constant'] = ONE_4PI_EPS0
            if force.getNonbondedMethod() != openmm.GBSAOBCForce.NoCutoff:
                parameters['gb_cutoff'] = force.getCutoffDistance()._value

        elif name == 'CustomGBForce':
            # The Amber app's OBC1, built in openmm/app/internal/customgbforces.py
            parameters['gb_model'] = 'obc1'
            assert force.getNumComputedValues() == 2, 'unexpected CustomGBForce'
            born = force.getComputedValueParameters(1)[1].replace(' ', '')
            assert OBC1_EXPRESSION in born, 'this CustomGBForce is not the app-level OBC1: ' + born
            terms = [force.getEnergyTermParameters(i)[0]
                     for i in range(force.getNumEnergyTerms())]
            assert _constant(terms[0], 'kappa') == 0.0, 'salt screening is not supported'
            rows = [force.getParticleParameters(i) for i in range(force.getNumParticles())]
            parameters['gb_charge'] = np.array([r[0] for r in rows])
            parameters['gb_or'] = np.array([r[1] for r in rows])
            parameters['gb_sr'] = np.array([r[2] for r in rows])
            parameters['gb_solvent'] = _constant(terms[0], 'solventDielectric')
            parameters['gb_solute'] = _constant(terms[0], 'soluteDielectric')
            surface = [t for t in terms if '28.3919551' in t]
            parameters['gb_surface'] = 28.3919551 if surface else 0.0
            parameters['gb_constant'] = GB_4PI_EPS0
            if force.getNonbondedMethod() != openmm.CustomGBForce.NoCutoff:
                parameters['gb_cutoff'] = force.getCutoffDistance()._value

    return parameters


class TorchForceField(torch.nn.Module):
    """Evaluates an Amber + implicit-solvent potential for a whole batch of configurations

    Positions are in nanometres and energies in kT, the conventions of every reference
    implementation in this corpus (transferable-samplers' OpenMMEnergy, bgflow's OpenMMBridge).

    This replaces the per-configuration OpenMM Context the library used to evaluate through. An
    OpenMM Context holds ONE configuration, so a batch cost a Python loop and no two energies were
    ever computed at the same time; here the whole batch is one set of tensor operations. Measured
    on this machine, energy and forces together, in float64 (gen/openmm_check/compare.py):

    | system     | atoms | OpenMM Reference | OpenMM CUDA | 32-process CPU pool | this, compiled |
    | ---        | ---   | ---              | ---         | ---                 | ---            |
    | Ace-A-Nme  |  22   | 18 900 /s        | 3 000 /s    | 30 500 /s           | 473 000 /s     |
    | GYDPETGTWG | 138   |    230 /s        |   950 /s    |  7 200 /s           |  18 200 /s     |

    NOTE: the score is where this loses ground per configuration. OpenMM's forces are essentially
    free -- hand-written in the kernel for the standard forces, and symbolically differentiated by
    Lepton at Context creation for the CustomGBForce -- so asking for them on top of the energy
    costs it 1.0x to 1.3x. Autograd costs 2.2x to 3.5x. Batching wins by so much more than that
    factor that the comparison above already includes forces on both sides.

    WARNING: every parameter is read out of the openmm.System, never re-derived, so OpenMM remains
    the authority on force-field files, atom typing and topology. What is reimplemented is the
    summation, not the force field. gen/openmm_check/compare.py asserts the agreement, which is
    ~1e-5 kJ/mol/nm on forces of order 4e3 -- five orders of magnitude tighter than the difference
    between OpenMM's own platforms.

    NOTE: the only platform that ground-truth means anything against is Reference. CUDA and OpenCL
    need Precision=double, and the CPU platform evaluates its nonbonded and GB terms in single
    precision INTERNALLY whatever it is asked for, so it is not a double-precision reference no
    matter how it is configured. Reference is also the only deterministic one.

    NOTE: an earlier version of this note said that below ~0.02 nm OpenMM's built-in
    GBSAOBCForce and its own CustomGBForce rendering of OBC2 disagree because they group the Born
    integral differently, and that this module followed the CustomGBForce. The disagreement was
    real and the explanation was wrong: the two are different models, the built-in one carrying
    Ponder's inside-atom correction. born_radii now follows each force it is read from; see its
    docstring.

    NOTE: at a near-collinear angle this module deliberately does NOT match OpenMM, and is the
    more physical of the two. OpenMM's HarmonicAngleForce clamps its cosine off +-1 exactly as an
    earlier version of bonded() did, so its angle force drops ~31% at pi - theta ~ 1e-6 where the
    clamp bites; the atan2 form here stays continuous through it. Both agree to 1e-4 kJ/mol/nm at
    pi - theta >= 1e-4, which is every geometry short of a clash.

    NOTE: unlike every other target in this library the score here is autograd rather than
    hand-written algebra. It is still exact: the energy is a closed-form expression and its
    derivative is machine-precision, not a finite difference. Hand-written first derivatives were
    the wrong trade because the score has to be ABLE to stay differentiable -- a loss may
    differentiate through it -- and that would mean hand-writing the second derivatives too.

    NOTE: being able to is not the same as always doing it. Retaining the backward graph is only
    paid for when differentiable_score is set, which is off by default; see evaluate().
    """

    def __init__(self, system, temperature, energy_cut=1e8, energy_max=1e20, chunk_size=None,
                 compile=True, differentiable_score=False, dtype=torch.float32):
        """Builds the force field

        Args:
            * system (openmm.System): The system whose parameters are copied
            * temperature (float): Temperature in kelvin, which fixes kT
            * energy_cut (float): linlogcut's high_val, in kT (default is 1e8)
            * energy_max (float): linlogcut's max_val, in kT (default is 1e20)
            * chunk_size (int or None): Configurations evaluated at once. None derives one from
                MEMORY_BUDGET through suggested_chunk_size, which is what keeps a 10^5-frame
                reference set from allocating the whole GPU (default is None)
            * compile (bool): Whether to torch.compile the potential, which is worth ~2.2x on the
                GPU and is the right default for a run of any length. Turn it OFF when:
                  - the run is short or interactive. Compilation costs ~15 s per distinct batch
                    shape on the GPU and ~20 s on the CPU, so a handful of batches never repays it
                  - the batch shape keeps changing. Two shapes are compiled before torch settles
                    into a dynamic kernel, and every new chunk size pays again
                  - a traceback has to be readable, or a number has to be bisected
                The eager potential is substituted whenever the graph has to survive the call, so
                a differentiable energy or score needs no flag; see evaluate() (default is True)
            * differentiable_score (bool): Whether the SCORE is kept differentiable, which is what
                a loss built on the score needs and what nothing else does. It is a plain
                attribute, so a training loop may switch it on later. The energy is differentiable
                either way (default is False)
            * dtype (torch.dtype): Working dtype, which the input is CAST to rather than promoted
                from (default is torch.float32)

                This default was float64 until 2026-08-25, on the strength of a measurement that
                float32 carried an error of 0.15 to 0.8 kT -- enough to move an importance weight by
                a factor of two. **That measurement was an artefact of a formulation this class no
                longer uses.** It was taken when the nonbonded sum kept every pair and subtracted
                the excluded ones back out; a bonded pair sits well inside sigma, so its
                (sigma/r)^12 is enormous and adding it in destroys the mantissa of every physical
                term before the subtraction can undo it. float64 has the digits to survive that,
                float32 does not. Exceptions are now dropped from the pair list instead (point 4 of
                the constructor), and the measurement had not been repeated since.

                Measured against float64 on 2048 released test frames: max |dE| is 1.2e-04 kT on
                Ace-A-Nme and 3.8e-04 kT on AAAAAA, term by term worst in the Born radius at
                1.0e-04 and 2.2e-04, and the force is off by 1.7e-05 to 3.9e-05 of its own rms.
                For scale, the 1 fs timestep's own bias is 0.08 kT -- some 250x larger -- so
                discretisation dominates precision by orders of magnitude. It buys 2.2x on
                Ace-A-Nme, 7.7x on AAAAAA and 11.0x on chignolin, and halves the memory, which
                doubles the chunk this class derives for itself.

                The ENSEMBLE is unchanged too, which is the claim that matters for a reference
                set: these are generated by an unadjusted integrator, where a force error biases the
                stationary law directly with no accept/reject to correct it. Measured on AAAAAA with
                FIVE independent float64 runs and three float32 ones, judged throughout by a float64
                energy: on the energy and all four cosine torsions, the median float32-against-
                float64 Wasserstein distance is LOWER than the median float64-against-float64 one.
                float32 agrees with float64 better than float64 agrees with itself.

                A single pair is not a floor. An earlier version of that test used one float64 pair
                and a torsion came out 4.1x above it, which reads as a bias; across pairs that same
                comparison ranges over a factor of ten.

                Use float64 when comparing against OpenMM, which resolves ~1e-6 kT and which
                float32 cannot reach, or when a number looks wrong and precision should be off the
                list of suspects. The validation chain is float64-against-OpenMM at 1e-9 relative,
                then float32-against-float64 at 1e-4 kT.
        """
        # Call the parent constructor
        super().__init__()
        parameters = extract_parameters(system)

        # 1. Everything that is not a tensor
        self.n_atoms = system.getNumParticles()
        self.kT = GAS_CONSTANT * temperature
        self.energy_cut = energy_cut
        self.energy_max = energy_max
        self.gb_model = parameters['gb_model']
        self.gb_surface = parameters['gb_surface']
        self.gb_cutoff = parameters['gb_cutoff']
        self.gb_factor = parameters['gb_constant'] * (1 / parameters['gb_solute']
                                                      - 1 / parameters['gb_solvent'])
        self.cutoff = parameters['cutoff']
        self.dtype = dtype
        self.differentiable_score = differentiable_score

        # 2. Reaction-field coefficients, which are what a non-periodic cutoff turns Coulomb into
        if self.cutoff is None:
            self.krf, self.crf = 0.0, 0.0
        else:
            dielectric = parameters['rf_dielectric']
            self.krf = (dielectric - 1) / (2 * dielectric + 1) / self.cutoff ** 3
            self.crf = 3 * dielectric / (2 * dielectric + 1) / self.cutoff

        # 3. Register every parameter array to the buffer, so .to(device) moves the whole potential
        for key, value in parameters.items():
            if isinstance(value, np.ndarray):
                tensor = torch.as_tensor(value)
                self.register_buffer(key, tensor if tensor.dtype == torch.int64
                                     else tensor.to(dtype))

        # 4. The upper-triangular pair list, whose exceptions are EXCLUDED rather than subtracted
        # back. An exception replaces the standard pair value, so summing the full triangle and
        # adding (replacement - standard) is algebraically identical and numerically fatal. The
        # standard Lennard-Jones term of a pair about to be replaced is no part of the true energy
        # and grows without bound: a 1-2 pair at 1e-2 nm already carries ~1e18 kJ/mol, where
        # float64 resolution is ~1e2 -- coarser than the whole physical nonbonded sum it is being
        # added to. That sum is then quantised away, the subtraction returns exactly zero, and the
        # nonbonded pair energy silently vanishes: measured against Reference on a 1-2 pair walked
        # from 1e-2 to 1e-5 nm, the error was 125 to 650 kJ/mol and sometimes DOWNWARD, which is a
        # spurious low-energy region rather than an honest inf. OpenMM builds an exclusion list for
        # the same reason. Only clashes show it, which is why it survived a validation restricted
        # to released frames -- the two formulas agree to 1e-6 kJ/mol on every one of those
        rows, cols = torch.triu_indices(self.n_atoms, self.n_atoms, offset=1)
        i, j = self.exception_index[:, 0], self.exception_index[:, 1]
        excluded = torch.minimum(i, j) * self.n_atoms + torch.maximum(i, j)
        keep = ~torch.isin(rows * self.n_atoms + cols, excluded)
        pair_rows, pair_cols = rows[keep], cols[keep]

        # 5. The nonbonded pair list and its Lorentz-Berthelot combinations
        self.register_buffer('pair_rows', pair_rows)
        self.register_buffer('pair_cols', pair_cols)
        self.register_buffer('pair_charge', self.charge[pair_rows] * self.charge[pair_cols])
        self.register_buffer('pair_sigma', 0.5 * (self.sigma[pair_rows] + self.sigma[pair_cols]))
        self.register_buffer('pair_epsilon',
                             torch.sqrt(self.epsilon[pair_rows] * self.epsilon[pair_cols]))
        self.register_buffer('eye', torch.eye(self.n_atoms, dtype=dtype))

        # 6. The Born term has no exclusions, so it keeps the WHOLE triangle and its own pair list
        self.register_buffer('gb_rows', rows)
        self.register_buffer('gb_cols', cols)
        if self.gb_model is not None:
            self.register_buffer('gb_radius', self.gb_or + OBC_OFFSET)
            self.register_buffer('gb_pair_charge', self.gb_charge[rows] * self.gb_charge[cols])

        # 7. The compiled potentials, built once and used wherever the graph need not survive
        self.compiled = torch.compile(self.potential) if compile else None
        self.compiled_harmonic = torch.compile(self.harmonic_potential) if compile else None

        # 8. Derived last, because the budget depends on whether the potential is compiled
        self.chunk_size = chunk_size or self.suggested_chunk_size()

    def bytes_per_configuration(self):
        """Evaluates the peak working memory of one configuration, energy and gradient, in bytes

        Keyed on whether the potential is compiled, which is worth ~8x, and then on the nonbonded
        branch, which only separates once compiled. Sized for energy AND gradient because
        evaluate() is the hot path and costs ~3x the energy alone.
        """
        if self.compiled is None:
            matrices = MATRICES_EAGER
        elif self.cutoff is None:
            matrices = MATRICES_COMPILED_NO_CUTOFF
        else:
            matrices = MATRICES_COMPILED_CUTOFF
        return int(matrices * self.n_atoms ** 2 * self.dtype.itemsize)

    def suggested_chunk_size(self, budget=MEMORY_BUDGET):
        """Returns how many configurations fit in budget bytes of working memory"""
        return max(1, budget // self.bytes_per_configuration())

    def bonded(self, positions):
        """Evaluates the harmonic bond, harmonic angle and periodic torsion terms, in kJ/mol"""
        return self.harmonic(positions) + self.torsion(positions)

    def harmonic(self, positions):
        """Evaluates the harmonic bond and angle terms alone, in kJ/mol

        These are the STIFF internal coordinates, and they are separated because they dominate the
        potential energy's spread while contributing none of its conformational barriers. Each
        quadratic degree of freedom contributes 1/2 kT^2 to the variance, so the stiff block alone
        would give sd(E) = sqrt((n_bonds + n_angles) / 2) -- which is within a few percent of the
        measured total on every peptide here, 5.34 against 5.40 on Ace-A-Nme and 7.52 against 8.17
        on Ace-AAA-Nme. Dropping it from the swap statistic takes sd down by 1.55x to 1.70x, and
        since the communication barrier density is sd / sqrt(pi) that is the ladder halving.
        Leaving these at full strength is what replica exchange with solute tempering does for the
        solvent (Wang, Friesner and Berne 2011) and what gREST does term by term (Kamiya and
        Sugita 2018).
        """
        # Bonds
        r = norm(positions[:, self.bond_index[:, 0]] - positions[:, self.bond_index[:, 1]])
        energy = (0.5 * self.bond_k * (r - self.bond_length) ** 2).sum(-1)

        # Angles, from the cross product rather than from acos of a cosine clamped off +-1. The
        # clamp saturates at pi - theta ~ 1.4e-6, and there it freezes theta and takes ~31% of the
        # angle force with it, discontinuously: measured on one Ace-A-Nme angle, |F| runs
        # 8402 -> 8424.5 -> 8424.7 through atan2 as pi - theta goes 1e-2 -> 1e-4 -> 1e-6, and
        # 8402 -> 8424.5 -> 5806 through the clamp. atan2 needs no clamp and stays continuous
        v1 = positions[:, self.angle_index[:, 0]] - positions[:, self.angle_index[:, 1]]
        v2 = positions[:, self.angle_index[:, 2]] - positions[:, self.angle_index[:, 1]]
        theta = torch.atan2(norm(torch.cross(v1, v2, dim=-1)), (v1 * v2).sum(-1))
        return energy + (0.5 * self.angle_k * (theta - self.angle_theta) ** 2).sum(-1)

    def torsion(self, positions):
        """Evaluates the periodic torsion terms alone, in kJ/mol"""
        b0 = positions[:, self.torsion_index[:, 0]] - positions[:, self.torsion_index[:, 1]]
        b1 = positions[:, self.torsion_index[:, 2]] - positions[:, self.torsion_index[:, 1]]
        b2 = positions[:, self.torsion_index[:, 3]] - positions[:, self.torsion_index[:, 2]]
        b1 = b1 / norm(b1).unsqueeze(-1)
        v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
        w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
        phi = torch.atan2((torch.cross(b1, v, dim=-1) * w).sum(-1), (v * w).sum(-1))
        return (self.torsion_k * (1 + torch.cos(self.torsion_n * phi
                                                - self.torsion_phase))).sum(-1)

    def pair_energy(self, r, charge_product, sigma, epsilon, reaction_field):
        """Evaluates one Lennard-Jones plus Coulomb pair term

        Args:
            * r (torch.Tensor of shape (n, n_pairs)): Pair distances
            * charge_product (torch.Tensor of shape (n_pairs,)): Products of the two charges
            * sigma (torch.Tensor of shape (n_pairs,)): Combined Lennard-Jones radii
            * epsilon (torch.Tensor of shape (n_pairs,)): Combined well depths
            * reaction_field (bool): Whether the cutoff and its reaction field apply, which they
                do for ordinary pairs and do not for the 1-4 exceptions

        Returns:
            * energy (torch.Tensor of shape (n, n_pairs)): The pair energies
        """
        # The cap has to sit on the BASE, and it has to come from the GRADIENT rather than the
        # energy. In float32 (sigma/r)^12 leaves the representable range around r = 4e-4 nm, and a
        # coincident pair then gives an inf energy and a NaN gradient where float64 stays finite.
        # Two things are easy to get wrong here, both measured:
        #
        # * Clamping the sixth power does NOT fix it. By then it has already overflowed to inf, and
        #   clamp passes zero gradient above its ceiling, so the backward is 0 * inf = NaN.
        # * A bound from base^12 keeps the ENERGY finite and still lets the gradient overflow,
        #   because d/dr of base^12 carries a further 12/r = 12 base / sigma. At r = 4e-4 nm the
        #   energy is 6.3e34 and the gradient 1.1e39, past float32's 3.4e38.
        #
        # base^13 is therefore what sets it. The bound is 230 in float32 and 1.3e23 in float64, so
        # it never binds in double precision -- MIN_SQUARE floors r at 1e-12 nm, where sigma/r
        # reaches only ~3e11 -- and it never binds on a real conformation either, where the closest
        # approach of ~0.05 nm puts sigma/r near 7.
        base = (sigma / r).clamp(max=0.25 * torch.finfo(r.dtype).max ** (1.0 / 13.0))
        ratio = base ** 6
        lennard_jones = 4 * epsilon * ratio * (ratio - 1)
        if reaction_field and self.cutoff is not None:
            coulomb = ONE_4PI_EPS0 * charge_product * (1 / r + self.krf * r ** 2 - self.crf)
            return (coulomb + lennard_jones) * (r < self.cutoff)
        return ONE_4PI_EPS0 * charge_product / r + lennard_jones

    def nonbonded(self, positions):
        """Evaluates the Coulomb and Lennard-Jones terms with OpenMM's exception handling"""
        r = norm(positions[:, self.pair_rows] - positions[:, self.pair_cols])
        energy = self.pair_energy(r, self.pair_charge, self.pair_sigma, self.pair_epsilon,
                                  True).sum(-1)

        # An exception REPLACES the standard pair value, which is already absent from pair_rows,
        # and carries no reaction field. See point 4 of the constructor for why it is not
        # subtracted back
        i, j = self.exception_index[:, 0], self.exception_index[:, 1]
        r = norm(positions[:, i] - positions[:, j])
        replacement = self.pair_energy(r, self.exception_charge, self.exception_sigma,
                                       self.exception_epsilon, False)
        return energy + replacement.sum(-1)

    def born_radii(self, r):
        """Evaluates the OBC Born radii from the full distance matrix

        The pairwise integral is the HCT one, with row i carrying its own offset radius and
        column j the scaled offset radius of its neighbour.

        NOTE: the obc2 branch adds Ponder's correction for atom i lying entirely inside atom j,
        2 (1 / or_i - 1 / L_ij) wherever or_i < sr_j - r_ij, and the obc1 branch does not. Each
        follows the force it is read from: obc2 comes only from OpenMM's built-in GBSAOBCForce,
        whose Reference kernel adds the term (ReferenceObc.cpp, computeBornRadii, OpenMM 8.5.2),
        and obc1 only from the Amber app's CustomGBForce, whose expression in
        openmm/app/internal/customgbforces.py omits it. Without it, Ace-A-Nme and Ace-AAA-Nme read
        0.6 / 6.5 / 12.7 kJ/mol and 0.9 / 8.2 / 14.3 kJ/mol BELOW Reference at 0.95 / 0.5 / 0.01 of
        the window, which is 0.0243 nm on both -- a clash made more attractive than it is. Nothing
        real changes: the closest approach in 20 000 released test frames of either system is
        0.089 nm, so the step fires on no pair of any reference set. The term is bounded by
        1 / or_i, since it fires only where L_ij = sr_j - r_ij > or_i, so it is safe in float32.
        This is the correction transferable-samplers' torch force field made in 16bf773.

        Args:
            * r (torch.Tensor of shape (n, n_atoms, n_atoms)): Interatomic distances, whose
                diagonal is whatever norm() returns for a zero displacement

        Returns:
            * born (torch.Tensor of shape (n, n_atoms)): Born radii
        """
        or1 = self.gb_or[None, :, None]
        sr2 = self.gb_sr[None, None, :]
        upper = r + sr2
        lower = torch.maximum(or1, torch.abs(r - sr2))
        term = 0.5 * (1 / lower - 1 / upper
                      + 0.25 * (r - sr2 ** 2 / r) * (1 / upper ** 2 - 1 / lower ** 2)
                      + 0.5 * torch.log(lower / upper) / r)
        if self.gb_model == 'obc2':
            term = term + (1 / or1 - 1 / lower) * (sr2 - r > or1)
        mask = (r + sr2 >= or1) * (1 - self.eye)
        if self.gb_cutoff is not None:
            mask = mask * (r < self.gb_cutoff)
        psi = (term * mask).sum(-1) * self.gb_or

        if self.gb_model == 'obc1':
            rescaled = torch.tanh(0.8 * psi + 2.909125 * psi ** 3)
        else:
            rescaled = torch.tanh(psi - 0.8 * psi ** 2 + 4.85 * psi ** 3)
        return 1 / (1 / self.gb_or - rescaled / self.gb_radius)

    def generalized_born(self, positions):
        """Evaluates the polar Born term and, where the system has one, the ACE surface term"""
        r = norm(positions[:, :, None] - positions[:, None, :])
        born = self.born_radii(r)

        # The self term, and the surface term that approximates the non-polar contribution
        energy = (-0.5 * self.gb_factor * self.gb_charge ** 2 / born).sum(-1)
        if self.gb_surface:
            energy = energy + (self.gb_surface * (self.gb_radius + OBC_PROBE) ** 2
                               * (self.gb_radius / born) ** 6).sum(-1)

        # The pair term, over every unique pair -- the Born term has no exclusions -- shifted when
        # a cutoff applies
        r = r[:, self.gb_rows, self.gb_cols]
        product = born[:, self.gb_rows] * born[:, self.gb_cols]
        f = torch.sqrt(r ** 2 + product * torch.exp(-r ** 2 / (4 * product)))
        if self.gb_cutoff is None:
            return energy - (self.gb_factor * self.gb_pair_charge / f).sum(-1)
        pair = self.gb_factor * self.gb_pair_charge * (1 / f - 1 / self.gb_cutoff)
        return energy - (pair * (r < self.gb_cutoff)).sum(-1)

    def potential(self, positions):
        """Evaluates the softened potential energy of one chunk, in kT

        Args:
            * positions (torch.Tensor of shape (n, n_atoms, 3)): Positions, in nanometres

        Returns:
            * energy (torch.Tensor of shape (n,)): Energies in kT, after linlogcut
        """
        energy = self.bonded(positions) + self.nonbonded(positions)
        if self.gb_model is not None:
            energy = energy + self.generalized_born(positions)
        return linlogcut(energy / self.kT, self.energy_cut, self.energy_max)

    def harmonic_potential(self, positions):
        """Evaluates the bond and angle terms of one chunk, in kT

        No linlogcut: the harmonic terms are bounded below and grow quadratically, so there is
        nothing for it to soften.
        """
        return self.harmonic(positions) / self.kT

    def harmonic_energy(self, x):
        """Evaluates the bond and angle energy of a batch, in kT"""
        positions = x.reshape(len(x), self.n_atoms, 3).to(self.dtype)
        potential = self.compiled_harmonic if self.compiled_harmonic is not None \
            else self.harmonic_potential
        return torch.cat([potential(chunk) for chunk in positions.split(self.chunk_size)])

    def harmonic_evaluate(self, x):
        """Evaluates the bond and angle energy and force of a batch, both from one pass

        Returns:
            * energy (torch.Tensor of shape (n,)): Energies, in kT
            * force (torch.Tensor of shape (n, n_atoms * 3)): Forces, in kT / nm
        """
        positions = x.reshape(len(x), self.n_atoms, 3).to(self.dtype)
        potential = self.compiled_harmonic if self.compiled_harmonic is not None \
            else self.harmonic_potential

        energies, forces = [], []
        for chunk in positions.split(self.chunk_size):
            with torch.enable_grad():
                inputs = chunk.detach().requires_grad_(True)
                energy = potential(inputs)
                (gradient,) = torch.autograd.grad(energy.sum(), inputs)
            energies.append(energy.detach())
            forces.append(-gradient)
        return torch.cat(energies), torch.cat(forces).reshape(len(x), -1)

    def energy(self, x):
        """Evaluates the energy of a batch in kT, differentiably

        Args:
            * x (torch.Tensor of shape (n, n_atoms * 3)): Positions, in nanometres

        Returns:
            * energy (torch.Tensor of shape (n,)): Energies, in kT, after linlogcut
        """
        positions = x.reshape(len(x), self.n_atoms, 3).to(self.dtype)
        potential = self.compiled if self.compiled is not None else self.potential
        return torch.cat([potential(chunk) for chunk in positions.split(self.chunk_size)])

    def evaluate(self, x):
        """Evaluates the energy and the force of every configuration in a batch

        Both come from one pass. What is kept alive afterwards depends on what is being asked:

        * x does not require grad -- nothing is kept. The graph is freed inside the loop and both
          returns are plain tensors. This is evaluation, sampling and metrics, and is the default.
        * x requires grad -- the ENERGY is differentiable, at the cost of the ordinary forward
          graph. The score is not: it comes back detached.
        * x requires grad and differentiable_score -- the score is differentiable too, which means
          the backward graph is retained on top of the forward one so it can be traversed a second
          time. Only a loss built on the score needs this.

        WARNING: the eager potential is substituted whenever the graph has to survive the call,
        whatever compile was set to, which is both of the differentiable cases above. torch.compile
        cannot do double backward at all, and its backward donates its own buffers, so asking it
        for a retained graph raises 'compiled with non-empty donated buffers'. That is not
        reproducible on a single batch shape: the first shape compiles a static kernel that has no
        donated buffers, and only once a second shape makes torch go dynamic does it fail. Setting
        torch._functorch.config.donated_buffer = False buys the compiled path back on a
        differentiable energy, but it is process-wide state that would also reach the caller's own
        model, so it is theirs to set and not ours.

        Args:
            * x (torch.Tensor of shape (n, n_atoms * 3)): Positions, in nanometres

        Returns:
            * energy (torch.Tensor of shape (n,)): Energies, in kT, after linlogcut
            * force (torch.Tensor of shape (n, n_atoms * 3)): Forces, in kT / nm
        """
        positions = x.reshape(len(x), self.n_atoms, 3).to(self.dtype)
        differentiable = positions.requires_grad
        create_graph = differentiable and self.differentiable_score
        potential = self.potential if differentiable or self.compiled is None else self.compiled

        energies, forces = [], []
        for chunk in positions.split(self.chunk_size):
            with torch.enable_grad():
                inputs = chunk if differentiable else chunk.detach().requires_grad_(True)
                energy = potential(inputs)
                (gradient,) = torch.autograd.grad(energy.sum(), inputs, create_graph=create_graph,
                                                  retain_graph=differentiable)
            energies.append(energy if differentiable else energy.detach())
            forces.append(-gradient)
        return torch.cat(energies), torch.cat(forces).reshape(len(x), -1)
