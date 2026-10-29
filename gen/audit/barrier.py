# Profile where on the annealing path a target's communication barrier sits

# Libraries
import argparse, pathlib, pprint, sys, time
import torch
from samplingdb import build
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nrpt import hmc_explorer, linear_schedule, non_reversible_parallel_tempering

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True)
parser.add_argument('--n_rungs', type=int, default=40)
parser.add_argument('--n_chains', type=int, default=64)
parser.add_argument('--n_rounds', type=int, default=8)
parser.add_argument('--beta_min', type=float, default=0.02)
parser.add_argument('--step_size', type=float, default=0.005)
parser.add_argument('--n_local', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# Build the target on a fine ladder, which resolves the barrier: a schedule with many more levels
# than 2 Lambda reads the density at many places
device = torch.device(args.device)
target = build(args.target).to(device)
beta = linear_schedule(args.n_rungs, args.beta_min, dtype=target.dtype, device=device)

torch.manual_seed(args.seed)
gen = torch.Generator(device='cpu').manual_seed(args.seed)
# The lattice jitter is gen/parallel_tempering/generate.py's
if hasattr(target, 'lattice_configuration'):
    start = target.lattice_configuration(args.n_chains, jitter=0.02, generator=gen)
    start = start.reshape(args.n_chains, -1)
else:
    start = target.initial_configuration().reshape(1, -1).expand(args.n_chains, target.dim).cpu()
    start = start + 0.05 * torch.randn(args.n_chains, target.dim, generator=gen,
                                       dtype=target.dtype)
x0 = start.to(device)[None].expand(args.n_rungs, args.n_chains, target.dim).contiguous()

# Run the sampler only to read its diagnostics: the schedule adaptation equalises the rejections,
# so the level spacing it settles on is itself the answer, and rejection / d beta is the density
start_time = time.time()
_, diagnostics = non_reversible_parallel_tempering(
    target.log_prob, target.score, x0, beta,
    explorer=hmc_explorer(n_steps=args.n_local, n_leapfrog=10), n_rounds=args.n_rounds,
    reference=None, step_size=args.step_size, generator=None, thin=1, return_diagnostics=True)
elapsed = time.time() - start_time

# The caloric curve and the barrier density, level by level. lambda(beta) = sqrt(Var_beta(V) / pi)
# for a Gaussian V, so the two columns are a check on each other, and Lambda(>= beta) is the share
# of the total a hot end at beta pays for
beta, mean, deviation = diagnostics['beta'], -diagnostics['value_mean'], diagnostics['value_std']
rejection, density = diagnostics['rejection'], diagnostics['barrier_density']
remaining = torch.cat([rejection.flip(0).cumsum(0).flip(0),
                       torch.zeros(1, dtype=beta.dtype, device=device)])

print('\n%s   Lambda %.3f over beta %.4f .. 1   in %.0f s'
      % (target.name, diagnostics['barrier'], args.beta_min, elapsed))
print('%8s %11s %10s %9s %9s %11s' % ('beta', 'E', 'sd(E)', 'reject', 'lambda', 'Lambda(>=)'))
for k in range(len(beta)):
    row = '%8.4f %11.2f %10.3f' % (beta[k], mean[k], deviation[k])
    if k < len(beta) - 1:
        row += ' %9.3f %9.2f' % (rejection[k], density[k])
    else:
        row += ' %9s %9s' % ('', '')
    print(row + ' %11.3f' % remaining[k])

# What a given hot end would cost, in levels and in the round trips a collection round can hold
print('\n%10s %10s %8s %10s %14s' % ('beta_min', 'Lambda', 'rungs', 'scans/trip', 'rounds for 10'))
for fraction in [0.0, 0.25, 0.5, 0.75, 0.9]:
    k = min(int(fraction * (len(beta) - 1)), len(beta) - 2)
    barrier = float(remaining[k])
    n_rungs = max(2, int(round(2 * barrier)))
    scans = n_rungs * (2 + 2 * barrier)
    print('%10.4f %10.3f %8d %10.0f %14d'
          % (beta[k], barrier, n_rungs, scans, max(1, int(torch.tensor(10 * scans).log2().ceil()))))

print('\nexplorer acceptance %.3f .. %.3f   step size %.5f .. %.5f'
      % (diagnostics['explorer_acceptance'].min(), diagnostics['explorer_acceptance'].max(),
         diagnostics['step_size'].min(), diagnostics['step_size'].max()))
