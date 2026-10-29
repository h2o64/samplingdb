# Write the published train split of every catalogued target: default_n samples at seed 0

# Libraries
import argparse, pathlib, pprint, sys, time
import numpy as np
import torch
from samplingdb import build, catalog
from samplingdb.data import cache_dir, sha256
from samplingdb.targets import spec_id
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from manifest import manifest

# Parse the arguments
parser = argparse.ArgumentParser(
    description='Write the default train split of every target (or of --targets) to '
                '<cache>/train/<name>_<spec_id>_<recipe>.npy with its manifest, and print the '
                'lines to add to REFERENCE_CHECKSUMS before running gen/publish/publish.py.')
parser.add_argument('--targets', type=str, nargs='*', default=None,
                    help='catalog names; every catalogued target when omitted')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# Run every recipe on the device, through train_samples so a split already cached is reused
folder = cache_dir() / 'train'
folder.mkdir(parents=True, exist_ok=True)
lines = []
for name in args.targets or catalog():
    target = build(name).to(args.device)
    path = folder / (target.train_key() + '.npy')
    if not path.exists():
        start = time.time()
        np.save(path, target.train_samples().cpu().numpy())
        protocol = {'train_recipe': target.train_recipe, 'n': target.default_n, 'seed': 0,
                    'device': torch.device(args.device).type, 'torch': torch.__version__}
        (folder / (target.train_key() + '_manifest.json')).write_text(manifest(
            target.name, spec_id(target.spec()), protocol, {path.name: sha256(path)}))
        print('wrote %-48s %.0f s' % (path.name, time.time() - start), flush=True)
    lines.append('    %r: %r,' % ('train/' + path.name, sha256(path)))

# What REFERENCE_CHECKSUMS needs to publish them
print('\nAdd to REFERENCE_CHECKSUMS in samplingdb/data.py, and bump REFERENCE_REVISION:')
print('\n'.join(lines))
