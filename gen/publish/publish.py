# Check the generated reference sets against the code, then publish them to the dataset repo

# Libraries
import argparse, hashlib, json, os, pathlib, pprint, sys, tempfile
import numpy as np
from huggingface_hub import CommitOperationAdd, HfApi, get_token, login
from samplingdb import CATALOG, build, catalog, data
from samplingdb.data import (REFERENCE_CHECKSUMS, REFERENCE_REPO, REFERENCE_REVISION, cache_dir,
                             sha256)
from samplingdb.targets import spec_id

# The dataset card's template, which this script fills with the table of files, and the catalog
# banner it shows, the light variant of the README's
CARD = pathlib.Path(__file__).resolve().parent / 'card.md'
BANNER = pathlib.Path(__file__).resolve().parents[2] / 'docs' / 'figures' / 'images' \
    / 'banner_light.png'

# Parse the arguments
parser = argparse.ArgumentParser(
    description='Check every reference set listed in samplingdb/data.py against the cache and the '
                'catalog, write the dataset card, and with --upload publish them to '
                + REFERENCE_REPO + ' under the tag ' + REFERENCE_REVISION + '.')
parser.add_argument('--upload', action='store_true',
                    help='upload the sets and the card, and tag the release '
                         'if it is new (logs in first if needed)')
parser.add_argument('--check_release', action='store_true',
                    help='only check the published release against the code, uploading nothing')
args = parser.parse_args()

# Save the arguments in a dictionary
config = vars(args)
config.update({'repo': REFERENCE_REPO, 'revision': REFERENCE_REVISION, 'cache': str(cache_dir())})

# Print the config
pprint.pprint(config)


def fail(lines):
    """Prints what went wrong and how to fix it, then exits"""
    print('\nNOT READY:')
    for line in lines:
        print('  - ' + line)
    sys.exit(1)


def size(path):
    """Returns a file's size, readable"""
    return '%.1f MB' % (path.stat().st_size / 1e6)


def published_targets():
    """Maps the dataset-repo path of every reference set and train split a catalogued target reads
    to the target

    Returns:
        * targets (dict): File in the dataset repo -> (catalog name, target)
    """
    targets = {}
    for name in catalog():
        kwargs = {'compile': False} if CATALOG[name]['cls'].__name__ == 'Peptide' else {}
        target = build(name, **kwargs)
        if target.reference_folder is not None:
            file = target.reference_folder + '/' + target.reference_key() + '.npy'
            targets[file] = (name, target)
        targets['train/' + target.train_key() + '.npy'] = (name, target)
    return targets


def check_file(file, name, target):
    """Checks one cached reference set against the registry, its manifest and its target

    Args:
        * file (str): Path in the dataset repo, e.g. 'pt/dw4_d9994289.npy'
        * name (str): Catalog name of the target
        * target (Target): The target

    Returns:
        * row (dict): What the dataset card's table shows about the set
        * problems (list of str): Everything that stops it from being published
    """
    path = cache_dir() / file
    if not path.exists():
        return None, [file + ' is not in the cache at ' + str(path) + '. Is SAMPLINGDB_CACHE '
                      'pointing at the cache that holds the generated sets?']

    # The bytes, against the registry and against the manifest written next to them
    problems = []
    digest = sha256(path)
    if digest != REFERENCE_CHECKSUMS[file]:
        problems.append(file + ': sha256 ' + digest + ' differs from REFERENCE_CHECKSUMS')
    manifest_path = path.with_name(path.stem + '_manifest.json')
    if not manifest_path.exists():
        return None, problems + [file + ': no manifest at ' + str(manifest_path)]
    manifest = json.loads(manifest_path.read_text())
    if manifest['checksums'].get(path.name) != digest:
        problems.append(file + ': the manifest records another sha256')
    if manifest.get('spec_id') != spec_id(target.spec()):
        problems.append(file + ': the manifest records spec_id ' + str(manifest.get('spec_id')))
    if manifest['target'] != target.name:
        problems.append(file + ': the manifest names the target ' + manifest['target'])

    # The array itself: the target's dimension, finite everywhere
    array = np.load(path, mmap_mode='r')
    if array.ndim != 2 or array.shape[1] != target.dim:
        problems.append(file + ': shape %s, but %s has dim %d' % (array.shape, name, target.dim))
    if not np.isfinite(array).all():
        problems.append(file + ': holds non-finite values')

    # What the card says about how independent the rows are
    protocol = manifest['protocol']
    if 'train_recipe' in protocol:
        quality = 'train split, recipe ' + protocol['train_recipe']
    elif 'efficiency' in protocol and 'min_ess_bulk' in protocol:
        quality = 'bulk ESS %.0f%% of the draws' % (100 * protocol['min_ess_bulk'] / len(array))
    elif 'efficiency' in protocol:
        quality = '%.0f%% effective' % (100 * protocol['efficiency'])
    else:
        quality = 'exact independent draws'
    row = {'file': file, 'name': name, 'rows': len(array), 'dim': array.shape[1],
           'dtype': str(array.dtype), 'size': size(path), 'quality': quality,
           'manifest': manifest_path}
    return row, problems


def card(rows):
    """Fills the dataset card's template with the release and its files"""
    table = ['| File | `build` name | Samples | dim | dtype | Size | Independence |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        table.append('| `%s` | `%s` | %d | %d | %s | %s | %s |' % (
            row['file'], row['name'], row['rows'], row['dim'], row['dtype'], row['size'],
            row['quality']))
    total = sum((cache_dir() / row['file']).stat().st_size for row in rows) / 1e9
    text = CARD.read_text()
    for key, value in [('{{repo}}', REFERENCE_REPO), ('{{revision}}', REFERENCE_REVISION),
                       ('{{files}}', '\n'.join(table)), ('{{total}}', '%.1f GB' % total),
                       ('{{example}}', rows[0]['file'])]:
        text = text.replace(key, value)
    return text


def git_blob_id(path):
    """Returns the git blob id of a file, which the hub reports for files kept outside LFS"""
    content = path.read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(content) + content).hexdigest()


def check_release(api):
    """Checks that the tagged release serves every registered set, then downloads the smallest one
    through the library's own code path into an empty cache

    Args:
        * api (huggingface_hub.HfApi): An API client, logged in or not

    Returns:
        * problems (list of str): Everything that differs from REFERENCE_CHECKSUMS
    """
    problems = []
    infos = {info.path: info for info in api.get_paths_info(
        REFERENCE_REPO, list(REFERENCE_CHECKSUMS), revision=REFERENCE_REVISION,
        repo_type='dataset')}
    for file, checksum in REFERENCE_CHECKSUMS.items():
        info = infos.get(file)
        if info is None:
            problems.append(file + ' is missing at ' + REFERENCE_REVISION)
        elif info.lfs is not None and info.lfs.sha256 != checksum:
            problems.append(file + ' has sha256 ' + info.lfs.sha256 + ' at ' + REFERENCE_REVISION)
    if problems:
        return problems

    # The library's own download, exactly as a user's first test_samples() call runs it
    smallest = min(REFERENCE_CHECKSUMS, key=lambda file: infos[file].size)
    folder, key = smallest[:-len('.npy')].split('/')
    with tempfile.TemporaryDirectory() as directory:
        cache = os.environ.get('SAMPLINGDB_CACHE')
        os.environ['SAMPLINGDB_CACHE'] = directory
        path = data.fetch_reference(folder, key, '')
        print('  downloaded %s through samplingdb, sha256 checked, %s' % (smallest, size(path)))
        if cache is None:
            del os.environ['SAMPLINGDB_CACHE']
        else:
            os.environ['SAMPLINGDB_CACHE'] = cache
    return problems


# Only check a release that is already published
if args.check_release:
    print('\nChecking %s at %s against samplingdb/data.py' % (REFERENCE_REPO, REFERENCE_REVISION))
    problems = check_release(HfApi())
    if problems:
        fail(problems)
    print('\nThe release serves every registered set with the registered sha256.')
    sys.exit(0)

# 1. Match every registered file to the catalog target that reads it. A file no target reads is
# the trace of a density change: its spec_id moved, so the set has to be regenerated and registered
# again
print('\n1/5  Matching the %d registered sets to the catalog' % len(REFERENCE_CHECKSUMS))
targets = published_targets()
orphans = [file for file in REFERENCE_CHECKSUMS if file not in targets]
if orphans:
    fail([file + ' is in REFERENCE_CHECKSUMS, but no catalog target reads it any more: its '
          'density changed. Regenerate the set and register the new file instead.'
          for file in orphans])
for file, (name, target) in targets.items():
    if file in REFERENCE_CHECKSUMS:
        continue
    path = cache_dir() / file
    if path.exists():
        print('  not registered: %s (%s) is in the cache but not in REFERENCE_CHECKSUMS. Once\n'
              '    it has passed its audit, publish it in the next release: add\n      %r: %r,\n'
              '    to REFERENCE_CHECKSUMS in samplingdb/data.py and bump REFERENCE_REVISION.'
              % (file, name, file, sha256(path)))
    else:
        print('  not generated: %s (%s) is not in the cache yet.' % (file, name))

# 2. Check every registered set in the cache
print('\n2/5  Checking the cached sets in %s' % cache_dir())
rows, problems = [], []
for file in REFERENCE_CHECKSUMS:
    row, issues = check_file(file, *targets[file])
    problems.extend(issues)
    if row is not None:
        rows.append(row)
        print('  %-44s %8d x %-5d %-8s %10s  %s%s' % (
            file, row['rows'], row['dim'], row['dtype'], row['size'], row['quality'],
            '' if not issues else '  <- see below'))
if problems:
    fail(problems)

# 3. Write the dataset card next to the sets, so it can be read before anything is uploaded
print('\n3/5  Writing the dataset card')
folder = cache_dir() / 'publish' / REFERENCE_REVISION
folder.mkdir(parents=True, exist_ok=True)
(folder / 'README.md').write_text(card(rows))
print('  ' + str(folder / 'README.md'))

if not args.upload:
    print('\nEverything is ready to publish %d sets (%.1f GB) as %s at %s. Nothing was uploaded.'
          % (len(rows), sum((cache_dir() / row['file']).stat().st_size for row in rows) / 1e9,
             REFERENCE_REPO, REFERENCE_REVISION))
    print('To publish, rerun with --upload. It asks for a Hugging Face token with write access the '
          'first time (https://huggingface.co/settings/tokens).')
    sys.exit(0)

# 4. Upload: log in, create the repo, add every set with its manifest, the card, then the tag
print('\n4/5  Uploading to https://huggingface.co/datasets/' + REFERENCE_REPO)
if get_token() is None:
    print('  Not logged in to Hugging Face: paste a token with write access, from '
          'https://huggingface.co/settings/tokens')
    login()
api = HfApi()
me = api.whoami()
namespaces = [me['name']] + [org['name'] for org in me.get('orgs', [])]
if REFERENCE_REPO.split('/')[0] not in namespaces:
    fail(['You are logged in as ' + me['name'] + ', but samplingdb reads its sets from '
          + REFERENCE_REPO + '. Set REFERENCE_REPO in samplingdb/data.py to "' + me['name']
          + '/samplingdb" (and the same name in README.md and docs/), then rerun. Users read the '
          'new name once that change is pushed.'])

# A tag is a promise to the code that pins it, so an existing one is checked, never moved. What
# the main branch still lacks -- a newer card -- is added on top of it
tagged = False
if api.repo_exists(REFERENCE_REPO, repo_type='dataset'):
    tags = [tag.name for tag in api.list_repo_refs(REFERENCE_REPO, repo_type='dataset').tags]
    tagged = REFERENCE_REVISION in tags
    if tagged:
        problems = check_release(api)
        if problems:
            fail(['The tag ' + REFERENCE_REVISION + ' already exists and serves other files. '
                  'Bump REFERENCE_REVISION in samplingdb/data.py for a new release.'] + problems)
        print('  %s is already published and matches the code; adding what main lacks'
              % REFERENCE_REVISION)
api.create_repo(REFERENCE_REPO, repo_type='dataset', private=False, exist_ok=True)

# One commit per set, so an interrupted upload resumes where it stopped. A tag pins the bytes it
# was cut at, so a set regenerated for a new release replaces the file on main, with its manifest,
# while the older tags keep serving theirs; a cache filled at an older tag is
# caught by data.verify_once. An existing tag was checked above and is never moved
wanted = [row['file'] for row in rows] + ['README.md', 'banner.png']
present = {info.path: info for info in api.get_paths_info(REFERENCE_REPO, wanted,
                                                          repo_type='dataset')}
for row in rows:
    file = row['file']
    published = file in present and present[file].lfs is not None \
        and present[file].lfs.sha256 == REFERENCE_CHECKSUMS[file]
    if published:
        print('  %-44s already uploaded' % file)
        continue
    print('  uploading ' + file + ' ...', flush=True)
    api.create_commit(REFERENCE_REPO, repo_type='dataset', commit_message='Add ' + file,
                      operations=[CommitOperationAdd(file, str(cache_dir() / file)),
                                  CommitOperationAdd(file[:-len('.npy')] + '_manifest.json',
                                                     str(row['manifest']))])
card_id = git_blob_id(folder / 'README.md')
banner = present.get('banner.png')
if 'README.md' not in present or present['README.md'].blob_id != card_id or banner is None \
        or banner.lfs is None or banner.lfs.sha256 != sha256(BANNER):
    api.create_commit(REFERENCE_REPO, repo_type='dataset',
                      commit_message='Write the dataset card of ' + REFERENCE_REVISION,
                      operations=[CommitOperationAdd('README.md', str(folder / 'README.md')),
                                  CommitOperationAdd('banner.png', str(BANNER))])
    print('  README.md, banner.png                        the dataset card and its banner')
if not tagged:
    api.create_tag(REFERENCE_REPO, tag=REFERENCE_REVISION, repo_type='dataset',
                   tag_message='samplingdb reference sets ' + REFERENCE_REVISION)
    print('  tagged ' + REFERENCE_REVISION)

# 5. Check the release the way a user meets it
print('\n5/5  Checking the release')
problems = check_release(api)
if problems:
    fail(problems)
print('\nPublished: https://huggingface.co/datasets/%s/tree/%s' % (REFERENCE_REPO,
                                                                  REFERENCE_REVISION))
