# Regenerate the pinned mixture mean matrices shipped in samplingdb/targets/assets

# Libraries
import hashlib, pathlib
import numpy as np
import jax

# Where the assets live
ASSETS = pathlib.Path(__file__).resolve().parents[2] / 'samplingdb' / 'targets' / 'assets'

# The two mixture families, with the draw that defines them
FAMILIES = [
    ('gmm40', 40, -40.0, 40.0, (2, 50, 200)),
    ('mos', 10, -10.0, 10.0, (2, 50, 200)),
]


def main():
    """Writes one .npy per (family, dim) and prints the sha256 of each float64 buffer"""
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, n_modes, low, high, dims in FAMILIES:
        for dim in dims:
            key = jax.random.PRNGKey(0)
            means = jax.random.uniform(shape=(n_modes, dim), key=key, minval=low, maxval=high)
            means = np.asarray(means, dtype=np.float64)
            path = ASSETS / (name + '_means_dim' + str(dim) + '.npy')
            np.save(path, means)
            digest = hashlib.sha256(means.tobytes()).hexdigest()
            print(path.name, means.shape, digest)


if __name__ == '__main__':
    main()
