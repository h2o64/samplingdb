# The manifest written next to every generated reference set and shard

# Libraries
import json


def manifest(name, spec, protocol, checksums):
    """Builds the manifest.json payload that travels with every generated reference set

    A reference ensemble is only as good as the protocol that made it, so every file the gen/
    scripts write gets a sibling manifest.json naming the target, the version of its density, the
    sampler settings and the sha256 of the array. That is what lets a number be traced back to the
    run that produced it, and what makes a regenerated set comparable to the one it replaces.

    Args:
        * name (str): The target's name
        * spec (str): The spec_id of the density the set was drawn from
        * protocol (dict): Generation protocol metadata (sampler, steps, seed, versions)
        * checksums (dict): File name -> sha256

    Returns:
        * payload (str): The manifest, as pretty-printed JSON
    """
    return json.dumps(
        {'target': name, 'spec_id': spec, 'protocol': protocol, 'checksums': checksums},
        indent=2,
        sort_keys=True,
    )
