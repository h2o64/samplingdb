# Keep every test's cache writes inside a temporary directory, and say what the cache cost us

# Libraries
import collections
import os
import pathlib
import sys
import pytest

# The samplers, the integrator and the chain diagnostics live in gen/, outside the package, so the
# tests that check them import from there
GEN = pathlib.Path(__file__).resolve().parents[1] / 'gen'
sys.path[:0] = [str(GEN), str(GEN / 'md')]


@pytest.fixture(scope='session', autouse=True)
def cache_dir(tmp_path_factory):
    """Points SAMPLINGDB_CACHE at a temporary directory, unless one is already set

    An existing SAMPLINGDB_CACHE is left alone so that a developer who has already downloaded the
    released glass configurations does not re-download them, and so the tests guarded on their
    presence actually run.
    """
    if 'SAMPLINGDB_CACHE' not in os.environ:
        os.environ['SAMPLINGDB_CACHE'] = str(tmp_path_factory.mktemp('samplingdb-cache'))
    yield os.environ['SAMPLINGDB_CACHE']


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Reports how many tests skipped for want of downloaded data, and which tiers went unchecked

    A green run on a cold cache is not the same thing as a green run on a warm one: most of the
    molecular and released-data tests skip themselves silently from inside a helper, so without
    this the summary reads 'passed' while whole tiers never executed. SAMPLINGDB_CACHE pointed at
    a warm cache is what turns them on.
    """
    skipped = terminalreporter.stats.get('skipped', [])
    if not skipped:
        return

    reasons = collections.Counter()
    for report in skipped:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        reasons[reason.replace('Skipped: ', '')] += 1

    terminalreporter.write_sep('-', 'skipped for want of data or an optional dependency')
    for reason, count in reasons.most_common():
        terminalreporter.write_line('  %4d  %s' % (count, reason[:96]))
    terminalreporter.write_line(
        '  set SAMPLINGDB_CACHE to a warm cache to run these; %d of %d tests did not execute'
        % (len(skipped), len(skipped) + len(terminalreporter.stats.get('passed', []))))
