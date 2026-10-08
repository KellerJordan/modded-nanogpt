"""CPU-only verification of the packed canonical-mask constructor.

Run from the repository root with numpy and tiktoken installed:
    python evals/check_canonical_mask.py

Compares with the pinned dense implementation from git, tests the actual packed
scatter against an independent scalar oracle, and compares every full-vocab byte.
No torch import, training data or GPU is required. Uses roughly 3 GB of RAM.
"""
import argparse
import ast
import hashlib
import json
import subprocess
import types
from pathlib import Path

import numpy as np
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'track_1_short/canonical_mask.py'
BASE_REF = '4ea6b937337a4889b8cfe3f38a93d120048d8f71'


def load_cpu(source):
    # Preserve CPU definitions verbatim; omit only torch imports and the wrapper class.
    tree = ast.parse(source)
    kept = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            continue
        if isinstance(node, ast.Import):
            node.names = [n for n in node.names if not n.name.startswith('torch')]
            if not node.names:
                continue
        if isinstance(node, ast.ImportFrom) and node.module == 'torch':
            continue
        kept.append(node)
    tree.body = kept
    module = types.ModuleType('canonical_cpu')
    exec(compile(tree, '<canonical CPU definitions>', 'exec'), module.__dict__)
    return module


def scatter_function(text):
    # Exercise the actual candidate block, not a reimplementation of it.
    tree = ast.parse(text)
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_canonical_mask')
    loop = next(n for n in build.body if isinstance(n, ast.For) and ast.unparse(n.target) == '(rank, (a, b))')
    block = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'len(xs)')
    wrapper = ast.parse('def scatter(mask, ps, xs, end_cls, start_cls):\n    stride = mask.shape[1]\n    flat_mask = mask.reshape(-1)\n')
    wrapper.body[0].body.extend(block.body)
    ast.fix_missing_locations(wrapper)
    seam = np.ones((5, 5), dtype=bool)
    seam[:, 4] = False
    seam[0, 0] = False
    seam[3, 2] = False
    space = {'np': np, '_SEAM_OK': seam}
    exec(compile(wrapper, '<candidate scatter>', 'exec'), space)
    return space['scatter']


def scalar_scatter(vocab, ps, xs, ends, starts):
    result = np.zeros((vocab, (vocab + 7) // 8), dtype=np.uint8)
    for p in ps:
        for x in xs:
            # Independent scalar expression of the three forbidden seam cases.
            if starts[x] == 4 or (ends[p] == 0 and starts[x] == 0) or (ends[p] == 3 and starts[x] == 2):
                continue
            result[p, x // 8] |= np.uint8(1 << (x % 8))
    return result


def vocabulary(seed):
    rng = np.random.default_rng(seed)
    ranks = {bytes([i]): i for i in range(256)}
    # Include whitespace, contractions, and UTF-8 pieces in addition to arbitrary bytes.
    for value in [b" 's", b"'re", b"don't", b' \n', b'\t\r', 'é中🙂'.encode()]:
        for end in range(2, len(value) + 1):
            b = value[:end]
            if b not in ranks:
                ranks[b] = len(ranks)
    tokens = list(ranks)
    while len(ranks) < 420:
        a, b = [tokens[int(rng.integers(len(tokens)))] for _ in range(2)]
        if len(a + b) <= 24 and a + b not in ranks:
            ranks[a + b] = len(ranks)
            tokens.append(a + b)
    order = rng.permutation(len(tokens))
    return {tokens[i]: ranks[tokens[i]] for i in order}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-ref', default=BASE_REF)
    args = parser.parse_args()
    base_text = subprocess.check_output(
        ['git', 'show', args.base_ref + ':track_1_short/canonical_mask.py'], cwd=ROOT, text=True)
    text = CANDIDATE.read_text()
    scatter = scatter_function(text)
    rng = np.random.default_rng(1008)
    cases = []
    for i in range(80):
        v = [17, 129, 256, 511][i % 4]
        n = min(v, 8 if i % 3 == 0 else 100)
        ps, xs = (rng.choice(v, size=n, replace=False) for _ in range(2))
        ends, starts = rng.integers(0, 5, size=(2, v))
        got = np.zeros((v, (v + 7) // 8), dtype=np.uint8)
        scatter(got, ps, xs, ends, starts)
        expected = scalar_scatter(v, ps, xs, ends, starts)
        np.testing.assert_array_equal(got, expected)
        cases.append({'vocab': v, 'pairs': n * n})
    # A tempting but wrong buffered OR drops colliding bits. The oracle must reject it.
    mutant = scatter_function(text.replace(
        'np.bitwise_or.at(flat_mask, positions[ok], bits[ok])',
        'flat_mask[positions[ok]] |= bits[ok]'))
    ends = starts = np.full(16, 1)
    ps, xs = np.array([0]), np.arange(8)
    corrupt = np.zeros((16, 2), dtype=np.uint8)
    mutant(corrupt, ps, xs, ends, starts)
    expected = scalar_scatter(16, ps, xs, ends, starts)
    assert not np.array_equal(corrupt, expected), 'Corruption control did not fail'

    candidate = load_cpu(text)
    originals = [load_cpu(base_text)]
    fixtures = []
    for seed in range(8):
        ranks = vocabulary(seed)
        size = len(ranks) + seed  # include padded IDs and non-byte-aligned shapes
        actual = candidate.build_canonical_mask(size, ranks)
        for reference in originals:
            np.testing.assert_array_equal(actual, reference.build_canonical_mask(size, ranks))
        fixtures.append({'seed': seed, 'vocab_size': size, 'sha256': hashlib.sha256(actual).hexdigest()})
    for size in (0, 1, 9):
        actual = candidate.build_canonical_mask(size, {})
        for reference in originals:
            np.testing.assert_array_equal(actual, reference.build_canonical_mask(size, {}))
    ranks = tiktoken.get_encoding('gpt2')._mergeable_ranks
    expected = originals[0].build_canonical_mask(50304, ranks)
    actual = candidate.build_canonical_mask(50304, ranks)
    np.testing.assert_array_equal(actual, expected)
    result = {'passed': True, 'scalar_scatter_cases': len(cases), 'generated_vocabularies': fixtures,
              'empty_vocabulary_sizes': [0, 1, 9], 'corruption_control_rejected': True,
              'source_sha256': hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
              'base_ref': args.base_ref,
              'base_source_sha256': hashlib.sha256(base_text.encode()).hexdigest(),
              'full_gpt2_shape': list(actual.shape),
              'full_gpt2_exact_match': True,
              'full_gpt2_sha256': hashlib.sha256(memoryview(actual).cast('B')).hexdigest()}
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
