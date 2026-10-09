# archtree/encoder.py
"""
Bench index <-> token conversion.

NB201: 6 tokens, 5^6 = 15625 archs. Pure base-5 math, no bench needed.
NB101: 26 tokens (5 interior-vertex ops, then 21 adjacency bits; see benchmarks/nb101.py).
       No math bijection: index = position of the graph hash in the sorted hash list of
       data/nb101_cache.npz, so the bench is required. Invalid cells map to None.
Trie depth = len(tokens): 6 for NB201, 26 for NB101.
"""
from typing import List

OPS_NB201 = ["none", "skip_connect", "conv_1x1", "conv_3x3", "avg_pool_3x3"]

# Same list and order as benchmarks/nb101.py::OPS (the official hash labels 0, 1, 2).
OPS_NB101 = ["conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"]


def encode_nb201(idx: int, bench) -> List[str]:
    """Index -> tokens, delegated to the bench."""
    return bench.index_to_tokens(idx)


def index_from_tokens_nb201(tokens: List[str]) -> int:
    """Tokens -> index via base-5 Horner decoding."""
    idx = 0
    for op in tokens:
        idx = idx * 5 + OPS_NB201.index(op)
    return idx


def encode_nb101(idx: int, bench) -> List[str]:
    """Index -> tokens, delegated to the bench (no math bijection exists)."""
    return bench.index_to_tokens(idx)


def index_from_tokens_nb101(tokens: List[str], bench) -> int:
    """Tokens -> index. Needs the bench: index is the hash_iterator() position."""
    return bench.tokens_to_index(tokens)