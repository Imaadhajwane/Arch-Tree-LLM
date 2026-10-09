# archtree/weight_bank.py
"""
Weight Bank: stores a small simulated state per trie path so that
ancestor inheritance, the on-disk format, and a future GPU-training hook
all exist and are testable. NB201/NB101 are table-lookup benchmarks, so
no real training happens here.
"""
from typing import Dict, List, Optional

import torch

from archtree.trie import ArchTrie
import archtree.storage as storage

FLUSH_EVERY = 25  # write to disk once per this many stores, to avoid per-iteration disk I/O


class WeightBank:
    def __init__(self, phase: str, bench: str, dataset: str, seed: int, namespace: str = ""):
        self.phase, self.bench, self.dataset, self.seed = phase, bench, dataset, seed
        self.namespace = namespace
        # Resume from disk if a bank for this run already exists
        self._store: Dict[tuple, dict] = (
            storage.load_weight_bank(phase, bench, dataset, seed, namespace) or {}
        )
        self._dirty_count = 0  # number of store() calls since creation

    def load_ancestor(self, trie: ArchTrie, path: List[str]) -> Optional[dict]:
        """Longest-prefix match: return the state of the deepest stored prefix of `path`, or None."""
        # `trie` is unused; kept so existing call sites don't break.
        # Start from the full path (exact match) and shrink one token at a time.
        for depth in range(len(path), 0, -1):
            key = tuple(path[:depth])
            if key in self._store:
                return self._store[key]
        return None

    def store(self, path: List[str], new_layer_state: dict) -> None:
        """Save the state for `path` and flush to disk periodically."""
        self._store[tuple(path)] = new_layer_state
        self._dirty_count += 1
        if self._dirty_count % FLUSH_EVERY == 0:
            self.flush()

    def flush(self) -> None:
        """Write the whole bank to disk. Call once more at the end of a run to save the last partial batch."""
        storage.save_weight_bank(self.phase, self.bench, self.dataset, self.seed,
                                 self._store, self.namespace)

    def simulate_train_new_layers(self, path: List[str], ancestor_state: Optional[dict]) -> dict:
        """
        Stand-in for real forward/backward training (replace only if a real supernet is attached).
        Returns a 1-element tensor derived from the ancestor's tensor plus small noise,
        tagged with the path depth so the inheritance chain can be verified.
        """
        base = ancestor_state["tensor"].clone() if ancestor_state else torch.zeros(1)
        return {"tensor": base + torch.randn(1) * 0.01, "depth": len(path)}