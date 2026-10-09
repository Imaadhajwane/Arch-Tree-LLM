# utils/checkpoint.py
"""
Tiny checkpoint helper: save progress every N iterations to a JSON file so a killed
run can resume from last_iter + 1 instead of from zero.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class Checkpoint:
    def __init__(self, path: str, save_every: int = 10):
        if save_every < 1:
            raise ValueError("save_every must be >= 1")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every

    def load(self) -> Dict[str, Any]:
        """Return the saved checkpoint, or a fresh one (last_iter=-1) if none exists or it is unreadable."""
        if self.path.exists():
            try:
                with open(self.path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                # An unreadable file shouldn't crash the resume; start over and say so
                logger.warning(f"[checkpoint] {self.path} unreadable ({e}) — starting from scratch.")
        return {"last_iter": -1, "state": {}}

    def save(self, last_iter: int, state: Optional[dict] = None) -> None:
        """Atomic write (temp file + rename): a kill mid-write leaves the previous checkpoint intact."""
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump({"last_iter": last_iter, "state": state or {}}, f)
        os.replace(tmp, self.path)

    def maybe_save(self, i: int, last_iter_total: int, state: Optional[dict] = None) -> None:
        """Call every loop iteration; writes only every `save_every` iterations and on the final one."""
        if i % self.save_every == 0 or i == last_iter_total - 1:
            self.save(i, state)