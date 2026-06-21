"""Step embedding (spec 4.2). Pluggable so the math stays testable and the system stays
runnable offline.

- SentenceTransformerEmbedder: the real semantic model (all-MiniLM-L6-v2), lazy-loaded on first
  use so importing the app never downloads a model. If the model fails to load (offline, no
  cache), it falls back to the hashing embedder once, logged, rather than crashing the watchdog.
- HashingEmbedder: deterministic hashed bag-of-words. No dependencies, no network. Identical
  text embeds identically (cosine 1.0), shared tokens give high similarity — enough for tight
  loops, exact for tests. Not semantic, so it is a graceful degradation, not the default.
"""

from __future__ import annotations

import hashlib
import re
import threading
from typing import Any, Protocol

import numpy as np

from .logging_config import get_logger, log_event
from .settings import Settings

logger = get_logger("embedder")

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


class Embedder(Protocol):
    def encode(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic hashed bag-of-words, L2-normalized."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec


class SentenceTransformerEmbedder:
    """Lazy wrapper around sentence-transformers with a hashing fallback on load failure."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._fallback: HashingEmbedder | None = None
        # encode() runs in worker threads (asyncio.to_thread), so guard the one-time load.
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._fallback is not None:
            return
        with self._lock:
            if self._model is not None or self._fallback is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
                log_event(logger, 20, "embedder.loaded", model=self._model_name)
            except Exception as exc:  # offline / no cache -> degrade, do not crash detection
                self._fallback = HashingEmbedder()
                log_event(logger, 30, "embedder.fallback", model=self._model_name, error=str(exc))

    def encode(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        if self._fallback is not None:
            return self._fallback.encode(text)
        return np.asarray(self._model.encode(text), dtype=np.float32)


def make_embedder(settings: Settings) -> Embedder:
    if settings.embed_hashing:
        log_event(logger, 20, "embedder.hashing_forced")
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.embed_model)
