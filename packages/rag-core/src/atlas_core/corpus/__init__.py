"""Corpus generation (seam S1).

Synthetic multi-tenant enterprise corpus with gold labels derived from spec
literals. The manifest is the independent source of truth for evaluation; the
retrieval/generation systems under test never compute their own expected answers.
"""

from atlas_core.corpus.generate import CorpusManifest, generate_corpus
from atlas_core.corpus.spec import CorpusSpec, TenantSpec

__all__ = ["CorpusManifest", "CorpusSpec", "TenantSpec", "generate_corpus"]
