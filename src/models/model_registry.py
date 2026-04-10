"""
Model Registry — single source of truth for all embedding model metadata.

Every subsystem (adapter loading, LanceDB schema, scoring, memory management)
consults this registry instead of doing its own string-matching and hardcoding.
"""

from dataclasses import dataclass
from typing import Optional, Dict
from src.models.embedding_adapters.base_adapter import EmbeddingType
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Declarative metadata for an embedding model."""
    model_id: str               # Full HuggingFace ID
    family: str                 # Short family key (e.g. "colqwen25", "colnomic3b")
    embedding_type: EmbeddingType
    dimension: int              # Per-token vector dimension
    model_class_name: str       # colpali_engine class name (e.g. "ColQwen2_5")
    processor_class_name: str   # colpali_engine processor name
    lancedb_table: str          # Unique LanceDB table name for this model
    vram_gb: float              # Estimated VRAM in GB (bfloat16)
    min_transformers: str       # Minimum transformers version required
    min_colpali_engine: str     # Minimum colpali_engine version required
    supports_flash_attn: bool = True
    description: str = ""


# ──────────────────────────────────────────────────────────────
# Known models. Entries that require transformers>=5.0.0 are
# marked — they won't load until Phase 5 dependency upgrade.
# ──────────────────────────────────────────────────────────────

_MODELS: Dict[str, ModelSpec] = {
    # === Current (works on transformers 4.x) ===
    "tsystems/colqwen2.5-3b-multilingual-v1.0": ModelSpec(
        model_id="tsystems/colqwen2.5-3b-multilingual-v1.0",
        family="colqwen25",
        embedding_type=EmbeddingType.MULTI_VECTOR,
        dimension=128,
        model_class_name="ColQwen2_5",
        processor_class_name="ColQwen2_5_Processor",
        lancedb_table="colqwen25",
        vram_gb=7.0,
        min_transformers="4.56",
        min_colpali_engine="0.3.9",
        description="ColQwen2.5 3B multilingual (LoRA — needs transformers<5 or use merged variant)",
    ),
    "tsystems/colqwen2.5-3b-multilingual-v1.0-merged": ModelSpec(
        model_id="tsystems/colqwen2.5-3b-multilingual-v1.0-merged",
        family="colqwen25",
        embedding_type=EmbeddingType.MULTI_VECTOR,
        dimension=128,
        model_class_name="ColQwen2_5",
        processor_class_name="ColQwen2_5_Processor",
        lancedb_table="colqwen25",
        vram_gb=7.0,
        min_transformers="5.0.0",
        min_colpali_engine="0.3.15",
        description="ColQwen2.5 3B multilingual merged (compatible with transformers 5.x)",
    ),
    "vidore/colqwen2.5-v0.2": ModelSpec(
        model_id="vidore/colqwen2.5-v0.2",
        family="colqwen25",
        embedding_type=EmbeddingType.MULTI_VECTOR,
        dimension=128,
        model_class_name="ColQwen2_5",
        processor_class_name="ColQwen2_5_Processor",
        lancedb_table="colqwen25_vidore",
        vram_gb=7.0,
        min_transformers="4.56",
        min_colpali_engine="0.3.9",
        description="ColQwen2.5 v0.2 (vidore official)",
    ),
    "nomic-ai/colnomic-embed-multimodal-3b": ModelSpec(
        model_id="nomic-ai/colnomic-embed-multimodal-3b",
        family="colnomic3b",
        embedding_type=EmbeddingType.MULTI_VECTOR,
        dimension=128,
        model_class_name="ColQwen2_5",
        processor_class_name="ColQwen2_5_Processor",
        lancedb_table="colnomic_3b",
        vram_gb=7.0,
        min_transformers="4.56",
        min_colpali_engine="0.3.9",
        description="ColNomic 3B — ViDoRe v3: 55.78, same ColQwen2_5 class",
    ),
    "nomic-ai/colnomic-embed-multimodal-7b": ModelSpec(
        model_id="nomic-ai/colnomic-embed-multimodal-7b",
        family="colnomic7b",
        embedding_type=EmbeddingType.MULTI_VECTOR,
        dimension=128,
        model_class_name="ColQwen2_5",
        processor_class_name="ColQwen2_5_Processor",
        lancedb_table="colnomic_7b",
        vram_gb=15.0,
        min_transformers="4.56",
        min_colpali_engine="0.3.9",
        description="ColNomic 7B — ViDoRe v3: 57.33, needs 16GB+ VRAM",
    ),

    # === Phase 6 (requires transformers>=5.0.0, colpali_engine>=0.3.14) ===
    "athrael-soju/colqwen3.5-4.5B-v3": ModelSpec(
        model_id="athrael-soju/colqwen3.5-4.5B-v3",
        family="colqwen35",
        embedding_type=EmbeddingType.MULTI_VECTOR,
        dimension=320,
        model_class_name="ColQwen3_5",
        processor_class_name="ColQwen3_5Processor",
        lancedb_table="colqwen35_4b",
        vram_gb=10.0,
        min_transformers="5.3.0",
        min_colpali_engine="0.3.15",
        description="ColQwen3.5 4.5B — ViDoRe v3: 61.46, best 128-dim model",
    ),
    "VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1": ModelSpec(
        model_id="VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1",
        family="colqwen3",
        embedding_type=EmbeddingType.MULTI_VECTOR,
        dimension=128,
        model_class_name="ColQwen3",
        processor_class_name="ColQwen3Processor",
        lancedb_table="colqwen3_4b",
        vram_gb=10.0,
        min_transformers="5.0.0",
        min_colpali_engine="0.3.14",
        description="SauerkrautLM ColQwen3 4B — ViDoRe v3 public: 56.03",
    ),
    "vidore/colSmol-500M": ModelSpec(
        model_id="vidore/colSmol-500M",
        family="colsmol",
        embedding_type=EmbeddingType.MULTI_VECTOR,
        dimension=128,
        model_class_name="ColIdefics3",
        processor_class_name="ColIdefics3Processor",
        lancedb_table="colsmol_500m",
        vram_gb=1.2,
        min_transformers="4.56",
        min_colpali_engine="0.3.9",
        description="ColSmol 500M — lightweight, ~1.2GB VRAM",
    ),
    "vidore/colSmol-256M": ModelSpec(
        model_id="vidore/colSmol-256M",
        family="colsmol",
        embedding_type=EmbeddingType.MULTI_VECTOR,
        dimension=128,
        model_class_name="ColIdefics3",
        processor_class_name="ColIdefics3Processor",
        lancedb_table="colsmol_256m",
        vram_gb=0.8,
        min_transformers="4.56",
        min_colpali_engine="0.3.9",
        description="ColSmol 256M — smallest viable ColPali model",
    ),
}


class ModelRegistry:
    """Registry for looking up model metadata by ID or fuzzy name match."""

    def __init__(self):
        self._models = dict(_MODELS)
        # Build reverse lookup: lowercase substrings for fuzzy matching
        self._lookup_keys = {}
        for model_id, spec in self._models.items():
            self._lookup_keys[model_id.lower()] = spec

    def get(self, model_id: str) -> Optional[ModelSpec]:
        """Exact lookup by full model ID."""
        return self._models.get(model_id)

    def detect(self, model_name: str) -> Optional[ModelSpec]:
        """Fuzzy match: find the best ModelSpec for a given name.

        Handles full HuggingFace IDs, partial names, and legacy table names.
        """
        if not model_name:
            return None

        name_lower = model_name.lower()

        # 1. Exact match
        if model_name in self._models:
            return self._models[model_name]

        # 2. Case-insensitive exact match
        for model_id, spec in self._models.items():
            if model_id.lower() == name_lower:
                return spec

        # 3. Input is a substring of a registered model ID
        #    (e.g. "colqwen2.5-3b" matches "tsystems/colqwen2.5-3b-multilingual-v1.0")
        for model_id, spec in self._models.items():
            if name_lower in model_id.lower():
                return spec

        # 4. A registered model ID is a substring of the input
        #    (e.g. full path containing the model name)
        for model_id, spec in self._models.items():
            if model_id.lower() in name_lower:
                return spec

        # 5. Match by family or lancedb_table name (legacy support)
        for spec in self._models.values():
            if spec.family == name_lower or spec.lancedb_table == name_lower:
                return spec

        # 6. Keyword-based fallback: check if all dash-separated parts of the
        #    input appear somewhere in the model ID (handles "colnomic-3b" →
        #    "nomic-ai/colnomic-embed-multimodal-3b")
        input_parts = set(name_lower.replace("/", "-").split("-"))
        input_parts.discard("")
        for model_id, spec in self._models.items():
            id_lower = model_id.lower().replace("/", "-")
            if input_parts and all(part in id_lower for part in input_parts):
                return spec

        return None

    def is_multi_vector(self, model_name: str) -> bool:
        """Check if a model produces multi-vector (ColBERT-style) embeddings."""
        spec = self.detect(model_name)
        return spec is not None and spec.embedding_type == EmbeddingType.MULTI_VECTOR

    def get_lancedb_table(self, model_name: str) -> str:
        """Get the LanceDB table name for a model. Each model gets its own table."""
        spec = self.detect(model_name)
        if spec:
            return spec.lancedb_table
        logger.warning(f"Unknown model '{model_name}', using default table name")
        return "default_lancedb"

    def get_dimension(self, model_name: str) -> int:
        """Get embedding dimension for a model."""
        spec = self.detect(model_name)
        if spec:
            return spec.dimension
        logger.warning(f"Unknown model '{model_name}', using default dimension 128")
        return 128

    def register(self, spec: ModelSpec) -> None:
        """Register a new model at runtime (for plugins/extensions)."""
        self._models[spec.model_id] = spec
        self._lookup_keys[spec.model_id.lower()] = spec
        logger.info(f"Registered model: {spec.model_id} (family={spec.family}, dim={spec.dimension})")

    def list_models(self) -> Dict[str, ModelSpec]:
        """Return all registered models."""
        return dict(self._models)


# Module-level singleton
registry = ModelRegistry()
