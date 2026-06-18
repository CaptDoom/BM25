from pydantic import BaseModel, Field
from typing import Dict, Optional

class PreprocessingConfig(BaseModel):
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    use_lemmatization: bool = False

class SpellCorrectionConfig(BaseModel):
    min_similarity: float = Field(default=0.85, ge=0.0, le=1.0)

class BM25Config(BaseModel):
    k1: float = Field(default=1.2, ge=0.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)
    top_k: int = Field(default=25, gt=0)

class DenseConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = Field(default=100, gt=0)

class RRFConfig(BaseModel):
    k: int = Field(default=60, gt=0)
    weights: Dict[str, float] = Field(default_factory=lambda: {"bm25": 1.0, "dense": 1.0})

class RerankerConfig(BaseModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = Field(default=15, gt=0)
    top_k: int = Field(default=10, gt=0)
    batch_size: int = Field(default=32, gt=0)

class ShardingConfig(BaseModel):
    shard_size: int = Field(default=1000000, gt=0)

class SystemConfig(BaseModel):
    dataset_name: str = "BeIR/scidocs"
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    spell_correction: SpellCorrectionConfig = Field(default_factory=SpellCorrectionConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    dense: Optional[DenseConfig] = Field(default_factory=DenseConfig)
    rrf: Optional[RRFConfig] = Field(default_factory=RRFConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    sharding: ShardingConfig = Field(default_factory=ShardingConfig)
    cache_dir: str = "./.cache"
    huggingface: Optional[Dict[str, str]] = None
