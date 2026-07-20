"""Configuration management using Pydantic + YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class PathsConfig(BaseModel):
    data_dir: str = "data/datasets"
    checkpoints_dir: str = "data/checkpoints"
    logs_dir: str = "data/logs"
    results_dir: str = "results"


class BeamSearchConfig(BaseModel):
    width: int = 10
    max_depth: int = 8
    score_weights: Dict[str, float] = Field(
        default={"train_accuracy": 0.6, "simplicity": 0.2, "consistency": 0.2}
    )


class MCTSConfig(BaseModel):
    simulations: int = 200
    exploration_constant: float = 1.414
    max_depth: int = 10
    rollout_depth: int = 3


class GeneticConfig(BaseModel):
    population_size: int = 50
    generations: int = 30
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elitism: int = 5


class SearchConfig(BaseModel):
    algorithm: str = "beam"
    beam_width: int = 10
    max_iterations: int = 500
    max_program_depth: int = 8
    early_stopping: bool = True
    memoization: bool = True
    parallel_workers: int = 4


class SolverConfig(BaseModel):
    max_time_per_task: int = 60
    max_attempts: int = 2
    ensemble_mode: bool = True
    solvers: List[str] = ["symbolic", "heuristic", "constraint"]
    enable_llm: bool = False


class VisionConfig(BaseModel):
    min_object_size: int = 1
    background_detection: str = "most_frequent"
    symmetry_tolerance: float = 0.0


class DSLConfig(BaseModel):
    max_program_length: int = 12
    enable_conditionals: bool = True
    enable_loops: bool = False


class MemoryConfig(BaseModel):
    backend: str = "json"
    max_size: int = 10000


class VerifierConfig(BaseModel):
    require_exact_match: bool = True
    min_confidence: float = 0.0


class LLMConfig(BaseModel):
    provider: str = "mock"
    model: str = "llama3.2"
    host: str = "http://localhost:11434"
    timeout: int = 30
    max_retries: int = 2


class EvaluationConfig(BaseModel):
    metrics: List[str] = ["exact_match", "object_consistency"]
    save_failures: bool = True
    failure_analysis: bool = True


class SubmissionConfig(BaseModel):
    output_file: str = "submission.json"
    validate_format: bool = True


class ProjectConfig(BaseModel):
    """Root configuration model for the ARC solver."""

    project: Dict[str, Any] = Field(
        default={"name": "arc-solver", "version": "0.1.0", "seed": 42, "log_level": "INFO"}
    )
    paths: PathsConfig = Field(default_factory=PathsConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    beam_search: BeamSearchConfig = Field(default_factory=BeamSearchConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    genetic: GeneticConfig = Field(default_factory=GeneticConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    dsl: DSLConfig = Field(default_factory=DSLConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    submission: SubmissionConfig = Field(default_factory=SubmissionConfig)

    @model_validator(mode="before")
    @classmethod
    def _fill_defaults(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return values


def load_config(config_path: Optional[str | Path] = None) -> ProjectConfig:
    """Load configuration from a YAML file, merging with defaults.

    Args:
        config_path: Path to YAML config file. If None, returns defaults.

    Returns:
        Validated ProjectConfig instance.
    """
    if config_path is None:
        return ProjectConfig()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ProjectConfig.model_validate(raw or {})


def load_default_config() -> ProjectConfig:
    """Load the default config from configs/default.yaml if present, else use defaults."""
    default_path = Path("configs/default.yaml")
    if default_path.exists():
        return load_config(default_path)
    return ProjectConfig()


__all__ = ["ProjectConfig", "load_config", "load_default_config"]
