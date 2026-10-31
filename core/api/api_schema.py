"""Request and response bodies of the API; the config itself is the recipe's schema (core.config.config_schema.BaseConfig)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

EXAMPLE_CONFIG: dict[str, Any] = {"pipeline": {"recipe": "llm_sft", "name": "sft-qwen3.5-0.8b", "seed": 20260920, "output_dir": "./output"},
                                  "model": {"architecture": "hybrid", "name": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17"},
                                  "data": {"sources": [{"name": "local_jsonl", "path": "samples/llm_sft.jsonl"}], "validation": 0.1},
                                  "training": {"epochs": 1, "lr": 1.0e-5, "batch_size": 4, "accumulation": 2, "max_length": 512, "max_steps": 4},
                                  "validation": {"max": {"loss": 5.0}}}
EXAMPLE_PLAN: dict[str, Any] = {"recipe": "llm_sft", "experiment": "sft-qwen3.5-0.8b-14fc97a0", "directory": "output/sft-qwen3.5-0.8b-14fc97a0",
                              "stages": ["data", "train", "validate"]}


class ConfigSummary(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"valid": True, **EXAMPLE_PLAN, "config": EXAMPLE_CONFIG}]})
    valid: bool = Field(default=True, description="Always true here: an invalid config is a 422 instead")
    recipe: str = Field(description="The recipe the config names (`pipeline.recipe`)")
    experiment: str = Field(description="Experiment key: the config's name and a hash of what decides its result")
    directory: str = Field(description="Where the experiment's files go")
    stages: list[str] = Field(description="Stages a run would execute or reuse, in order")
    config: dict[str, Any] = Field(description="The config with every default filled in")


class JobRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"config_path": "configs/llm/sft.yaml"}, {"config": EXAMPLE_CONFIG}]})
    config_path: str | None = Field(default=None, description="YAML path on the server, relative to the working directory")
    config: dict[str, Any] | None = Field(default=None, description="Inline config with the same shape as the YAML")

    @model_validator(mode="after")
    def _one_source(self) -> "JobRequest":
        if (self.config_path is None) == (self.config is None):
            raise ValueError("Give exactly one of config_path or config")
        return self


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "experiment": "sft-qwen3.5-0.8b-benchmarks-e7b8dffd", "recipe": "evaluation_benchmark",
        "directory": "output/sft-qwen3.5-0.8b-benchmarks-e7b8dffd", "report": "output/sft-qwen3.5-0.8b-benchmarks-e7b8dffd/report/report.json",
        "model": {"experiment": "output/sft-qwen3.5-0.8b-14fc97a0", "checkpoint": "validate", "recipe": "llm_sft"},
        "scores": {"arc_easy": 0.41}, "contamination": None}]})
    experiment: str = Field(description="The evaluation experiment's key")
    recipe: str = Field(description="The evaluation recipe that wrote the report")
    directory: str = Field(description="The evaluation experiment's directory")
    report: str = Field(description="Path of report.json")
    model: dict[str, Any] = Field(description="The evaluated model (experiment, checkpoint, recipe), or the runs of a comparison")
    scores: dict[str, Any] = Field(description="The headline numbers; the report holds everything")
    contamination: dict[str, Any] | None = None


class JobResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "job_id": "sft-qwen3.5-0.8b-14fc97a0-1f2e3d", "status": "running", **EXAMPLE_PLAN,
        "progress": {"stage": "train", "done": 2, "total": 4, "note": "loss 1.9831"}, "device_slot": 0, "result": None, "error": None}]})
    job_id: str = Field(description="The experiment key plus a hash of the stages run: the same request gives the same job")
    status: str = Field(description="pending, running, done, failed, cancelled or interrupted")
    recipe: str = Field(description="The recipe the config names")
    experiment: str = Field(description="Experiment key")
    directory: str = Field(description="Where the experiment's files go")
    stages: list[str] = Field(description="Stages this job runs or reuses, in order")
    progress: dict[str, Any] = Field(description="The running stage and how far it is")
    device_slot: int | None = Field(default=None, description="Which worker, and so which accelerator, ran the job")
    result: dict[str, Any] | None = Field(default=None, description="Every stage's outputs, once the job is done")
    error: str | None = Field(default=None, description="Why the job failed")
