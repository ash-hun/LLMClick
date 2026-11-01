"""Config of `llm_quantize`: an LLM recipe whose `method` is SFT (rows and loss) plus the `quantization:` section."""

from typing import ClassVar

from pydantic import Field, model_validator

from modeling.quantize.quantize_scheme import SCHEMES
from modeling.llm.methods.sft_method import SFT
from modeling.llm.llm_config import LLMConfig
from modeling.modeling_config import Keyed


class QuantizationConfig(Keyed):
    """`name` picks the scheme; other keys go to it (`scheme: W8A16` for gptq, `smoothing` for w8a8, `dampening` for both)."""
    rows: int = Field(default=128, ge=1, description="Training rows used for calibration, the first ones in file order")


class QuantizeConfig(LLMConfig):
    """Rows are SFT conversations: the train fold calibrates, the held-out fold measures the loss before and after."""
    method_class: ClassVar = SFT
    method: SFT.Config = Field(default_factory=SFT.Config)
    quantization: QuantizationConfig

    @model_validator(mode="after")
    def _quantizable(self) -> "QuantizeConfig":
        SCHEMES.check(self.quantization.name)
        if self.training.adapter is not None:
            raise ValueError("llm_quantize trains nothing; remove training.adapter")
        return self
