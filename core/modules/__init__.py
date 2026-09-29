"""Importing this package registers every built-in builder, converter, backbone, teacher and benchmark."""

from core.modules.data import builders, converters, synthetic  # noqa: F401
from core.modules.evaluation import benchmarks  # noqa: F401
from core.modules.model import backbones  # noqa: F401
