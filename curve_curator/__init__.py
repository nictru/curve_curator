# ############# #
# Curve Curator #
# ############# #
#
# Florian P. Bayer - 2025
#

__version__ = '0.6.0'

from .api import run_pipeline_api  # noqa: F401 — public API

__all__ = ["__version__", "run_pipeline_api"]
