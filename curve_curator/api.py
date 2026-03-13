# api.py
# Python API for running the CurveCurator pipeline in-process.
#
# Florian P. Bayer / drevalpy - 2025
#

import contextlib
import io
import os

import pandas as pd

from . import data_parser, quality_control, quantification, thresholding, toml_parser


def run_pipeline_api(config: dict, *, mad: bool = False) -> pd.DataFrame:
    """Run the CurveCurator pipeline in-process from a pre-built config dict.

    Pure function: accepts a config dict, returns a fitted ``pd.DataFrame``,
    and performs **no disk I/O**.  All caching is the caller's responsibility.

    The config dict must satisfy two requirements before being passed here:

    1. All values in ``config['Paths']`` must be **absolute paths** (so that
       ``toml_parser.set_default_values`` → ``update_toml_paths`` is a no-op).
    2. A ``'__file__'`` key must be present:
       ``config['__file__'] = {'Path': '/abs/path/to/config.toml'}``

    stdout from CurveCurator's internal ``user_interface`` module is suppressed.

    Parameters
    ----------
    config:
        Config dict in CurveCurator TOML structure, with absolute paths and
        ``__file__`` injected (see above).
    mad:
        Whether to run the MAD outlier analysis step.  Defaults to ``False``
        because ``mad_analysis`` writes ``mad.txt`` to disk, violating the
        no-disk-I/O contract of this function.  Pass ``mad=True`` only when
        the output directory is intentionally writable.

    Returns
    -------
    pd.DataFrame
        Fitted curves table in CurveCurator output format.
    """
    config = toml_parser.set_default_values(config)
    with contextlib.redirect_stdout(io.StringIO()):
        # Suppress tqdm progress bars that some quantification internals emit
        os.environ.setdefault("TQDM_DISABLE", "1")
        data = data_parser.load(config)
        data = quantification.run_pipeline(data, config=config)
        data = thresholding.apply_significance_thresholds(data, config=config)
        if mad:
            quality_control.mad_analysis(data, config=config)
    return data
