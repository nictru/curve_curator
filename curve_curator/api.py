# api.py
# Python API for running the CurveCurator pipeline in-process.
#
# Florian P. Bayer / drevalpy - 2025
#

import os

import pandas as pd

from . import data_parser, quality_control, quantification, thresholding, toml_parser, torch_fitting


def run_pipeline_api(config: dict, *, mad: bool = False, device: str = "cpu") -> pd.DataFrame:
    """Run the CurveCurator pipeline in-process from a pre-built config dict.

    Uses a batched PyTorch LBFGS fitting backend that runs on CPU or GPU.

    Pure function: accepts a config dict, returns a fitted ``pd.DataFrame``,
    and performs **no disk I/O**.  All caching is the caller's responsibility.

    The config dict must satisfy two requirements before being passed here:

    1. All values in ``config['Paths']`` must be **absolute paths** (so that
       ``toml_parser.set_default_values`` → ``update_toml_paths`` is a no-op).
    2. A ``'__file__'`` key must be present:
       ``config['__file__'] = {'Path': '/abs/path/to/config.toml'}``

    CurveCurator's internal print statements are routed through a NullHandler
    logger (configured in ``user_interface.py``) so no output reaches
    sys.stdout/stderr from worker threads.

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
    device:
        PyTorch device string for the fitting backend, e.g. ``"cpu"``,
        ``"cuda"``, ``"cuda:0"``, ``"mps"``.  Falls back to CPU automatically
        if the requested device is unavailable.

    Returns
    -------
    pd.DataFrame
        Fitted curves table in CurveCurator output format.
    """
    # Suppress tqdm progress bars that some quantification internals emit
    os.environ.setdefault("TQDM_DISABLE", "1")
    config = toml_parser.set_default_values(config)
    data = data_parser.load(config)
    data, _preprocess_result = quantification._preprocess(data, config)
    data = torch_fitting.batch_fit_4pl(data, config, device=device)
    data = thresholding.apply_significance_thresholds(data, config=config)
    if mad:
        quality_control.mad_analysis(data, config=config)
    return data
