from __future__ import annotations

import contextlib
from pathlib import Path

RingStatistics = None


def get_project_root() -> Path:
    """ Returns the project root directory. """
    return Path(__file__).resolve().parent.parent.parent


def get_project_src() -> Path:
    """ Returns the path to the src directory located at the project root. """
    return get_project_root() / "src"


def get_data_dir() -> Path:
    """ Returns the path to the data directory located at the project root. """
    return get_project_root() / "data"


def instantiate_julia() -> None:
    """ 
    Initialises Julia, activates the environment, and imports RingStatistics.  
    """

    global RingStatistics

    # run "which julia" to add the Julia binary to the PATH.

    try:
        from julia import Pkg

        # first activate the environment and instantiate the packages.
        Pkg.activate( str(get_project_src() / "RingStatistics") )
        Pkg.instantiate()

        # now attempt to import the RingStatistics module.
        from julia import RingStatistics as RS

        # assign the RingStatistics module to the global variable.
        RingStatistics = RS
        RS.import_check()

    except ImportError:
        raise ImportError(
            "The julia package is required to run this function. "
            "Please install it using pip."
        ) from None
    

with contextlib.suppress(ImportError):
    instantiate_julia()