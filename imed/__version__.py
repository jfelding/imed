# yourpackage/_version.py
import pathlib

import tomllib

def _get_version():
    pyproject = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    # open in binary mode and let tomllib.load read & parse it
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    # PEP 621 layout
    return data["project"]["version"]

__version__ = _get_version()
