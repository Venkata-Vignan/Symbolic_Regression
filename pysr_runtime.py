import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
JULIA_DEPOT = PROJECT_ROOT / ".julia_depot"
JULIA_DEPOT.mkdir(exist_ok=True)
JULIA_PROJECT = PROJECT_ROOT / "venv" / "julia_env"


def configure_pysr_environment():
    existing = os.environ.get("JULIA_DEPOT_PATH", "")
    default_depots = [
        str(JULIA_DEPOT),
        str(Path.home() / ".julia"),
        str(Path.home() / "AppData" / "Local" / "Programs" / "Julia-1.10.10" / "local" / "share" / "julia"),
        str(Path.home() / "AppData" / "Local" / "Programs" / "Julia-1.10.10" / "share" / "julia"),
    ]

    if existing:
        os.environ["JULIA_DEPOT_PATH"] = existing
    else:
        os.environ["JULIA_DEPOT_PATH"] = ";".join(default_depots)

    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(JULIA_PROJECT))
    os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")


def load_pysr_pickle(model_path):
    configure_pysr_environment()
    import pickle

    with open(model_path, "rb") as file:
        return pickle.load(file)
