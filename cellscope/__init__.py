__version__ = "1.0.0"

__all__ = [
    "run_pipeline",
    "__version__",
]


def run_pipeline(*args, **kwargs):
    """Lazily import the heavy pipeline for library callers."""
    from .pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)
