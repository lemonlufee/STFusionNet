"""Training package API (lazy exports).

Avoid importing heavy modules at package import time to prevent side effects
when running `python -m training.train_main`.
"""

__all__ = ["train_run", "_apply_model_params"]


def __getattr__(name: str):
    if name in __all__:
        from .train_main import train_run, _apply_model_params
        mapping = {
            "train_run": train_run,
            "_apply_model_params": _apply_model_params,
        }
        return mapping[name]
    raise AttributeError(name)
