"""Evaluation package API (lazy exports).

This avoids eager imports that can interfere with `python -m` execution.
"""

__all__ = [
    "evaluate", "plot_timeseries_best_station", "plot_log_scatter_per_feature",
    "inverse_transform_lastdim", "compute_metrics_per_feature",
    "visualize_scatters",
    "analyze_feature_importance_shap", "analyze_temporal_importance_captum",
]


def __getattr__(name: str):
    if name in __all__:
        from .eval_metrics import (
            evaluate, plot_timeseries_best_station, plot_log_scatter_per_feature,
            inverse_transform_lastdim, compute_metrics_per_feature,
            visualize_scatters,
            analyze_feature_importance_shap, analyze_temporal_importance_captum,
        )
        mapping = {
            "evaluate": evaluate,
            "plot_timeseries_best_station": plot_timeseries_best_station,
            "plot_log_scatter_per_feature": plot_log_scatter_per_feature,
            "inverse_transform_lastdim": inverse_transform_lastdim,
            "compute_metrics_per_feature": compute_metrics_per_feature,
            "visualize_scatters": visualize_scatters,
            "analyze_feature_importance_shap": analyze_feature_importance_shap,
            "analyze_temporal_importance_captum": analyze_temporal_importance_captum,
        }
        return mapping[name]
    raise AttributeError(name)
