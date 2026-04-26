"""Data pipeline package API (lazy exports).

Keep package import light-weight and side-effect free.
"""

__all__ = [
    "load_raw_data", "choose_target_lakes", "physical_cleaning",
    "split_by_time_per_lake_train_val_test", "impute_strict_per_lake",
    "add_time_features", "join_optional_meteo", "fit_scaler",
    "normalize_df", "augment_series_per_lake", "build_windows_grouped",
    "build_graph_windows_from_df",
]


def __getattr__(name: str):
    if name in __all__:
        from .data_pipeline import (
            load_raw_data, choose_target_lakes, physical_cleaning,
            split_by_time_per_lake_train_val_test, impute_strict_per_lake,
            add_time_features, join_optional_meteo, fit_scaler,
            normalize_df, augment_series_per_lake, build_windows_grouped,
            build_graph_windows_from_df,
        )
        mapping = {
            "load_raw_data": load_raw_data,
            "choose_target_lakes": choose_target_lakes,
            "physical_cleaning": physical_cleaning,
            "split_by_time_per_lake_train_val_test": split_by_time_per_lake_train_val_test,
            "impute_strict_per_lake": impute_strict_per_lake,
            "add_time_features": add_time_features,
            "join_optional_meteo": join_optional_meteo,
            "fit_scaler": fit_scaler,
            "normalize_df": normalize_df,
            "augment_series_per_lake": augment_series_per_lake,
            "build_windows_grouped": build_windows_grouped,
            "build_graph_windows_from_df": build_graph_windows_from_df,
        }
        return mapping[name]
    raise AttributeError(name)
