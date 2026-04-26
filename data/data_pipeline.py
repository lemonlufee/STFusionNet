import os
import warnings
import numpy as np
import pandas as pd
import math
import torch
from typing import List, Tuple, Optional, Dict, Union, overload, Literal, Any
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.experimental import enable_iterative_imputer  # type: ignore  # noqa: F401
    from sklearn.impute import IterativeImputer
except Exception:  # pragma: no cover
    IterativeImputer = None  # type: ignore[assignment]
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_taihu import Config


# -----------------------------------------------------------------------------
# Caches & one-time logging guards
#
# During grid-search, the project may call the data-preparation pipeline many
# times (once per hyper-parameter combo). Re-reading the raw CSV and printing
# dataset summaries repeatedly will spam the terminal and slow down tuning.
# We cache the raw dataframe and the selected-station subset, and print key
# summaries only once per process.
# -----------------------------------------------------------------------------

_RAW_DF_CACHE: Dict[str, pd.DataFrame] = {}
_SELECTED_DF_CACHE: Dict[str, pd.DataFrame] = {}
_PRINT_ONCE_FLAGS: set[str] = set()


def _print_once(flag: str, msg: str) -> None:
    if flag in _PRINT_ONCE_FLAGS:
        return
    _PRINT_ONCE_FLAGS.add(flag)
    print(msg)


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Best-effort conversion to float.

    VS Code / Pylance may treat pandas/numpy scalars as a broad "Scalar" type
    that could be complex, which triggers static type errors on float(...).
    This helper accepts `Any` and performs robust casting, while keeping
    runtime behavior unchanged for normal float-like inputs.
    """

    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        # Fallback: take real-part if it's a complex-like scalar
        try:
            return float(np.real(x))
        except Exception:
            return default


def _safe_int(x: Any, default: int = 0) -> int:
    """Best-effort conversion to int for pandas/numpy scalar-like objects."""
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

def load_raw_data(cfg: Config) -> pd.DataFrame:
    """Load raw CSV (cached) and normalize dtypes.

    Notes
    -----
    - Grid-search calls this many times; we cache the processed dataframe.
    - The original CSV may contain mixed-type numeric columns (triggering
      pandas DtypeWarning). We read as strings and then coerce numeric
      columns explicitly.
    """

    if not os.path.exists(cfg.RAW_DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {cfg.RAW_DATA_FILE}")

    cache_key = os.path.abspath(cfg.RAW_DATA_FILE)
    if cache_key in _RAW_DF_CACHE:
        return _RAW_DF_CACHE[cache_key].copy()

    # Read as strings to avoid mixed-type inference issues.
    # Try common encodings used by local/Windows-exported CSV files.
    df = None
    read_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.DtypeWarning)
                df = pd.read_csv(cfg.RAW_DATA_FILE, low_memory=False, dtype=str, encoding=enc)
            break
        except Exception as e:
            read_err = e
            df = None
    if df is None:
        if read_err is not None:
            raise read_err
        raise RuntimeError(f"Failed to read CSV: {cfg.RAW_DATA_FILE}")

    # ---- Normalize column names: trim spaces / remove BOM / case-insensitive matching ----
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    col_lower = {c.lower(): c for c in df.columns}

    # ---- Required columns ----
    if "id" not in col_lower or "date" not in col_lower:
        raise ValueError(f"Required columns missing: ID/Date. Current columns={df.columns.tolist()}")
    if col_lower["id"] != "ID":
        df = df.rename(columns={col_lower["id"]: "ID"})
    if col_lower["date"] != "Date":
        df = df.rename(columns={col_lower["date"]: "Date"})

    # ---- Geographic columns: support lon/lat, Lon/Lat, longitude/latitude ----
    lon_key = None
    lat_key = None
    for k in ["lon", "longitude", "lng"]:
        if k in col_lower:
            lon_key = col_lower[k]
            break
    for k in ["lat", "latitude"]:
        if k in col_lower:
            lat_key = col_lower[k]
            break

    if lon_key is not None and lon_key != "lon":
        df = df.rename(columns={lon_key: "lon"})
    if lat_key is not None and lat_key != "lat":
        df = df.rename(columns={lat_key: "lat"})

    # Coerce numeric columns explicitly (feature/target/lon/lat)
    numeric_cols = set(getattr(cfg, "FEATURE_COLS", []) + getattr(cfg, "TARGET_FEATURES", []))
    numeric_cols.update(["lon", "lat"])
    for c in numeric_cols:
        if c in df.columns:
            s = df[c].astype(str).str.strip()
            df[c] = pd.to_numeric(s, errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    _RAW_DF_CACHE[cache_key] = df
    return df.copy()

def choose_target_lakes(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Select stations.

    Legacy behavior selected top-K by data volume only. New behavior supports manual
    selection first, then geographic selection (nearest stations to anchor/centroid).
    """
    # Cache selected stations because tuning calls this repeatedly.
    mode = str(getattr(cfg, "STATION_SELECT_MODE", "geo")).lower()
    manual_ids = list(getattr(cfg, "MANUAL_STATION_IDS", []) or [])
    anchor_id = str(getattr(cfg, "GEO_ANCHOR_ID", "") or "")
    anchor_lon = getattr(cfg, "GEO_ANCHOR_LON", None)
    anchor_lat = getattr(cfg, "GEO_ANCHOR_LAT", None)
    radius = getattr(cfg, "GEO_RADIUS_KM", None)
    cache_key = "|".join(
        [
            os.path.abspath(getattr(cfg, "RAW_DATA_FILE", "")),
            f"mode={mode}",
            f"topk={int(cfg.TOP_K_LAKES)}",
            f"minsteps={int(cfg.MIN_EFFECTIVE_STEPS)}",
            f"anchor_id={anchor_id}",
            f"anchor_lon={anchor_lon}",
            f"anchor_lat={anchor_lat}",
            f"radius={radius}",
            "manual=" + ",".join([str(x) for x in manual_ids]),
        ]
    )
    if cache_key in _SELECTED_DF_CACHE:
        return _SELECTED_DF_CACHE[cache_key].copy()

    dff = df.copy()
    dff["__ID_STR__"] = dff["ID"].astype(str)

    valid_mask = dff[cfg.FEATURE_COLS].notna().any(axis=1)
    counts = (
        dff.loc[valid_mask]
        .groupby("__ID_STR__")["Date"]
        .nunique()
        .sort_values(ascending=False)
    )

    _print_once(f"station_detect:{cache_key}", f"Detected {len(counts)} stations.")
    qualified = counts[counts >= cfg.MIN_EFFECTIVE_STEPS]
    if qualified.empty:
        raise RuntimeError(f"No station has at least {cfg.MIN_EFFECTIVE_STEPS} effective time steps.")
    topk = int(getattr(cfg, "TOP_K_LAKES", -1))
    use_all = topk <= 0

    # `mode` and `manual_ids` already extracted for caching.

    # ------------------------------
    # 1) Manual selection
    # ------------------------------
    if mode == "manual" and len(manual_ids) > 0:
        manual_str = [str(x) for x in manual_ids]
        exist = [x for x in manual_str if x in qualified.index]
        if len(exist) == 0:
            print("[WARN] MANUAL_STATION_IDS not found or not qualified; fallback to geo selection.")
        else:
            chosen = exist if use_all else exist[:topk]
            _print_once(
                f"station_select:{cache_key}",
                f"Manual station selection: {len(chosen)} station(s) selected.",
            )
            out = dff[dff["__ID_STR__"].isin(chosen)].copy()
            out = out.drop(columns=["__ID_STR__"], errors="ignore")
            _SELECTED_DF_CACHE[cache_key] = out
            return out.copy()

    # ------------------------------
    # 2) Geographic-related selection
    # ------------------------------
    # Get per-station coordinates (mean of available lon/lat)
    coord = (
        dff.groupby("__ID_STR__")[["lon", "lat"]].mean(numeric_only=True)
        if ("lon" in dff.columns and "lat" in dff.columns)
        else pd.DataFrame(index=qualified.index, columns=["lon", "lat"])
    )

    # Keep only qualified
    coord = coord.reindex(qualified.index)
    coord = coord.dropna(subset=["lon", "lat"], how="any")

    if coord.empty:
        # fallback: still pick top-K by data length (no coordinates)
        top_ids = list(qualified.index) if use_all else list(qualified.head(topk).index)
        _print_once(
            f"station_select:{cache_key}",
            f"[WARN] lon/lat unavailable; fallback to top-{len(top_ids)} by data coverage.",
        )
        out = dff[dff["__ID_STR__"].isin([str(x) for x in top_ids])].copy()
        out = out.drop(columns=["__ID_STR__"], errors="ignore")
        _SELECTED_DF_CACHE[cache_key] = out
        return out.copy()

    # Determine anchor
    anchor_lon = getattr(cfg, "GEO_ANCHOR_LON", None)
    anchor_lat = getattr(cfg, "GEO_ANCHOR_LAT", None)
    anchor_id = str(getattr(cfg, "GEO_ANCHOR_ID", "") or "")
    if (anchor_lon is not None) and (anchor_lat is not None):
        lon0, lat0 = _safe_float(anchor_lon), _safe_float(anchor_lat)
    elif anchor_id and anchor_id in coord.index:
        lon0, lat0 = _safe_float(coord.loc[anchor_id, "lon"]), _safe_float(coord.loc[anchor_id, "lat"])
    else:
        lon0, lat0 = _safe_float(coord["lon"].mean()), _safe_float(coord["lat"].mean())

    # Vectorized haversine distance (km)
    lon1 = np.deg2rad(coord["lon"].to_numpy(np.float64))
    lat1 = np.deg2rad(coord["lat"].to_numpy(np.float64))
    lon0r = np.deg2rad(lon0)
    lat0r = np.deg2rad(lat0)
    dlon = lon1 - lon0r
    dlat = lat1 - lat0r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0r) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    dist_km = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    dist_s = pd.Series(dist_km, index=coord.index, name="dist_km")

    radius = getattr(cfg, "GEO_RADIUS_KM", None)
    if radius is not None:
        radius = float(radius)
        dist_s = dist_s[dist_s <= radius]
        if dist_s.empty:
            # radius too strict: ignore radius
            dist_s = pd.Series(dist_km, index=coord.index, name="dist_km")

    chosen = list(dist_s.sort_values().index) if use_all else list(dist_s.sort_values().head(topk).index)
    _print_once(
        f"station_select:{cache_key}",
        f"Geo station selection: {len(chosen)} station(s) selected, anchor=({lon0:.4f},{lat0:.4f}).",
    )
    out = dff[dff["__ID_STR__"].isin(chosen)].copy()
    out = out.drop(columns=["__ID_STR__"], errors="ignore")
    _SELECTED_DF_CACHE[cache_key] = out
    return out.copy()

def physical_cleaning(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    for col, (mn, mx) in cfg.PHYSICAL_LIMITS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mask = (df[col] < mn) | (df[col] > mx)
            df.loc[mask, col] = np.nan
    return df

def split_by_time_per_lake(df: pd.DataFrame, cfg: Config, train_ratio: float = 0.8):
    """Backward-compatible 2-way split wrapper.

    This function is kept only for compatibility with old scripts.
    Internally it calls the strict 3-way splitter and merges val+test to test.
    """
    _print_once("deprecated_split_2way", "[WARN] split_by_time_per_lake is deprecated. Use split_by_time_per_lake_train_val_test.")
    if df.empty:
        return pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)
    val_ratio = float(max(0.0, min(0.49, 1.0 - float(train_ratio) - 0.01)))
    df_train, df_val, df_test, _ = split_by_time_per_lake_train_val_test(
        df,
        cfg,
        train_ratio=float(train_ratio),
        val_ratio=val_ratio,
        overlap=getattr(cfg, "SPLIT_OVERLAP", cfg.SEQ_LEN),
    )
    if df_train.empty:
        return pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)
    df_test_2way = pd.concat([df_val, df_test], ignore_index=True) if (not df_val.empty or not df_test.empty) else pd.DataFrame(columns=df.columns)
    return df_train, df_test_2way


def split_by_time_per_lake_train_val_test(
    df: pd.DataFrame,
    cfg: Config,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    overlap: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Strict Train/Val/Test split per station (ID), time-ordered.

    - Train/Val/Test are split by ratio on each station's timeline.
    - Val/Test are given a warm-up *history overlap* (default: SEQ_LEN) so that
      windowing can start immediately, while labels remain strictly after the split.

    Returns:
        df_train, df_val, df_test, split_meta
    """

    if df.empty:
        empty = pd.DataFrame(columns=df.columns)
        return empty, empty, empty, {"skipped_ids": [], "reason": "empty_df"}

    train_ratio = float(train_ratio if train_ratio is not None else getattr(cfg, "TRAIN_RATIO", 0.7))
    val_ratio = float(val_ratio if val_ratio is not None else getattr(cfg, "VAL_RATIO", 0.1))
    if train_ratio <= 0 or val_ratio < 0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError(f"Invalid split ratios: train_ratio={train_ratio}, val_ratio={val_ratio}")

    overlap = int(overlap if overlap is not None else getattr(cfg, "SPLIT_OVERLAP", cfg.SEQ_LEN))
    # Guard against label leakage: if overlap > SEQ_LEN, some Val/Test labels may fall into Train period.
    seq_len = int(getattr(cfg, "SEQ_LEN", 30))
    if overlap > seq_len:
        overlap = seq_len

    overlap = max(0, overlap)

    id_col = getattr(cfg, "NODE_ID_COL", "ID")
    time_col = getattr(cfg, "TIME_COL", "Date")
    dff = df.copy()
    dff[time_col] = pd.to_datetime(dff[time_col], errors="coerce")
    dff = dff.dropna(subset=[id_col, time_col]).copy()

    seq_len = int(cfg.SEQ_LEN)
    pred_len = int(getattr(cfg, "PRED_LEN", 1))
    min_need = seq_len + pred_len

    train_chunks, val_chunks, test_chunks = [], [], []
    per_lake = {}
    skipped = []

    for lake_id, g in dff.groupby(id_col):
        g = g.sort_values(time_col).reset_index(drop=True)
        n = len(g)
        # need enough points for train + val + test to each form at least 1 window
        if n < (min_need * 3):
            skipped.append(str(lake_id))
            continue

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        # enforce minimum sizes
        train_end = max(train_end, min_need)
        val_end = max(val_end, train_end + min_need)

        # ensure test has enough
        if (n - val_end) < min_need:
            val_end = n - min_need
        if val_end <= train_end or (n - val_end) < min_need:
            skipped.append(str(lake_id))
            continue

        train_df = g.iloc[:train_end].copy()
        val_start = max(0, train_end - overlap)
        val_df = g.iloc[val_start:val_end].copy()
        test_start = max(0, val_end - overlap)
        test_df = g.iloc[test_start:].copy()

        # windowing sanity
        if len(train_df) < min_need or len(val_df) < min_need or len(test_df) < min_need:
            skipped.append(str(lake_id))
            continue

        train_chunks.append(train_df)
        val_chunks.append(val_df)
        test_chunks.append(test_df)

        per_lake[str(lake_id)] = {
            "n": int(n),
            "train_end_idx": int(train_end),
            "val_end_idx": int(val_end),
            "train_end_time": str(g.loc[train_end - 1, time_col]),
            "val_end_time": str(g.loc[val_end - 1, time_col]),
        }

    if not train_chunks:
        empty = pd.DataFrame(columns=df.columns)
        meta = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": float(1.0 - train_ratio - val_ratio),
            "overlap": overlap,
            "skipped_ids": skipped,
            "per_lake": {},
        }
        return empty, empty, empty, meta

    df_train = pd.concat(train_chunks, ignore_index=True)
    df_val = pd.concat(val_chunks, ignore_index=True)
    df_test = pd.concat(test_chunks, ignore_index=True)

    meta = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": float(1.0 - train_ratio - val_ratio),
        "overlap": overlap,
        "skipped_ids": skipped,
        "per_lake": per_lake,
    }
    return df_train, df_val, df_test, meta



@overload
def impute_strict_per_lake(
    df: pd.DataFrame,
    cfg: Config,
    *,
    train_means: Optional[Dict[str, Any]] = ...,
    return_train_means: Literal[False] = ...,
) -> pd.DataFrame: ...

@overload
def impute_strict_per_lake(
    df: pd.DataFrame,
    cfg: Config,
    *,
    train_means: Optional[Dict[str, Any]] = ...,
    return_train_means: Literal[True],
) -> Tuple[pd.DataFrame, Dict[str, Any]]: ...


def impute_strict_per_lake(
    df: pd.DataFrame,
    cfg: Config,
    *,
    train_means: Optional[Dict[str, Any]] = None,
    return_train_means: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Align each station (ID) to a fixed time grid (default 4H), average duplicates per grid,
    then impute missing values.

    Hybrid strategy (research version):
      1) short gap (<= 1 day) linear interpolation (inside gaps only)
      2) long gap (> 1 day) spatial IDW at same timestamp
      3) still missing: MICE (cross-feature iterative imputation)
      4) final fallback: historical seasonal mean (month-hour + week-hour climatology), then train means

    IMPORTANT (leakage control):
      - To avoid Train/Val/Test leakage, DO NOT run this on the concatenated full dataset.
        Instead, run it separately for each split.
      - For Val/Test, pass train_means computed from the (imputed) training split only.

    Returns:
      - DataFrame with columns: ["ID","Date", <FEATURE_COLS>, "lon","lat"]
      - If return_train_means=True: also returns the dict of train_means.
    """
    if df.empty:
        if return_train_means:
            return df.copy(), (train_means or {})
        return df.copy()

    method = str(getattr(cfg, "IMPUTE_METHOD", "linear")).lower().strip()
    if method not in {"linear", "spatial", "mice", "hybrid"}:
        method = "linear"
    interp_limit = getattr(cfg, "INTERP_LIMIT_STEPS", None)
    if interp_limit is not None:
        interp_limit = int(interp_limit)
        if interp_limit <= 0:
            interp_limit = None
    short_gap_hours = float(max(1.0, getattr(cfg, "HYBRID_SHORT_GAP_MAX_HOURS", 24.0)))
    try:
        freq_td = pd.to_timedelta(str(getattr(cfg, "RESAMPLE_FREQ", "4h")))
        sec = max(1.0, float(freq_td.total_seconds()))
        short_gap_steps = int(max(1, round((short_gap_hours * 3600.0) / sec)))
    except Exception:
        # Fallback for unknown freq parsing.
        short_gap_steps = 6

    # Ensure datetime
    out_df = df.copy()
    out_df["Date"] = pd.to_datetime(out_df["Date"], errors="coerce")
    out_df = out_df.dropna(subset=["Date", "ID"]).copy()

    # We impute numeric water-quality variables; lon/lat handled separately as constants per station.
    value_cols = list(cfg.FEATURE_COLS)
    # In case TARGET_FEATURES contains extra cols not in FEATURE_COLS
    for c in getattr(cfg, "TARGET_FEATURES", []):
        if c not in value_cols:
            value_cols.append(c)

    def process_one_station(g: pd.DataFrame) -> pd.DataFrame:
        if g.empty:
            return pd.DataFrame()

        g = g.sort_values("Date").set_index("Date")

        # Keep only numeric cols (coerce), then resample to fixed grid with mean aggregation.
        gg = g[value_cols].copy()
        for c in gg.columns:
            gg[c] = pd.to_numeric(gg[c], errors="coerce")

        # Align to fixed time grid and average duplicates within each bin.
        res = gg.resample(cfg.RESAMPLE_FREQ).mean()

        # Time interpolation:
        # - linear: keep legacy behavior (inside + boundary)
        # - hybrid: ONLY short inside gaps (<=1 day), no boundary ffill/bfill
        if method == "linear":
            res = res.interpolate(method="time", limit=interp_limit, limit_area="inside")
            for c in res.columns:
                s = res[c]
                fv = s.first_valid_index()
                lv = s.last_valid_index()
                if fv is None or lv is None:
                    continue
                if s.loc[:fv].isna().any():
                    s.loc[:fv] = s.loc[:fv].bfill()
                if s.loc[lv:].isna().any():
                    s.loc[lv:] = s.loc[lv:].ffill()
                res[c] = s
        elif method == "hybrid":
            res = res.interpolate(method="time", limit=short_gap_steps, limit_area="inside")

        # lon/lat: keep constant (first non-null) per station if provided
        lon_const = np.nan
        lat_const = np.nan
        if "lon" in g.columns:
            s = pd.to_numeric(g["lon"], errors="coerce").dropna()
            if len(s) > 0:
                lon_const = float(s.iloc[0])
        if "lat" in g.columns:
            s = pd.to_numeric(g["lat"], errors="coerce").dropna()
            if len(s) > 0:
                lat_const = float(s.iloc[0])

        res["lon"] = lon_const
        res["lat"] = lat_const
        res["ID"] = g["ID"].iloc[0] if "ID" in g.columns else None
        return res.reset_index()

    parts: List[pd.DataFrame] = []
    for _sid, g in out_df.groupby("ID"):
        parts.append(process_one_station(g))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["Date","ID"] + value_cols + ["lon","lat"])

    def _spatial_idw_fill(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        if data.empty or ("lon" not in data.columns) or ("lat" not in data.columns):
            return data
        sid_list = sorted(data["ID"].astype(str).unique().tolist())
        if len(sid_list) < 2:
            return data
        coord = data.groupby("ID")[["lon", "lat"]].median(numeric_only=True).reindex(sid_list)
        coord = coord.dropna()
        sid_list = [str(s) for s in coord.index.tolist()]
        if len(sid_list) < 2:
            return data
        lons = coord["lon"].astype(float).to_numpy()
        lats = coord["lat"].astype(float).to_numpy()
        n = len(sid_list)
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                dij = haversine_km(float(lons[i]), float(lats[i]), float(lons[j]), float(lats[j]))
                dist[i, j] = dij
                dist[j, i] = dij

        k = int(max(1, getattr(cfg, "SPATIAL_K", 4)))
        p = float(max(0.5, getattr(cfg, "SPATIAL_POWER", 2.0)))
        eps = 1e-6

        out_local = data.copy()
        for c in cols:
            piv = out_local.pivot(index="Date", columns="ID", values=c).reindex(columns=sid_list)
            if piv.empty:
                continue
            vals = piv.to_numpy(dtype=np.float64, copy=True)
            for i in range(n):
                miss_rows = np.where(~np.isfinite(vals[:, i]))[0]
                if miss_rows.size == 0:
                    continue
                nn_idx = np.argsort(dist[i])
                nn_idx = [j for j in nn_idx if j != i][:k]
                if len(nn_idx) == 0:
                    continue
                d = dist[i, nn_idx]
                w = 1.0 / np.power(d + eps, p)
                for r in miss_rows:
                    neigh_vals = vals[r, nn_idx]
                    ok = np.isfinite(neigh_vals)
                    if not np.any(ok):
                        continue
                    ww = w[ok]
                    vv = neigh_vals[ok]
                    sw = float(np.sum(ww))
                    if sw <= 0:
                        continue
                    vals[r, i] = float(np.sum(ww * vv) / sw)
            filled = pd.DataFrame(vals, index=piv.index, columns=piv.columns)
            # Keep type-checker friendly API (some stubs do not expose dropna),
            # and rely on left-merge to preserve originally missing rows.
            long = (
                filled.reset_index()
                .melt(id_vars=["Date"], var_name="ID", value_name=c)
            )
            out_local = out_local.drop(columns=[c]).merge(long, on=["Date", "ID"], how="left")
        return out_local

    def _mice_fill(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        if data.empty or len(cols) == 0:
            return data
        if IterativeImputer is None:
            _print_once("warn_no_mice", "[WARN] sklearn IterativeImputer unavailable; skip MICE imputation.")
            return data
        out_local = data.copy()
        block = out_local[cols].apply(pd.to_numeric, errors="coerce")
        mask_nan = block.isna()
        if int(mask_nan.to_numpy().sum()) == 0:
            return out_local
        try:
            imputer = IterativeImputer(
                max_iter=int(max(5, getattr(cfg, "MICE_MAX_ITER", 20))),
                random_state=int(getattr(cfg, "MICE_RANDOM_SEED", 2025)),
                sample_posterior=False,
            )
            imp = imputer.fit_transform(block.to_numpy(dtype=np.float64))
            imp_df = pd.DataFrame(imp, columns=cols, index=block.index)
            out_local[cols] = block.where(~mask_nan, imp_df)
        except Exception as e:
            _print_once("warn_mice_failed", f"[WARN] MICE imputation failed, fallback without MICE: {e}")
        return out_local

    def _build_seasonal_profile(data: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
        profile: Dict[str, Any] = {
            "id_month_hour": {},
            "id_dow_hour": {},
            "id_hour": {},
            "id_month": {},
            "global_month_hour": {},
            "global_dow_hour": {},
            "global_hour": {},
            "global_month": {},
            "id_global": {},
            "global": {},
        }
        if data.empty:
            return profile
        tmp = data[["ID", "Date"] + cols].copy()
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        tmp = tmp.dropna(subset=["ID", "Date"])
        if tmp.empty:
            return profile
        tmp["month"] = pd.DatetimeIndex(tmp["Date"].values).month.astype(np.int16)
        tmp["dow"] = pd.DatetimeIndex(tmp["Date"].values).dayofweek.astype(np.int16)
        tmp["hour"] = pd.DatetimeIndex(tmp["Date"].values).hour.astype(np.int16)
        tmp["ID"] = tmp["ID"].astype(str)
        for c in cols:
            s = pd.to_numeric(tmp[c], errors="coerce")
            t = tmp.loc[s.notna(), ["ID", "month", "dow", "hour"]].copy()
            if t.empty:
                profile["global"][c] = 0.0
                continue
            t["v"] = s[s.notna()].astype(float).to_numpy()
            g_imh = t.groupby(["ID", "month", "hour"])["v"].mean()
            g_idh = t.groupby(["ID", "dow", "hour"])["v"].mean()
            g_ih = t.groupby(["ID", "hour"])["v"].mean()
            g_im = t.groupby(["ID", "month"])["v"].mean()
            g_mh = t.groupby(["month", "hour"])["v"].mean()
            g_dh = t.groupby(["dow", "hour"])["v"].mean()
            g_h = t.groupby(["hour"])["v"].mean()
            g_m = t.groupby(["month"])["v"].mean()
            g_i = t.groupby(["ID"])["v"].mean()
            id_month_hour: Dict[Tuple[str, int, int], float] = {}
            for key, v in g_imh.items():
                if not isinstance(key, tuple) or len(key) != 3:
                    continue
                sid_k, mon_k, hour_k = key
                id_month_hour[(str(sid_k), _safe_int(mon_k), _safe_int(hour_k))] = float(v)
            profile["id_month_hour"][c] = id_month_hour

            id_dow_hour: Dict[Tuple[str, int, int], float] = {}
            for key, v in g_idh.items():
                if not isinstance(key, tuple) or len(key) != 3:
                    continue
                sid_k, dow_k, hour_k = key
                id_dow_hour[(str(sid_k), _safe_int(dow_k), _safe_int(hour_k))] = float(v)
            profile["id_dow_hour"][c] = id_dow_hour

            id_hour: Dict[Tuple[str, int], float] = {}
            for key, v in g_ih.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    continue
                sid_k, hour_k = key
                id_hour[(str(sid_k), _safe_int(hour_k))] = float(v)
            profile["id_hour"][c] = id_hour

            id_month: Dict[Tuple[str, int], float] = {}
            for key, v in g_im.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    continue
                sid_k, mon_k = key
                id_month[(str(sid_k), _safe_int(mon_k))] = float(v)
            profile["id_month"][c] = id_month

            global_month_hour: Dict[Tuple[int, int], float] = {}
            for key, v in g_mh.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    continue
                mon_k, hour_k = key
                global_month_hour[(_safe_int(mon_k), _safe_int(hour_k))] = float(v)
            profile["global_month_hour"][c] = global_month_hour

            global_dow_hour: Dict[Tuple[int, int], float] = {}
            for key, v in g_dh.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    continue
                dow_k, hour_k = key
                global_dow_hour[(_safe_int(dow_k), _safe_int(hour_k))] = float(v)
            profile["global_dow_hour"][c] = global_dow_hour

            global_hour: Dict[int, float] = {}
            for key, v in g_h.items():
                global_hour[_safe_int(key)] = float(v)
            profile["global_hour"][c] = global_hour

            global_month: Dict[int, float] = {}
            for key, v in g_m.items():
                global_month[_safe_int(key)] = float(v)
            profile["global_month"][c] = global_month

            id_global: Dict[str, float] = {}
            for key, v in g_i.items():
                id_global[str(key)] = float(v)
            profile["id_global"][c] = id_global
            profile["global"][c] = float(np.nanmean(t["v"].to_numpy(dtype=np.float64)))
        return profile

    def _seasonal_lookup(profile: Dict[str, Any], col: str, sid: str, dt: pd.Timestamp) -> Optional[float]:
        if pd.isna(dt):
            return None
        m = int(dt.month)
        d = int(dt.dayofweek)
        h = int(dt.hour)
        v = profile.get("id_month_hour", {}).get(col, {}).get((sid, m, h), None)
        if v is not None and np.isfinite(v):
            return float(v)
        v = profile.get("id_dow_hour", {}).get(col, {}).get((sid, d, h), None)
        if v is not None and np.isfinite(v):
            return float(v)
        v = profile.get("id_hour", {}).get(col, {}).get((sid, h), None)
        if v is not None and np.isfinite(v):
            return float(v)
        v = profile.get("id_month", {}).get(col, {}).get((sid, m), None)
        if v is not None and np.isfinite(v):
            return float(v)
        v = profile.get("global_month_hour", {}).get(col, {}).get((m, h), None)
        if v is not None and np.isfinite(v):
            return float(v)
        v = profile.get("global_dow_hour", {}).get(col, {}).get((d, h), None)
        if v is not None and np.isfinite(v):
            return float(v)
        v = profile.get("global_hour", {}).get(col, {}).get(h, None)
        if v is not None and np.isfinite(v):
            return float(v)
        v = profile.get("global_month", {}).get(col, {}).get(m, None)
        if v is not None and np.isfinite(v):
            return float(v)
        v = profile.get("id_global", {}).get(col, {}).get(sid, None)
        if v is not None and np.isfinite(v):
            return float(v)
        v = profile.get("global", {}).get(col, None)
        if v is not None and np.isfinite(v):
            return float(v)
        return None

    def _seasonal_fill(data: pd.DataFrame, cols: List[str], profile: Dict[str, Any]) -> pd.DataFrame:
        if data.empty:
            return data
        out_local = data.copy()
        out_local["Date"] = pd.to_datetime(out_local["Date"], errors="coerce")
        sid_series = out_local["ID"].astype(str)
        for c in cols:
            s = pd.to_numeric(out_local[c], errors="coerce")
            miss_idx = s[s.isna()].index.to_numpy(dtype=np.int64)
            if miss_idx.size == 0:
                continue
            for idx in miss_idx:
                sid = sid_series.iloc[int(idx)]
                raw_dt = out_local.at[idx, "Date"]
                if pd.isna(raw_dt):
                    continue
                dt_val = pd.to_datetime(str(raw_dt), errors="coerce")
                if pd.isna(dt_val):
                    continue
                dt = pd.Timestamp(dt_val)
                vv = _seasonal_lookup(profile, c, sid, dt)
                if vv is not None:
                    out_local.at[idx, c] = vv
        return out_local

    # Additional imputation stages.
    if method in {"spatial", "hybrid"}:
        out = _spatial_idw_fill(out, value_cols)
    if method in {"mice", "hybrid"}:
        out = _mice_fill(out, value_cols)

    # Remaining NaNs:
    # - hybrid: historical seasonal mean fallback (train-only profile on Val/Test)
    # - others: legacy train-mean fallback
    if train_means is None:
        train_means = {}
        for c in value_cols:
            mu = pd.to_numeric(out[c], errors="coerce").mean(skipna=True)
            train_means[c] = float(mu) if pd.notna(mu) else 0.0
        if method == "hybrid":
            train_means["__seasonal_profile__"] = _build_seasonal_profile(out, value_cols)

    if method == "hybrid":
        prof = train_means.get("__seasonal_profile__", None) if isinstance(train_means, dict) else None
        if not isinstance(prof, dict):
            prof = _build_seasonal_profile(out, value_cols)
        out = _seasonal_fill(out, value_cols, prof)

    fill_map = {c: float(train_means.get(c, 0.0)) for c in value_cols}
    out[value_cols] = out[value_cols].apply(pd.to_numeric, errors="coerce").fillna(value=fill_map)
    out[value_cols] = out[value_cols].fillna(0.0)

    if return_train_means:
        return out, train_means
    return out


def add_time_features(df: pd.DataFrame, cfg: Optional[Config] = None) -> pd.DataFrame:
    out = df.copy()

    time_col = cfg.TIME_COL if cfg is not None else ("Datetime" if "Datetime" in out.columns else "Date")
    id_col = cfg.NODE_ID_COL if cfg is not None else ("ID" if "ID" in out.columns else "id")

    if time_col not in out.columns:
        raise RuntimeError(f"add_time_features: missing time column {time_col}, current columns: {out.columns.tolist()}")
    if id_col not in out.columns:
        raise RuntimeError(f"add_time_features: missing station column {id_col}, current columns: {out.columns.tolist()}")

    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).copy()

    # Use DatetimeIndex to access month/hour to keep static analyzers happy.
    dt_index = pd.DatetimeIndex(out[time_col].values)
    out["month"] = dt_index.month.astype(np.int16)
    out["hour"] = dt_index.hour.astype(np.int16)

    out["month_sin"] = np.sin(2 * np.pi * out["month"].to_numpy(np.float32) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"].to_numpy(np.float32) / 12.0)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"].to_numpy(np.float32) / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"].to_numpy(np.float32) / 24.0)

    # Per-station monotonic index feature (safe and deterministic).
    out["t_index"] = out.groupby(id_col).cumcount().astype(np.int32)

    return out

def join_optional_meteo(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if not cfg.METEO_FILE or not os.path.exists(cfg.METEO_FILE): return df
    met = pd.read_csv(cfg.METEO_FILE)
    if "Date" in met.columns: met["Date"] = pd.to_datetime(met["Date"])
    if "ID" in met.columns:
        return df.merge(met, on=["ID", "Date"], how="left")
    return df

def fit_scaler(train_df: pd.DataFrame, cols: List[str]):
    """Fit a leakage-safe scaler on Train split only.

    Use mean/std standardization so features are approximately Gaussian-like.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[cols].values.astype(np.float32))
    return scaler


def normalize_df(df: pd.DataFrame, scaler: StandardScaler, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df[cols] = scaler.transform(df[cols].values.astype(np.float32))
    return df

def augment_series_per_lake(df: pd.DataFrame, cfg: Config, feature_cols: List[str]) -> pd.DataFrame:
    # Resolve time column explicitly to avoid NameError in augmentation.
    time_col = getattr(cfg, "TIME_COL", "Date")
    if time_col not in df.columns:
        time_col = "Datetime" if "Datetime" in df.columns else "Date"

    out_frames = [df]
    for lake_id, g in df.groupby("ID"):
        g = g.sort_values(time_col).reset_index(drop=True)
        base = g[feature_cols].values.astype(np.float32)
        for _ in range(cfg.AUG_TIMES):
            prev = np.vstack([base[0:1], base[:-1]])
            nxt  = np.vstack([base[1:], base[-1:]])
            beta1 = np.random.rand(*base.shape).astype(np.float32) * 0.15 + 0.05
            beta2 = np.random.rand(*base.shape).astype(np.float32) * 0.15 + 0.05
            noise = np.random.randn(*base.shape).astype(np.float32) * cfg.NOISE_SCALE * (np.abs(base) + 1e-6)
            aug = base + beta1 * (prev - base) + beta2 * (nxt - base) + noise
            aug_df = g.copy()
            aug_df[feature_cols] = aug
            aug_df["__aug_tag__"] = 1
            out_frames.append(aug_df)
    return pd.concat(out_frames, ignore_index=True)

def build_windows_grouped(
    df: pd.DataFrame,
    cfg: Config,
    input_cols: List[str],
    target_cols: List[str],
    *,
    return_meta: bool = False,
    station_vocab: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Build sliding windows per station (grouped by ID).

    Non-graph baselines use per-station 3D supervision:
      X: [S, SEQ_LEN, F]
      y: [S, PRED_LEN, D]

    If return_meta=True, also returns:
      - station_idx: [S] (int), mapping each sample to a station.
      - station_names: list[str], the station vocabulary (index -> station ID).
      - pred_times: np.datetime64 array [S], timestamp of the first prediction horizon (t+1).

    NOTE (typing / editor friendliness):
      Historically this function returned 2 values when return_meta=False and 5 values
      when return_meta=True. To keep editor typing stable, it now always returns 5.
    """
    X_list, y_list, sid_list, t_list = [], [], [], []
    # Optional shared mapping for consistent station indices across splits
    id2idx = None
    uniq = None
    if station_vocab is not None:
        uniq = [str(s) for s in station_vocab]
        id2idx = {s: i for i, s in enumerate(uniq)}
    # Try to infer time column for plotting
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    for lake_id, g in df.groupby("ID"):
        g = g.sort_values(time_col).reset_index(drop=True)
        vals_input = g[input_cols].values.astype(np.float32)
        vals_target = g[target_cols].values.astype(np.float32)
        L = len(vals_input)
        if L < cfg.SEQ_LEN + cfg.PRED_LEN: continue
        for i in range(0, L - cfg.SEQ_LEN - cfg.PRED_LEN + 1):
            x = vals_input[i : i + cfg.SEQ_LEN]
            y = vals_target[i + cfg.SEQ_LEN : i + cfg.SEQ_LEN + cfg.PRED_LEN]
            X_list.append(x)
            y_list.append(y)
            if return_meta:
                sid_list.append(str(lake_id))
                # Timestamp aligned to the first prediction step
                t_list.append(pd.to_datetime(g[time_col].iloc[i + int(cfg.SEQ_LEN)]))
    X = np.stack(X_list) if X_list else np.empty((0, int(cfg.SEQ_LEN), len(input_cols)), dtype=np.float32)
    y = np.stack(y_list) if y_list else np.empty((0, int(getattr(cfg, "PRED_LEN", 1)), len(target_cols)), dtype=np.float32)

    # Always return 5 values (see NOTE above)
    if not return_meta:
        sid_idx = np.empty((X.shape[0],), dtype=np.int64)
        station_names: List[str] = [str(s) for s in station_vocab] if station_vocab is not None else []
        pred_times = np.empty((X.shape[0],), dtype="datetime64[ns]")
        return X, y, sid_idx, station_names, pred_times

    # Map station IDs -> contiguous integer indices for safe TensorDataset.
    station_names: List[str]
    if id2idx is None:
        # Use sorted unique IDs for deterministic mapping.
        uniq_list = sorted(set(str(s) for s in sid_list))
        id2idx = {s: i for i, s in enumerate(uniq_list)}
        station_names = uniq_list
    else:
        # If station_vocab is given, it has already been used to build id2idx above.
        station_names = [str(s) for s in station_vocab] if station_vocab is not None else sorted([str(k) for k in id2idx.keys()])

    sid_idx = np.asarray([id2idx[str(s)] for s in sid_list], dtype=np.int64)
    pred_times = np.asarray(t_list, dtype="datetime64[ns]") if t_list else np.empty((0,), dtype="datetime64[ns]")
    return X, y, sid_idx, station_names, pred_times

# ==========================================
# 3. Data preprocessing helpers for graph/spatiotemporal models
# ==========================================
def haversine_km(lon1, lat1, lon2, lat2):
    # Earth radius (km)
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def build_knn_adj(lons, lats, k=6, sigma_km=20.0):
    """Return normalized adjacency A_hat: [N,N] with self-loop, D^-1/2 A D^-1/2."""
    N = len(lons)
    # Distance matrix
    dist = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            dist[i, j] = haversine_km(lons[i], lats[i], lons[j], lats[j])

    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        idx = np.argsort(dist[i])[1:k+1]  # skip itself
        for j in idx:
            w = math.exp(-float(dist[i, j]) / float(sigma_km))
            A[i, j] = w
            A[j, i] = max(A[j, i], w)  # undirected graph (keep the larger weight)

    # self-loop
    for i in range(N):
        A[i, i] = 1.0

    # normalize: D^-1/2 A D^-1/2
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_hat, dtype=torch.float32)

def build_graph_windows_from_df(
    df: pd.DataFrame,
    cfg: Config,
    input_cols: List[str],
    target_cols: List[str],
    *,
    node_ids: Optional[List[str]] = None,
    adj_hat: Optional[np.ndarray] = None,
    train_fill_means: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Build ST-GCN/GNN windows: inputs [S, T, N, F], targets [S, N, D].
    This version explicitly avoids fillna(0)/nan_to_num(0) shortcuts.
    It first aligns all stations to a shared timeline, then performs time-based
    interpolation and boundary fill to avoid large leading all-zero segments.
    Returns: X [S, SEQ_LEN, N, Fin], y [S, N, Dout], times [S], adj_hat [N,N], node_ids.
    """


    # -------------------------
    # Internal utilities: adjacency construction and normalization
    def _normalize_adj(adj: np.ndarray) -> np.ndarray:
        'D^{-1/2}(A+I)D^{-1/2}'
        A = adj.astype(np.float32, copy=True)
        A = A + np.eye(A.shape[0], dtype=np.float32)
        deg = A.sum(axis=1)
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = np.diag(deg_inv_sqrt.astype(np.float32))
        return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)

    def _build_knn_adjacency(coords: np.ndarray, k: int, sigma_km: float = 20.0) -> np.ndarray:
        """KNN adjacency on geographic coordinates using Haversine distance.

        Args:
          coords: [N,2] array (lon,lat) in degrees
          k: number of neighbors per node
          sigma_km: RBF sigma in kilometers (exp(-d^2/(2*sigma_km^2)))

        Returns:
          A: [N,N] symmetric adjacency (float32)
        """
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords should be [N,2], got {coords.shape}")

        # Vectorized haversine distance matrix
        lon = np.deg2rad(coords[:, 0]).astype(np.float64)
        lat = np.deg2rad(coords[:, 1]).astype(np.float64)
        dlon = lon[:, None] - lon[None, :]
        dlat = lat[:, None] - lat[None, :]
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        dist_km = (6371.0 * c).astype(np.float32)  # Earth radius
        np.fill_diagonal(dist_km, np.inf)

        Nn = coords.shape[0]
        A = np.zeros((Nn, Nn), dtype=np.float32)
        kk = int(max(1, min(int(k), Nn - 1)))
        nn_idx = np.argsort(dist_km, axis=1)[:, :kk]

        use_weight = (sigma_km is not None) and float(sigma_km) > 0
        if use_weight:
            denom = 2.0 * (float(sigma_km) ** 2)

        for i in range(Nn):
            js = nn_idx[i]
            if use_weight:
                w = np.exp(-(dist_km[i, js] ** 2) / denom).astype(np.float32)
                A[i, js] = w
            else:
                A[i, js] = 1.0

        # symmetric
        A = np.maximum(A, A.T)
        return A

    # -------------------------
    # 0) Validate columns and required fields
    if "ID" not in df.columns:
        raise ValueError("df missing ID column")

    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    if time_col not in df.columns:
        raise ValueError("df missing Date/Datetime column")

    dff = df.copy()
    dff[time_col] = pd.to_datetime(dff[time_col], errors="coerce")
    dff = dff.dropna(subset=[time_col, "ID"])

    # -------------------------
    # 1) Determine node order
    # -------------------------
    if node_ids is None:
        node_ids = sorted(dff["ID"].dropna().unique().tolist())

    N = len(node_ids)
    if N == 0:
        raise RuntimeError("No station IDs were found in the dataframe")

    # -------------------------
    # 2) Build graph (if not provided)
    if adj_hat is None:
        lon_col = getattr(cfg, "LON_COL", "lon")
        lat_col = getattr(cfg, "LAT_COL", "lat")

        # Normalize column names for case/whitespace compatibility
        col_map = {c.strip().lower(): c for c in dff.columns}
        lon_col_real = col_map.get(lon_col.strip().lower(), None)
        lat_col_real = col_map.get(lat_col.strip().lower(), None)
        if lon_col_real is None or lat_col_real is None:
            raise RuntimeError(f"Missing coordinate columns; cannot build graph. Required: {lon_col}/{lat_col}")

        # Pick one valid coordinate pair per station (prefer non-null)
        coords = []
        missing = []
        for nid in node_ids:
            g = dff.loc[dff["ID"] == nid, [lon_col_real, lat_col_real]]
            if g.empty:
                missing.append(nid)
                coords.append((np.nan, np.nan))
                continue
            lon_s = pd.to_numeric(g[lon_col_real], errors="coerce")
            lat_s = pd.to_numeric(g[lat_col_real], errors="coerce")
            # take the first non-null coordinate value per station
            lon_v = lon_s.dropna().iloc[0] if lon_s.notna().any() else np.nan
            lat_v = lat_s.dropna().iloc[0] if lat_s.notna().any() else np.nan
            coords.append((float(lon_v) if pd.notna(lon_v) else np.nan,
                           float(lat_v) if pd.notna(lat_v) else np.nan))

        coords_arr = np.asarray(coords, dtype=float)
        if np.isnan(coords_arr).any():
            bad = [node_ids[i] for i in range(N) if np.isnan(coords_arr[i]).any()]
            raise RuntimeError(f"These stations have NaN coordinates and cannot build graph: {bad}")

        # Build KNN graph
        adj = _build_knn_adjacency(
            coords_arr,
            k=int(getattr(cfg, "KNN_K", 6)),
            sigma_km=float(getattr(cfg, "KNN_SIGMA_KM", getattr(cfg, "KNN_SIGMA", 20.0))),
        )
        adj_hat = _normalize_adj(adj)

    # -------------------------
    # 3) Build a shared timeline (prefer common overlap interval by default)
    # Per-station time ranges
    mins, maxs = [], []
    for nid in node_ids:
        g = dff.loc[dff["ID"] == nid, time_col]
        if g.empty:
            continue
        mins.append(g.min())
        maxs.append(g.max())

    if not mins or not maxs:
        raise RuntimeError("Time column is empty; cannot build timeline")

    # Common overlap interval: start=max(min_i), end=min(max_i)
    start = max(mins)
    end = min(maxs)

    freq = getattr(cfg, "RESAMPLE_FREQ", "D")
    if pd.isna(start) or pd.isna(end) or start >= end:
        # Fallback: use global min/max interval
        start = min(mins)
        end = max(maxs)

    full_index = pd.date_range(start=start, end=end, freq=freq)
    if len(full_index) < (cfg.SEQ_LEN + cfg.PRED_LEN + 1):
        # If overlap is too short, fallback to union timeline
        all_times = pd.to_datetime(dff[time_col].unique())
        all_times = pd.DatetimeIndex(sorted(set(all_times)))
        full_index = all_times

    # -------------------------
    # 4) Rebuild complete timeline per station + impute
    def _time_features_from_index(idx: pd.DatetimeIndex) -> pd.DataFrame:
        month = idx.month
        hour = idx.hour
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        # Step index within the current segment (0..1). Using a simple normalized
        # step counter avoids relying on pandas-internal int reprs (asi8/view),
        # and is stable across platforms.
        t_index = np.arange(len(idx), dtype=np.float32)
        if len(idx) > 1:
            t_index = t_index / float(len(idx) - 1)
        return pd.DataFrame(
            {
                "month_sin": month_sin,
                "month_cos": month_cos,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "t_index": t_index,
            },
            index=idx,
        )

    # Columns to impute: inputs + targets (deduplicated)
    cols_need = list(dict.fromkeys(list(input_cols) + list(target_cols)))

    # Time feature columns (if included in input_cols, overwrite from parsed time)
    time_feat_cols = [c for c in ["month_sin", "month_cos", "hour_sin", "hour_cos", "t_index"] if c in cols_need]

    X_nodes = []  # per station: [T, Fin]
    Y_nodes = []  # per station: [T, Dout]

    # Precompute time features
    tf = _time_features_from_index(pd.DatetimeIndex(full_index))

    for nid in node_ids:
        g = dff.loc[dff["ID"] == nid].copy()
        g = g.sort_values(time_col)
        g = g.set_index(time_col)
        # Some stations may still contain duplicated timestamps. Collapse them
        # before any reindex call to prevent reindex duplicate-label errors.
        if g.index.has_duplicates:
            g = g.groupby(level=0).mean(numeric_only=True).sort_index()

        # Keep only required columns; create missing cols as NaN first.
        for c in cols_need:
            if c not in g.columns:
                g[c] = np.nan

        g = g[cols_need]

        # Align to shared full_index
        g = g.reindex(full_index)

        # Convert value columns to numeric
        for c in cols_need:
            if c in time_feat_cols:
                continue
            g[c] = pd.to_numeric(g[c], errors="coerce")

        # Overwrite time feature columns
        for c in time_feat_cols:
            g[c] = tf[c].values

        # Reindexing to the shared `full_index` can (re-)introduce gaps even if
        # upstream preprocessing already imputed within each split. For graph
        # windows we must ensure *every* node has a complete time series.
        #
        # Leakage note:
        # - This imputation happens *within the current split dataframe* (train/val/test)
        #   and never crosses split boundaries.
        # - We primarily use time interpolation + boundary fill; remaining NaNs
        #   are filled with TRAIN-only means (provided by caller).
        check_cols = [c for c in cols_need if c not in time_feat_cols]
        if check_cols and g[check_cols].isna().any().any():
            # 1) Time interpolation on shared index
            try:
                g[check_cols] = g[check_cols].interpolate(method="time", limit_direction="both")
            except Exception:
                # Fallback: linear interpolation
                g[check_cols] = g[check_cols].interpolate(method="linear", limit_direction="both")
            # 2) Boundary fill
            g[check_cols] = g[check_cols].ffill().bfill()
            # 3) Last-resort fill using TRAIN-only means (leakage-safe)
            if train_fill_means is not None:
                fill_map = {c: float(train_fill_means.get(c, 0.0)) for c in check_cols}
                g[check_cols] = g[check_cols].fillna(value=fill_map)
            # 4) Still NaN -> data quality issue.
            if g[check_cols].isna().any().any():
                bad_cols = [c for c in check_cols if g[c].isna().any()]
                # Fill with per-split column means (if possible), then 0.0.
                for c in bad_cols:
                    mu = pd.to_numeric(g[c], errors="coerce").mean(skipna=True)
                    g[c] = g[c].fillna(float(mu) if pd.notna(mu) else 0.0)
                # Still NaN (entire column missing) -> 0.0
                g[bad_cols] = g[bad_cols].fillna(0.0)
                _print_once(
                    f"graph_nan_fallback:{nid}:{','.join(bad_cols)}",
                    f"[WARN] graph windows had full-column NaNs; fallback fill applied (mean/0.0). "
                    f"station={nid}, cols={bad_cols}.",
                )

        # Append complete station series inside the node loop
        X_nodes.append(g[input_cols].values.astype(np.float32))
        Y_nodes.append(g[target_cols].values.astype(np.float32))

    # [N, T, F] -> [T, N, F]
    X_all = np.stack(X_nodes, axis=1)
    Y_all = np.stack(Y_nodes, axis=1)

    T = X_all.shape[0]
    seq_len = int(cfg.SEQ_LEN)
    pred_len = int(getattr(cfg, "PRED_LEN", 1))

    if T <= (seq_len + pred_len):
        raise RuntimeError(f"time length too short: T={T}, SEQ_LEN={seq_len}, PRED_LEN={pred_len}")

    # -------------------------
    # 5) Build sliding windows
    # -------------------------
    X, Y, times = [], [], []
    # Prediction timestamp starts at seq_len
    for t in range(seq_len, T - pred_len + 1):
        X.append(X_all[t - seq_len : t])       # [SEQ, N, Fin]
        Y.append(Y_all[t : t + pred_len])       # [PRED, N, Dout]
        times.append(full_index[t])

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    times = np.asarray(times)

    # Final sanity check: forbid NaN/Inf.
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise RuntimeError(
            "NaN/Inf remains after interpolation/filling; please check raw data for full-column missing values."
        )

    return X, Y, times, np.asarray(adj_hat, dtype=float), node_ids
