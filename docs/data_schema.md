# Data Schema

The training pipeline expects a station-time CSV table. The default file name is:

```text
processed_taihu_data_with_coords.csv
```

Required columns:

| Column | Type | Description |
| --- | --- | --- |
| `ID` | string | Station identifier. |
| `Date` | datetime-like | Observation timestamp. |
| `lon` | float | Station longitude. |
| `lat` | float | Station latitude. |
| `Temp` | float | Water temperature. |
| `DO` | float | Dissolved oxygen. |
| `TN` | float | Total nitrogen. |
| `Cond` | float | Conductivity. |
| `pH` | float | pH value. |
| `CODMn` | float | Permanganate index or an equivalent organic-pollution indicator. |
| `TP` | float | Total phosphorus. |
| `Turb` | float | Turbidity. |

Default input features are configured in `config/config_taihu.py` as `FEATURE_COLS`. Default prediction targets are configured as `TARGET_FEATURES`.

For compatibility with historical Taihu exports, the loader also accepts `PI`
as an alias of `CODMn` and `Tur` as an alias of `Turb`. The canonical project
and figure labels remain `CODMn` and `Turb`.
