"""Export compact JSON files for the website dashboard.

Produces, under export.output_dir (default public/data):
  meta.json                          variables, locations, sources, generation time
  latest.json                        latest daily value per variable/location, with anomaly
  daily/<variable>__<location>.json  merged daily series, record statistics and day-of-year climatology
  dust_events.json                   Saharan dust episodes and what chlorophyll/light did around them
  marine_heatwaves.json              marine heatwave events and today's status for each sea area (SST)
  marine_heatwaves__<variable>.json  the same for temperature at depth
  forecast/<variable>__<location>.json  daily forecast for the days after today
  upwelling.json                     upwelling index events and status
  blooms.json                        chlorophyll bloom events (with possible triggers) and status
  rain.json                          heavy rain and first-flush wet spells (run-off), and the current dry spell
  tides.json                         tide predictions, high and low waters, recent observed vs predicted
  wildlife.json                      NEMO sightings: counts, species calendar, gelatinous and invasive watches
  <private_dir>/nemo_matched.csv     every sighting with the sea conditions at the time (not published)

It also copies the dashboard page (the site/ folder) next to the data, so GitHub Pages
publishes page and data together.
"""
from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import derived as dv
from . import strandings as sl
from . import wildlife as wl
from .db import Database
from .stats import circular_mean_deg

SUM_VARS = {"precip", "precip_1h", "precip_3h", "precip_6h", "precip_12h", "precip_24h"}
MAX_VARS = {"wind_gust"}

# PAR from broadband shortwave radiation:
#   PAR fraction of incoming shortwave ~0.48, and 4.57 umol photons per joule of PAR.
#   daily mean W m-2 * 86400 s * 0.48 * 4.57e-6 mol/J  ->  einstein (mol photons) m-2 day-1
PAR_FRACTION = 0.48
PHOTONS_PER_JOULE = 4.57e-6
PAR_FACTOR = 86400 * PAR_FRACTION * PHOTONS_PER_JOULE   # ~0.1895


def par_from_shortwave(daily_mean_sw: pd.Series) -> pd.Series:
    """Daily PAR (E m-2 d-1) from daily mean shortwave radiation (W m-2)."""
    return daily_mean_sw * PAR_FACTOR


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def daily_aggregate(series: list[tuple], variable: str) -> pd.Series:
    """[(datetime, value), ...] -> pd.Series indexed by date."""
    if not series:
        return pd.Series(dtype="float64")
    df = pd.DataFrame(series, columns=["t", "v"])
    df["t"] = pd.to_datetime(df["t"])
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    g = df.groupby(df["t"].dt.normalize())["v"]
    if variable.endswith("_dir"):
        out = g.apply(lambda s: circular_mean_deg(s.values))
    elif variable in SUM_VARS:
        # only report a daily total when (nearly) every hour is present
        counts = g.count()
        out = g.sum().where(counts >= 20)
    elif variable in MAX_VARS:
        out = g.max()
    else:
        out = g.mean()
    return out.astype("float64").dropna()


def merge_by_priority(per_source: dict[str, pd.Series], priority: list[str]) -> pd.DataFrame:
    """Pick, for each day, the value from the highest-priority source that has one."""
    order = [s for s in priority if s in per_source] + [s for s in per_source if s not in priority]
    frames = []
    taken = pd.DatetimeIndex([])
    for src in order:
        s = per_source[src]
        new = s[~s.index.isin(taken)]
        if not new.empty:
            frames.append(pd.DataFrame({"value": new.values, "source": src}, index=new.index))
            taken = taken.union(new.index)
    if not frames:
        return pd.DataFrame(columns=["value", "source"])
    return pd.concat(frames).sort_index()


def record_stats(merged: pd.DataFrame) -> dict | None:
    """Whole-record statistics: the horizontal reference lines on the full-record chart."""
    if merged.empty:
        return None
    v = merged["value"].astype(float)
    return {
        "n": int(v.size),
        "mean": _r(v.mean()),
        "sd": _r(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": _r(v.min()), "min_date": v.idxmin().strftime("%Y-%m-%d"),
        "max": _r(v.max()), "max_date": v.idxmax().strftime("%Y-%m-%d"),
        "first": merged.index.min().strftime("%Y-%m-%d"),
        "last": merged.index.max().strftime("%Y-%m-%d"),
    }


def climatology(merged: pd.DataFrame, min_years=3, half_window=7, circular=False):
    """Day-of-year statistics over all years, using a centred 15-day window wrapped round the year.

    Returns parallel arrays indexed by day of year 1..365: mean, sd, min, max, p10, p90, n.
    """
    if merged.empty:
        return None
    years = merged.index.year.nunique()
    if years < min_years:
        return None
    doy = np.minimum(merged.index.dayofyear.values, 365)
    vals = merged["value"].astype(float).values
    out = {k: [] for k in ("mean", "sd", "min", "max", "p10", "p90", "n")}
    for d in range(1, 366):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, 365 - dist)
        sel = vals[dist <= half_window]
        sel = sel[np.isfinite(sel)]
        if sel.size < 5:
            for k in out:
                out[k].append(None if k != "n" else int(sel.size))
            continue
        if circular:
            out["mean"].append(_r(circular_mean_deg(sel), 1))
            for k in ("sd", "min", "max", "p10", "p90"):
                out[k].append(None)
        else:
            out["mean"].append(_r(sel.mean()))
            out["sd"].append(_r(sel.std(ddof=1)))
            out["min"].append(_r(sel.min()))
            out["max"].append(_r(sel.max()))
            out["p10"].append(_r(np.quantile(sel, 0.1)))
            out["p90"].append(_r(np.quantile(sel, 0.9)))
        out["n"].append(int(sel.size))
    return {"years": int(years), "first_year": int(merged.index.year.min()),
            "last_year": int(merged.index.year.max()), "window_days": 2 * half_window + 1, **out}


def dust_events(config: dict, merged_by_key: dict, log=print) -> dict:
    """Group days above the dust thresholds into episodes and compare response variables
    in the N days before an episode with the N days after it."""
    cfg = config.get("export", {}).get("dust", {})
    indicators = cfg.get("indicators", [])
    if not indicators:
        return {"episodes": [], "indicators": []}
    flagged: dict[pd.Timestamp, list] = defaultdict(list)
    used = []
    for ind in indicators:
        key = (ind["variable"], ind["location"])
        m = merged_by_key.get(key)
        if m is None or m.empty:
            continue
        s = m["value"].astype(float)
        mask = s > float(ind["threshold"])
        if ind.get("with"):
            w = ind["with"]
            other = merged_by_key.get((w["variable"], ind["location"]))
            if other is None:
                continue
            o = other["value"].astype(float).reindex(s.index)
            mask &= o < float(w["below"])
        for day, val in s[mask].items():
            flagged[day].append({"indicator": ind["variable"], "value": _r(val, 3)})
        used.append({**ind, "days_flagged": int(mask.sum())})

    days = sorted(flagged)
    episodes = []
    for day in days:
        if episodes and (day - episodes[-1]["_end"]).days <= 1:
            episodes[-1]["_end"] = day
            episodes[-1]["_days"].append(day)
        else:
            episodes.append({"_start": day, "_end": day, "_days": [day]})

    window = int(cfg.get("response_window_days", 7))
    responses = cfg.get("response", [])
    out = []
    for ep in episodes:
        start, end = ep["_start"], ep["_end"]
        peak = {}
        for d in ep["_days"]:
            for rec in flagged[d]:
                if rec["value"] is not None and rec["value"] > peak.get(rec["indicator"], -np.inf):
                    peak[rec["indicator"]] = rec["value"]
        resp = []
        for r in responses:
            m = merged_by_key.get((r["variable"], r["location"]))
            if m is None or m.empty:
                continue
            s = m["value"].astype(float)
            before = s[(s.index >= start - pd.Timedelta(days=window)) & (s.index < start)]
            after = s[(s.index > end) & (s.index <= end + pd.Timedelta(days=window))]
            b = before.mean() if before.size else np.nan
            a = after.mean() if after.size else np.nan
            resp.append({"variable": r["variable"], "location": r["location"],
                         "before_mean": _r(b), "after_mean": _r(a),
                         "n_before": int(before.size), "n_after": int(after.size),
                         "change_pct": _r((a - b) / b * 100, 1) if np.isfinite(a) and np.isfinite(b) and b else None})
        out.append({"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d"),
                    "days": len(ep["_days"]), "peak": peak, "response": resp})
    return {"indicators": used, "response_window_days": window, "episodes": out}


MHW_CATEGORIES = {1: "Moderate", 2: "Strong", 3: "Severe", 4: "Extreme"}


def _leap_doy(index: pd.DatetimeIndex) -> np.ndarray:
    """Day of year on a 366-day calendar, so 1 March is day 61 in every year."""
    return np.array([pd.Timestamp(2000, d.month, d.day).dayofyear for d in index])


def _circular_smooth(a: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    padded = np.concatenate([a[-half:], a, a[:half]])
    kernel = np.ones(width) / width
    return np.convolve(padded, kernel, mode="valid")


def mhw_climatology(sst: pd.Series, baseline=(1991, 2020), half_window=5, smooth_days=31, min_years=10):
    """Seasonal mean and 90th percentile threshold for each day of a 366-day year (Hobday et al. 2016)."""
    s = sst.dropna()
    y0, y1 = baseline
    base = s[(s.index.year >= y0) & (s.index.year <= y1)]
    if base.index.year.nunique() < min_years:
        base = s                                            # record too short for the fixed baseline
    y0, y1 = int(base.index.year.min()), int(base.index.year.max())
    if base.index.year.nunique() < 3:
        return None
    doy = _leap_doy(base.index)
    vals = base.values.astype(float)
    seas = np.full(366, np.nan)
    thresh = np.full(366, np.nan)
    for d in range(1, 367):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, 366 - dist)
        sel = vals[dist <= half_window]
        if sel.size >= 10:
            seas[d - 1] = sel.mean()
            thresh[d - 1] = np.quantile(sel, 0.9)
    if np.isnan(seas).any():                                # fill any empty days before smoothing
        idx = np.arange(366)
        ok = ~np.isnan(seas)
        seas = np.interp(idx, idx[ok], seas[ok], period=366)
        thresh = np.interp(idx, idx[ok], thresh[ok], period=366)
    return {"baseline": [int(y0), int(y1)], "seas": _circular_smooth(seas, smooth_days),
            "thresh": _circular_smooth(thresh, smooth_days)}


def marine_heatwaves(sst: pd.Series, baseline=(1991, 2020), min_days=5, max_gap_days=2):
    """Detect marine heatwaves in a daily SST series.

    A heatwave is at least `min_days` in a row above the seasonal 90th percentile; events separated by
    `max_gap_days` or fewer are joined. Category = how many times the threshold's distance above the
    seasonal mean the peak reached (1 Moderate, 2 Strong, 3 Severe, 4+ Extreme).
    """
    clim = mhw_climatology(sst, baseline)
    if clim is None or sst.dropna().empty:
        return None
    s = sst.dropna().astype(float)
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full)
    doy = _leap_doy(full) - 1
    seas, thresh = clim["seas"][doy], clim["thresh"][doy]
    above = (s.values > thresh) & np.isfinite(s.values)

    runs, start = [], None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        if (not flag or i == len(above) - 1) and start is not None:
            end = i if flag else i - 1
            if end - start + 1 >= min_days:
                runs.append([start, end])
            start = None
    joined = []
    for r in runs:
        if joined and r[0] - joined[-1][1] - 1 <= max_gap_days:
            joined[-1][1] = r[1]
        else:
            joined.append(r)

    events = []
    for a, b in joined:
        v = s.values[a:b + 1]
        inten = v - seas[a:b + 1]
        ratio = inten / np.maximum(thresh[a:b + 1] - seas[a:b + 1], 1e-6)
        k = int(np.nanargmax(inten))
        cat = int(min(4, max(1, np.floor(np.nanmax(ratio)))))
        events.append({
            "start": full[a].strftime("%Y-%m-%d"), "end": full[b].strftime("%Y-%m-%d"), "days": int(b - a + 1),
            "peak_date": full[a + k].strftime("%Y-%m-%d"), "max_intensity": _r(np.nanmax(inten), 2),
            "mean_intensity": _r(np.nanmean(inten), 2), "cumulative_intensity": _r(np.nansum(inten), 1),
            "category": cat, "category_name": MHW_CATEGORIES[cat],
        })

    last = len(full) - 1
    status = {"date": full[last].strftime("%Y-%m-%d"), "sst": _r(s.values[last], 2),
              "seas": _r(seas[last], 2), "thresh": _r(thresh[last], 2), "state": "none"}
    if events and events[-1]["end"] == status["date"]:
        ev = events[-1]
        ratio = (s.values[last] - seas[last]) / max(thresh[last] - seas[last], 1e-6)
        status.update(state="heatwave", since=ev["start"], days=ev["days"],
                      category=int(min(4, max(1, np.floor(ratio)))) if np.isfinite(ratio) and ratio >= 1 else 1,
                      intensity=_r(s.values[last] - seas[last], 2))
        status["category_name"] = MHW_CATEGORIES[status["category"]]
    elif above[last]:
        n = 0
        while last - n >= 0 and above[last - n]:
            n += 1
        status.update(state="warm", days=int(n), intensity=_r(s.values[last] - seas[last], 2))
    return {"baseline": clim["baseline"], "first": full[0].strftime("%Y-%m-%d"), "min_days": min_days, "max_gap_days": max_gap_days,
            "seas": [_r(x, 3) for x in clim["seas"]], "thresh": [_r(x, 3) for x in clim["thresh"]],
            "events": events, "status": status}


DERIVED_LABELS = {"derived_light": "Derived from satellite PAR and KD490",
                  "derived_rain": "Summed from daily rainfall"}


def source_label(config: dict, code: str) -> str:
    return config.get("sources", {}).get(code, {}).get("label") or DERIVED_LABELS.get(code) or code


def _write_series(path, config, variable, location, merged, clim, rec, options=None):
    srcs = list(dict.fromkeys(merged["source"]))
    idx = {s: i for i, s in enumerate(srcs)}
    vmeta = config["variables"].get(variable, {})
    payload = {
        "variable": variable,
        "location": location,
        "unit": vmeta.get("unit"),
        "name": vmeta.get("name", variable),
        "statistic": vmeta.get("daily_statistic", "mean"),
        "sources": srcs,
        "source_labels": [source_label(config, s) for s in srcs],
        "source_options": options or [],
        "record": rec,
        "climatology": clim,
        "data": [[d.strftime("%Y-%m-%d"), round(float(v), 4), idx[s]]
                 for d, v, s in zip(merged.index, merged["value"], merged["source"])],
    }
    path.write_text(json.dumps(payload, separators=(",", ":")))


def load_strandings(cfg: dict, root: Path, log=print):
    """TNP's strandings log, from the private link if there is one, else a committed copy."""
    if not cfg.get("enabled"):
        return None
    from .sources.base import Source
    data = None
    if cfg.get("url_env") and os.environ.get(cfg["url_env"]):
        r = Source.http_get(os.environ[cfg["url_env"]], timeout=120)
        if r.status_code >= 400:
            log(f"Strandings: link returned {r.status_code}, falling back to the committed copy")
        else:
            data = r.content
    if data is None and cfg.get("file") and (root / cfg["file"]).exists():
        data = (root / cfg["file"]).read_bytes()
    if data is None:
        log(f"Strandings: no log found (set {cfg.get('url_env')} or add {cfg.get('file')})")
        return None
    # A hand-kept spreadsheet can arrive in any state, and a library can be missing from an
    # install. Neither is a reason to lose the sea data, so nothing here is allowed to be fatal.
    try:
        records = sl.clean(sl.read_log(data, cfg.get("sheet")), cfg)
    except Exception as e:
        log(f"Strandings: could not read the log ({e})")
        return None
    log(f"Strandings: {len(records)} records from the log")
    return records


def export_wildlife(config: dict, wcfg: dict, merged_by_key: dict, tide_fit, out_dir: Path, log=print,
                    triggers: dict | None = None, today=None):
    """Read NEMO records, attach the conditions, and write the public summary and the private file."""
    import yaml

    root = Path(__file__).resolve().parents[1]
    species_cfg = yaml.safe_load((root / wcfg.get("species_file", "species.yaml")).read_text())
    src = wcfg.get("source", {})
    fmt = src.get("time_format", "%d/%m/%Y %H:%M")
    if src.get("url_env") and os.environ.get(src["url_env"]):
        # preferred: the export lives behind a private link held in GitHub secrets, so no personal
        # data is ever committed to this repository
        from .sources.base import Source
        r = Source.http_get(os.environ[src["url_env"]], timeout=120)
        if r.status_code >= 400:
            raise FileNotFoundError(f"NEMO export link returned {r.status_code}")
        raw = wl.read_records(r.text, fmt)
    elif src.get("csv") and (root / src["csv"]).exists():
        raw = wl.read_records((root / src["csv"]).read_text(encoding="utf-8-sig"), fmt)
    else:
        raise FileNotFoundError(f"no NEMO export found (set {src.get('url_env', 'NEMO_CSV_URL')} "
                                f"or add {src.get('csv')})")

    records = wl.clean(raw, config, bbox=wcfg.get("bbox"))
    records = wl.tag_species(records, species_cfg)
    matched = wl.attach_conditions(records, merged_by_key, wcfg.get("conditions", []), tide_fit)

    scfg = wcfg.get("strandings") or {}
    log_records = load_strandings(scfg, root, log=log)
    strandings_summary = (sl.summary(log_records, int(scfg.get("recent_days", 365)))
                          if log_records is not None else None)

    private = root / wcfg.get("private_dir", "private")
    private.mkdir(parents=True, exist_ok=True)
    # Workflow files can be downloaded by anyone on a public repository, so the written notes stay out
    # unless they are explicitly asked for. Record ids are kept, so a note can be read back in NEMO.
    keep_notes = bool(wcfg.get("include_notes"))
    matched.drop(columns=[] if keep_notes else ["notes"]).to_csv(private / "nemo_matched.csv", index=False)
    cand = wl.stranding_candidates(records)
    cand.drop(columns=[] if keep_notes else ["notes"]).to_csv(private / "stranding_candidates.csv", index=False)

    # ---- public summary: counts, calendar, watches; no names, no notes, positions on a grid ----
    day = pd.to_datetime(records["local_date"])
    groups = sorted(records["group"].unique())
    calendar = {g: [int(((records["group"] == g) & (day.dt.month == m)).sum()) for m in range(1, 13)]
                for g in groups}
    gel = records["gelatinous"] != ""
    gel_index = wl.daily_index(records, gel)
    gel_events, gel_status = wl.bloom_events(gel_index, int(wcfg.get("bloom_min_records", 3)),
                                             float(wcfg.get("bloom_min_share", 0.4)))
    drift_index = wl.daily_index(records, records["gelatinous"] == "drifter")
    recent_days = int(wcfg.get("recent_days", 30))
    # Counted back from today, not from the newest record in the file. Anchoring on the newest
    # record means a single date typed wrong puts the window in the future, where there is nothing
    # to find, and the section then reports no sightings while the file is full of them.
    cutoff = pd.Timestamp(today or datetime.now(timezone.utc).date()) - pd.Timedelta(days=recent_days)
    recent = records[day > cutoff]
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "records": int(len(records)),
        "first": records["local_date"].min(), "last": records["local_date"].max(),
        "species_count": int(records["species"].nunique()),
        "by_group": {g: int(n) for g, n in records["group"].value_counts().items()},
        "by_year": {str(y): int(n) for y, n in day.dt.year.value_counts().sort_index().items()},
        "calendar": calendar,
        "recent": {"days": recent_days, "records": int(len(recent)),
                   "by_group": {g: int(n) for g, n in recent["group"].value_counts().items()},
                   # a record outside every area box has no area: null, not the NaN that json writes
                   # out unquoted and every browser then refuses to parse
                   "list": [{"date": r.local_date, "group": r.group, "species": r.species,
                             "area": r.area if isinstance(r.area, str) else None,
                             "cell": r.cell_1km if not r.sensitive else None,
                             "verified": bool(r.verified)}
                            for r in recent.sort_values("utc_time", ascending=False).head(25).itertuples()]},
        "gelatinous": {"events": gel_events, "status": gel_status,
                       "drifter_recent": int(((records["gelatinous"] == "drifter") & (day > cutoff)).sum()),
                       "water_column_recent": int(((records["gelatinous"] == "water_column") & (day > cutoff)).sum()),
                       "stinging_recent": int((records["stinging"] & (day > cutoff)).sum()),
                       "by_month": [int(((records["gelatinous"] != "") & (day.dt.month == m)).sum())
                                    for m in range(1, 13)],
                       # so an empty month can say when the last one actually was, rather than just 0
                       "last_record": (records.loc[records["gelatinous"] != "", "local_date"].max()
                                       if (records["gelatinous"] != "").any() else None),
                       "total": int((records["gelatinous"] != "").sum())},
        "invasive": wl.invasive_watch(records, species_cfg),
        # species followed closely, each appearance with the sea conditions behind it
        "watch": [wl.watch_report(records, merged_by_key, w, triggers)
                  for w in (wcfg.get("watch_species") or [])],
        "strandings_log": strandings_summary,
        "strandings": {"candidates": int(len(cand)),
                       "by_group": {g: int(n) for g, n in cand["group"].value_counts().items()},
                       "by_year": {str(y): int(n) for y, n in
                                   pd.to_datetime(cand["local_date"]).dt.year.value_counts().sort_index().items()},
                       # whether the export carried anything to read at all: with no condition field
                       # and no notes, zero means "nothing to search", not "nothing happened"
                       "text_source": bool(records["condition"].astype(str).str.strip().any()
                                           or ("notes" in records and
                                               records["notes"].astype(str).str.strip().any())),
                       "note": "from the app; the TNP strandings log is the record of truth"},
        # An export with the User column stripped out carries no contributor counts at all, so this
        # comes back empty rather than as a year of NaN.
        "effort": {"contributors_per_year": {str(y): int(v) for y, v in
                                             records.groupby(day.dt.year)["day_contributors"]
                                             .max().dropna().items()},
                   "records_per_day_median": _r(records.groupby("local_date").size().median(), 1)},
    }
    # allow_nan=False turns a stray NaN into a loud error here rather than a file that looks fine on
    # the server and fails silently in the browser.
    (out_dir / "wildlife.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    return len(records)


def export(config: dict, db: Database, log=print, today=None):
    ecfg = config.get("export", {})
    out_dir = Path(ecfg.get("output_dir", "public/data"))
    (out_dir / "daily").mkdir(parents=True, exist_ok=True)
    (out_dir / "forecast").mkdir(parents=True, exist_ok=True)
    priorities = ecfg.get("daily_priority", {})
    variables = config.get("variables", {})
    today = pd.Timestamp(today or datetime.now(timezone.utc).date())

    n_clean = db.clean_out_of_range(variables)
    if n_clean:
        log(f"Blanked {n_clean} out-of-range statistics before export")

    grouped = defaultdict(dict)    # (variable, location) -> {source: daily series up to today}
    forecasts = defaultdict(dict)  # (variable, location) -> {source: daily series after today}

    def add(variable, location, source, s):
        if s is None or s.empty:
            return
        s = s.dropna()
        past, future = s[s.index <= today], s[s.index > today]
        if not past.empty:
            grouped[(variable, location)][source] = past
        if not future.empty:
            forecasts[(variable, location)][source] = future

    up = ecfg.get("upwelling") or {}
    up_wind = up.get("wind") or {}
    tides_cfg = ecfg.get("tides") or {}
    tide_payload = None
    tide_fit = None

    for source, variable, location in db.series_keys():
        if variable == "sea_level_hourly":
            continue                                           # handled by the tide analysis below
        stat = variables.get(variable, {}).get("daily_statistic", "mean")
        series = db.fetch_series(variable, location, source, statistic=stat)
        s = daily_aggregate(series, variable)
        if s.empty:
            continue
        add(variable, location, source, s)
        if variable == "sw_rad":
            # derived series kept separate from satellite PAR, so the two are never silently mixed
            df = pd.DataFrame(series, columns=["t", "v"])
            df["t"] = pd.to_datetime(df["t"])
            counts = df.groupby(df["t"].dt.normalize())["v"].count()
            complete = s[counts.reindex(s.index).fillna(0) >= 20]
            if not complete.empty:
                add("par_era5", location, source, par_from_shortwave(complete))
        if (variable == "wind_speed" and up and location == up_wind.get("location")
                and source in (up_wind.get("sources") or [])):
            ui = dv.daily_upwelling_index(series, db.fetch_series("wind_dir", location, source),
                                          float(up.get("coast_bearing", 70)), float(up.get("latitude", 36.5)))
            add("upwelling_index", location, source, ui)

    if tides_cfg:
        hourly = db.fetch_series("sea_level_hourly", tides_cfg["location"], tides_cfg["source"])
        if hourly:
            hs = pd.Series({pd.Timestamp(t): v for t, v in hourly})
            res = dv.tide_analysis(hs, datetime.now(timezone.utc).replace(tzinfo=None),
                                   predict_days=int(tides_cfg.get("predict_days", 7)),
                                   sample_offset_minutes=float(tides_cfg.get("sample_offset_minutes", 30)),
                                   datum=str(tides_cfg.get("datum", "chart")),
                                   chart_datum_offset_m=tides_cfg.get("chart_datum_offset_m"),
                                   max_daily_offset_m=float(tides_cfg.get("max_daily_offset_m", 0.5)))
            if res:
                level, surge, tide_payload = res
                # a fit of the last year, kept for placing sightings in the tidal cycle
                shifted = hs.copy()
                shifted.index = shifted.index + pd.Timedelta(
                    minutes=float(tides_cfg.get("sample_offset_minutes", 30)))
                recent = shifted[shifted.index > shifted.index.max() - pd.Timedelta(days=365)]
                tide_fit = dv.fit_tide(recent) if len(recent) > 24 * 30 else None
                add("sea_level", tides_cfg["location"], tides_cfg["source"], level)
                add("surge", tides_cfg["location"], tides_cfg["source"], surge)
                tide_payload.update(station=source_label(config, tides_cfg["source"]),
                                    location=tides_cfg["location"])

    def merged_for(variable, location):
        per_source = grouped.get((variable, location), {})
        return merge_by_priority(per_source, priorities.get(variable, [])) if per_source else pd.DataFrame()

    # light reaching the seabed, from the combined satellite PAR and KD490 of each area
    light = ecfg.get("seabed_light") or {}
    if light:
        for loc in config.get("areas", {}):
            par, kd = merged_for("par", loc), merged_for("kd490", loc)
            if par.empty or kd.empty:
                continue
            for d, series in dv.light_at_depths(par["value"].astype(float), kd["value"].astype(float),
                                                light.get("depths", [5, 10, 20])).items():
                add(f"par_{int(d)}m", loc, "derived_light", series)

    # rainfall totals over the last week and month, which say more about run-off than one day's rain
    rain_cfg = ecfg.get("rain") or {}
    rain_var = rain_cfg.get("variable", "precip")
    if rain_cfg:
        for (variable, loc) in list(grouped):
            if variable != rain_var:
                continue
            daily = merged_for(variable, loc)
            if daily.empty:
                continue
            for w in rain_cfg.get("windows", [7, 30]):
                add(f"{rain_var}_{int(w)}d", loc, "derived_rain", dv.rolling_totals(daily["value"], int(w)))

    merged_by_key = {}
    latest = []
    for (variable, location), per_source in sorted(grouped.items()):
        merged = merge_by_priority(per_source, priorities.get(variable, []))
        if merged.empty:
            continue
        merged_by_key[(variable, location)] = merged
        clim = climatology(merged, circular=variable.endswith("_dir"))
        rec = record_stats(merged)

        # Where more than one source exists, also publish each source on its own, so the dashboard
        # can offer "combined" or any single source (with its own record and normal range).
        options = []
        if len(per_source) > 1:
            order = [s for s in priorities.get(variable, []) if s in per_source] + \
                    [s for s in per_source if s not in priorities.get(variable, [])]
            for src in order:
                single = merge_by_priority({src: per_source[src]}, [src])
                if single.empty:
                    continue
                fname = f"{variable}__{location}__{src}.json"
                _write_series(out_dir / "daily" / fname, config, variable, location, single,
                              climatology(single, circular=variable.endswith("_dir")), record_stats(single))
                options.append({"source": src, "label": source_label(config, src), "file": fname,
                                "first": single.index.min().strftime("%Y-%m-%d"),
                                "last": single.index.max().strftime("%Y-%m-%d")})
        _write_series(out_dir / "daily" / f"{variable}__{location}.json", config, variable, location,
                      merged, clim, rec, options)

        last_day = merged.index.max()
        last_val = float(merged.loc[last_day, "value"])
        anomaly = z = mean = sd = None
        if clim and not variable.endswith("_dir"):
            d = min(last_day.dayofyear, 365) - 1
            mean, sd = clim["mean"][d], clim["sd"][d]
            if mean is not None:
                anomaly = _r(last_val - mean)
                z = _r((last_val - mean) / sd, 2) if sd else None
        latest.append({"variable": variable, "location": location, "date": last_day.strftime("%Y-%m-%d"),
                       "value": _r(last_val), "clim_mean": mean, "clim_sd": sd, "anomaly": anomaly, "z": z,
                       "source": merged.loc[last_day, "source"]})

    # ---------------- forecasts ----------------
    fcfg = ecfg.get("forecast") or {}
    for var, bc in (fcfg.get("bias_correct") or {}).items():
        for (fvar, loc), per_source in list(forecasts.items()):
            if fvar != bc["from"] or (var, loc) in forecasts:
                continue
            obs, model = merged_by_key.get((var, loc)), merged_by_key.get((fvar, loc))
            if obs is None or model is None:
                continue
            diff = (obs["value"].astype(float) - model["value"].astype(float)).dropna()
            diff = diff[diff.index > today - pd.Timedelta(days=int(bc.get("days", 14)))]
            if diff.size < 3:
                continue
            for src, fc in per_source.items():
                forecasts[(var, loc)][src] = fc + float(diff.mean())
                forecasts[(var, loc)]["_offset"] = float(diff.mean())
    forecast_keys = []
    for (var, loc), per_source in sorted(forecasts.items()):
        offset = per_source.pop("_offset", None)
        if (var, loc) not in merged_by_key or not per_source:
            continue
        order = [s for s in priorities.get(var, []) if s in per_source] + [s for s in per_source if s not in priorities.get(var, [])]
        src = order[0]
        fc = per_source[src].sort_index()
        vmeta = variables.get(var, {})
        payload = {"variable": var, "location": loc, "unit": vmeta.get("unit"), "issued": today.strftime("%Y-%m-%d"),
                   "source": src, "source_label": source_label(config, src),
                   "bias_offset": _r(offset, 3) if offset is not None else None,
                   "data": [[d.strftime("%Y-%m-%d"), _r(float(v))] for d, v in fc.items()]}
        (out_dir / "forecast" / f"{var}__{loc}.json").write_text(json.dumps(payload, separators=(",", ":")))
        forecast_keys.append(f"{var}__{loc}")

    # ---------------- events ----------------
    events = dust_events(config, merged_by_key, log=log)
    (out_dir / "dust_events.json").write_text(json.dumps(events, indent=1))

    mcfg = ecfg.get("marine_heatwaves") or {}
    mhw_by_var = {}
    if mcfg:
        mvars = [mcfg.get("variable", "sst")] + list(mcfg.get("extra_variables") or [])
        for var in mvars:
            out = {}
            for loc in mcfg.get("locations", []):
                m = merged_by_key.get((var, loc))
                if m is None or m.empty:
                    continue
                res = marine_heatwaves(m["value"].astype(float), tuple(mcfg.get("baseline", (1991, 2020))),
                                       int(mcfg.get("min_days", 5)), int(mcfg.get("max_gap_days", 2)))
                if res:
                    out[loc] = res
            mhw_by_var[var] = out
            name = "marine_heatwaves.json" if var == mvars[0] else f"marine_heatwaves__{var}.json"
            (out_dir / name).write_text(json.dumps(out, separators=(",", ":")))
        log("Marine heatwaves: " + ", ".join(f"{v}/{k} {len(r['events'])}" for v, o in mhw_by_var.items() for k, r in o.items()))

    up_events = []
    if up:
        m = merged_by_key.get(("upwelling_index", up_wind.get("location")))
        if m is not None and not m.empty:
            up_events, status = dv.upwelling_events(m["value"].astype(float), float(up.get("threshold", 500)),
                                                    int(up.get("min_days", 2)), int(up.get("max_gap_days", 1)))
            for ev in up_events:
                ev["response"] = dv.response_stats(merged_by_key, up.get("response", []), pd.Timestamp(ev["start"]),
                                                   pd.Timestamp(ev["end"]), int(up.get("response_window_days", 7)),
                                                   int(up.get("response_lag_days", 3)))
            fc = forecasts.get(("upwelling_index", up_wind.get("location")), {})
            fc_src = next((s for s in (up_wind.get("sources") or []) if s in fc), None)
            fc_days = [[d.strftime("%Y-%m-%d"), _r(float(v), 0)] for d, v in fc[fc_src].items()] if fc_src else []
            payload = {"threshold": float(up.get("threshold", 500)), "min_days": int(up.get("min_days", 2)),
                       "coast_bearing": float(up.get("coast_bearing", 70)), "unit": "m3 s-1 km-1",
                       "location": up_wind.get("location"), "events": up_events, "status": status,
                       "forecast": fc_days}
            (out_dir / "upwelling.json").write_text(json.dumps(payload, separators=(",", ":")))
            log(f"Upwelling: {len(up_events)} events, now {status['state'] if status else 'n/a'}")

    rain_payload = None
    if rain_cfg:
        m = merged_by_key.get((rain_var, rain_cfg.get("location", "gibraltar_airport")))
        if m is not None and not m.empty:
            rain_payload = dv.rain_events(m["value"].astype(float), float(rain_cfg.get("wet_mm", 1)),
                                          float(rain_cfg.get("heavy_mm", 20)), float(rain_cfg.get("flush_mm", 10)),
                                          int(rain_cfg.get("dry_days", 30)))
            rain_payload.update({k: rain_cfg.get(k) for k in ("location", "wet_mm", "heavy_mm", "flush_mm", "dry_days")})
            (out_dir / "rain.json").write_text(json.dumps(rain_payload, separators=(",", ":")))
            log(f"Rain: {len(rain_payload['events'])} heavy or first-flush spells, "
                f"{rain_payload['status']['dry_days']} dry days to {rain_payload['status']['date']}")
    rain_trigger = [{"start": e["start"], "end": e["end"]} for e in (rain_payload or {}).get("events", [])]

    bcfg = ecfg.get("blooms") or {}
    if bcfg:
        blooms = {}
        for loc in bcfg.get("locations", []):
            m = merged_by_key.get((bcfg.get("variable", "chl"), loc))
            if m is None or m.empty:
                continue
            res = dv.detect_blooms(m["value"].astype(float), float(bcfg.get("percentile", 0.9)),
                                   int(bcfg.get("min_days", 3)), int(bcfg.get("max_gap_days", 2)),
                                   int(bcfg.get("interp_days", 3)), int(bcfg.get("min_clear_days", 2)))
            if not res:
                continue
            triggers = {"upwelling": up_events,
                        "dust": [{"start": e["start"], "end": e["end"]} for e in events.get("episodes", [])],
                        "heatwave": (mhw_by_var.get(mcfg.get("variable", "sst"), {}).get(loc) or {}).get("events", []),
                        "rain": rain_trigger}
            dv.attach_triggers(res["events"], triggers, int(bcfg.get("trigger_lookback_days", 10)))
            blooms[loc] = res
        (out_dir / "blooms.json").write_text(json.dumps(blooms, separators=(",", ":")))
        log("Blooms: " + ", ".join(f"{k} {len(v['events'])}" for k, v in blooms.items()))

    if tide_payload:
        (out_dir / "tides.json").write_text(json.dumps(tide_payload, separators=(",", ":")))

    wcfg = ecfg.get("wildlife") or {}
    if wcfg.get("enabled"):
        # The same event lists the bloom card uses, so a sighting and a chlorophyll bloom name the
        # same heatwave. Each watch species picks the area it is actually reported in.
        mhw_main = mhw_by_var.get(mcfg.get("variable", "sst"), {})
        watch_area = next((w.get("area") for w in (wcfg.get("watch_species") or []) if w.get("area")),
                          None)
        wildlife_triggers = {
            "upwelling": up_events,
            "dust": [{"start": e["start"], "end": e["end"]} for e in events.get("episodes", [])],
            "heatwave": (mhw_main.get(watch_area) or mhw_main.get(mcfg.get("locations", [None])[0])
                         or {}).get("events", []),
        }
        try:
            n_w = export_wildlife(config, wcfg, merged_by_key, tide_fit, out_dir, log=log,
                                  triggers=wildlife_triggers, today=today)
            log(f"Wildlife: {n_w} sightings processed")
        except FileNotFoundError as e:
            log(f"Wildlife: skipped ({e})")

    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variables": variables,
        "groups": {"physical": "Physical", "chemical": "Chemical", "biological": "Biological"},
        "areas": config.get("areas", {}),
        "points": config.get("points", {}),
        "sources": {k: {"label": v.get("label"), "description": v.get("description"), "product": v.get("product"),
                        "dataset": v.get("dataset_id") or v.get("short_name") or v.get("station_id") or v.get("station")}
                    for k, v in config["sources"].items()},
        "series": [f"{v}__{l}" for v, l in sorted(merged_by_key)],
        "forecasts": forecast_keys,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    (out_dir / "latest.json").write_text(json.dumps(latest, indent=1))

    # publish the dashboard page alongside the data
    site_dir = Path(ecfg.get("site_dir", Path(__file__).resolve().parents[1] / "site"))
    if site_dir.is_dir():
        for f in site_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, out_dir.parent / f.name)
    log(f"Exported {len(merged_by_key)} series, {len(forecast_keys)} forecasts and "
        f"{len(events['episodes'])} dust episodes to {out_dir}")
    return len(merged_by_key)
