"""Derived products built from harvested series (no network, no heavy libraries).

- upwelling index: Ekman transport from hourly wind, and upwelling events
- light reaching the seabed: satellite PAR attenuated with a KdPAR estimated from KD490
- chlorophyll bloom events against the seasonal normal
- tides: harmonic analysis of a tide gauge, tide predictions, high and low waters, and surge
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def runs(flags, min_len: int, max_gap: int = 0):
    """[start, end] index pairs of True runs at least `min_len` long; runs `max_gap` or fewer apart are joined."""
    flags = np.asarray(flags, dtype=bool)
    found, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if (not f or i == len(flags) - 1) and start is not None:
            end = i if f else i - 1
            if end - start + 1 >= min_len:
                found.append([start, end])
            start = None
    joined = []
    for r in found:
        if joined and r[0] - joined[-1][1] - 1 <= max_gap:
            joined[-1][1] = r[1]
        else:
            joined.append(r)
    return joined


def leap_doy(index: pd.DatetimeIndex) -> np.ndarray:
    """Day of year on a 366-day calendar, so 1 March is day 61 in every year."""
    return np.array([pd.Timestamp(2000, d.month, d.day).dayofyear for d in index])


def circular_smooth(a: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    padded = np.concatenate([a[-half:], a, a[:half]])
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def seasonal_quantiles(series: pd.Series, qs=(0.5, 0.9), half_window=15, smooth_days=31, min_n=10):
    """Per-day-of-year quantiles on a 366-day calendar, from all years, smoothed."""
    s = series.dropna()
    doy = leap_doy(s.index)
    vals = s.values.astype(float)
    out = {q: np.full(366, np.nan) for q in qs}
    for d in range(1, 367):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, 366 - dist)
        sel = vals[dist <= half_window]
        if sel.size >= min_n:
            for q in qs:
                out[q][d - 1] = np.quantile(sel, q)
    idx = np.arange(366)
    for q in qs:
        ok = np.isfinite(out[q])
        if not ok.any():
            return None
        if not ok.all():
            out[q] = np.interp(idx, idx[ok], out[q][ok], period=366)
        out[q] = circular_smooth(out[q], smooth_days)
    return out


# ------------------------------------------------------------------------------------------------
# Upwelling
# ------------------------------------------------------------------------------------------------
RHO_AIR = 1.22        # kg m-3
RHO_SEA = 1025.0      # kg m-3
OMEGA = 7.2921e-5     # s-1


def drag_coefficient(speed):
    """Large and Pond (1981) neutral drag coefficient, held constant below 4 m/s."""
    u = np.asarray(speed, dtype="float64")
    return np.where(u <= 11.0, 1.2e-3, (0.49 + 0.065 * u) * 1e-3)


def ekman_upwelling_index(speed, direction_from, coast_bearing: float, latitude: float):
    """Offshore Ekman transport per km of coast (m3 s-1 km-1); positive favours upwelling.

    speed (m/s) and direction the wind blows FROM (deg). coast_bearing is the direction along the
    coast with the sea on the right-hand side (Northern Hemisphere Ekman transport is 90 deg to the
    right of the wind, so wind blowing along this bearing pushes surface water offshore).
    For the Spanish Alboran coast west of Malaga the coast runs towards about 70 deg (ENE).
    """
    u = np.asarray(speed, dtype="float64")
    rad = np.radians(np.asarray(direction_from, dtype="float64"))
    east, north = -u * np.sin(rad), -u * np.cos(rad)                 # wind vector blowing towards
    b = np.radians(coast_bearing)
    along = east * np.sin(b) + north * np.cos(b)
    tau = RHO_AIR * drag_coefficient(u) * u * along                  # N m-2, alongshore component
    f = 2 * OMEGA * np.sin(np.radians(latitude))
    return tau / (RHO_SEA * f) * 1000.0                              # m2 s-1 -> m3 s-1 per km


def daily_upwelling_index(hourly_speed: list, hourly_dir: list, coast_bearing: float, latitude: float,
                          min_hours: int = 20) -> pd.Series:
    """Hourly (time, value) lists for speed and direction -> daily mean upwelling index."""
    if not hourly_speed or not hourly_dir:
        return pd.Series(dtype="float64")
    sp = pd.Series({pd.Timestamp(t): v for t, v in hourly_speed})
    dr = pd.Series({pd.Timestamp(t): v for t, v in hourly_dir})
    df = pd.concat({"s": sp, "d": dr}, axis=1).dropna()
    if df.empty:
        return pd.Series(dtype="float64")
    idx = ekman_upwelling_index(df["s"].values, df["d"].values, coast_bearing, latitude)
    h = pd.Series(idx, index=df.index)
    g = h.groupby(h.index.normalize())
    return g.mean().where(g.count() >= min_hours).dropna()


def response_stats(merged_by_key: dict, responses: list, start: pd.Timestamp, end: pd.Timestamp,
                   window: int, lag_after: int = 0, clim_by_key: dict | None = None):
    """Mean of each response variable in the `window` days before `start` and from start to end+lag."""
    out = []
    for r in responses:
        m = merged_by_key.get((r["variable"], r["location"]))
        if m is None or m.empty:
            continue
        s = m["value"].astype(float)
        before = s[(s.index >= start - pd.Timedelta(days=window)) & (s.index < start)]
        during = s[(s.index >= start) & (s.index <= end + pd.Timedelta(days=lag_after))]
        b = before.mean() if before.size else np.nan
        a = during.mean() if during.size else np.nan
        item = {"variable": r["variable"], "location": r["location"], "before_mean": _r(b), "during_mean": _r(a),
                "n_before": int(before.size), "n_during": int(during.size),
                "change": _r(a - b) if np.isfinite(a) and np.isfinite(b) else None,
                "change_pct": _r((a - b) / b * 100, 1) if np.isfinite(a) and np.isfinite(b) and b else None}
        out.append(item)
    return out


def upwelling_events(index: pd.Series, threshold: float, min_days: int = 2, max_gap_days: int = 1):
    s = index.dropna().astype(float)
    if s.empty:
        return [], None
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full)
    flags = (s.values > threshold) & np.isfinite(s.values)
    events = []
    for a, b in runs(flags, min_days, max_gap_days):
        v = s.values[a:b + 1]
        k = int(np.nanargmax(v))
        events.append({"start": full[a].strftime("%Y-%m-%d"), "end": full[b].strftime("%Y-%m-%d"),
                       "days": int(b - a + 1), "peak_date": full[a + k].strftime("%Y-%m-%d"),
                       "peak_index": _r(np.nanmax(v), 0), "mean_index": _r(np.nanmean(v), 0)})
    last = len(full) - 1
    status = {"date": full[last].strftime("%Y-%m-%d"), "index": _r(s.values[last], 0), "state": "none"}
    if events and events[-1]["end"] == status["date"]:
        status.update(state="upwelling", since=events[-1]["start"], days=events[-1]["days"])
    elif flags[last]:
        n = 0
        while last - n >= 0 and flags[last - n]:
            n += 1
        status.update(state="favourable", days=int(n))
    return events, status


# ------------------------------------------------------------------------------------------------
# Light at the seabed
# ------------------------------------------------------------------------------------------------
def kd_par_from_kd490(kd490):
    """Morel et al. (2007) relation between Kd(PAR) over the first optical depth and Kd(490)."""
    k = np.asarray(kd490, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        kp = 0.0864 + 0.884 * k - 0.00137 / k
    return np.clip(kp, 0.02, None)


def light_at_depths(par: pd.Series, kd490: pd.Series, depths) -> dict:
    """{depth: daily PAR at that depth} for days with both surface PAR and KD490."""
    df = pd.concat({"par": par, "kd": kd490}, axis=1).dropna()
    if df.empty:
        return {}
    kp = kd_par_from_kd490(df["kd"].values)
    return {d: pd.Series(df["par"].values * np.exp(-kp * float(d)), index=df.index) for d in depths}


# ------------------------------------------------------------------------------------------------
# Blooms
# ------------------------------------------------------------------------------------------------
def detect_blooms(chl: pd.Series, percentile: float = 0.9, min_days: int = 3, max_gap_days: int = 2,
                  interp_days: int = 3, min_obs: int = 2, min_years: int = 3):
    """Chlorophyll bloom events: runs of days above the seasonal `percentile` of all years.

    Cloud gaps of up to `interp_days` are bridged by interpolating log10(chl); an event also needs at
    least `min_obs` real (cloud-free) days. Peak ratio = peak chlorophyll / seasonal median.
    """
    s = chl.dropna().astype(float)
    s = s[s > 0]
    if s.empty or s.index.year.nunique() < min_years:
        return None
    q = seasonal_quantiles(np.log10(s), qs=(0.5, percentile))
    if q is None:
        return None
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    logs = np.log10(s).reindex(full)
    real = np.isfinite(logs.values)
    filled = logs.interpolate(limit=interp_days, limit_area="inside")
    doy = leap_doy(full) - 1
    med, thr = q[0.5][doy], q[percentile][doy]
    flags = (filled.values > thr) & np.isfinite(filled.values)

    events = []
    for a, b in runs(flags, min_days, max_gap_days):
        obs_idx = [i for i in range(a, b + 1) if real[i]]
        if len(obs_idx) < min_obs:
            continue
        k = max(obs_idx, key=lambda i: logs.values[i])
        peak = 10 ** logs.values[k]
        events.append({"start": full[a].strftime("%Y-%m-%d"), "end": full[b].strftime("%Y-%m-%d"),
                       "days": int(b - a + 1), "clear_days": len(obs_idx),
                       "peak_date": full[k].strftime("%Y-%m-%d"), "peak": _r(peak, 3),
                       "peak_ratio": _r(peak / 10 ** med[k], 2)})
    last = len(full) - 1
    status = {"date": full[last].strftime("%Y-%m-%d"), "value": _r(10 ** logs.values[last], 3),
              "normal": _r(10 ** med[last], 3), "threshold": _r(10 ** thr[last], 3),
              "ratio": _r(10 ** (logs.values[last] - med[last]), 2), "state": "none"}
    if events and events[-1]["end"] == status["date"]:
        status.update(state="bloom", since=events[-1]["start"], days=events[-1]["days"])
    elif flags[last]:
        n = 0
        while last - n >= 0 and flags[last - n]:
            n += 1
        status.update(state="elevated", days=int(n))
    return {"percentile": int(percentile * 100), "years": [int(s.index.year.min()), int(s.index.year.max())],
            "normal": [_r(10 ** x, 3) for x in q[0.5]], "threshold": [_r(10 ** x, 3) for x in q[percentile]],
            "events": events, "status": status}


def rolling_totals(daily: pd.Series, days: int) -> pd.Series:
    """Running total over the last `days` days (inclusive), only where every one of those days has data."""
    if daily is None or daily.empty:
        return pd.Series(dtype=float)
    s = daily.astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    return full.rolling(int(days), min_periods=int(days)).sum().dropna()


def rain_events(daily: pd.Series, wet_mm: float = 1.0, heavy_mm: float = 20.0, flush_mm: float = 10.0,
                dry_days: int = 30) -> dict:
    """Wet spells that matter for run-off, from daily rainfall (mm).

    A wet spell is a run of consecutive days with at least `wet_mm`. It is listed when either
      - heavy: any day in it reaches `heavy_mm`, or
      - first flush: it totals at least `flush_mm` and follows `dry_days` or more dry days,
        so the first proper rain washes what has built up on a dry catchment into the sea.
    Missing days break a spell and are not counted as dry.
    Returns {"events": [...], "status": {...}} with dates as YYYY-MM-DD.
    """
    out = {"events": [], "status": None}
    if daily is None or daily.empty:
        return out
    s = daily.astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    vals = full.values
    days = full.index
    wet = np.where(np.isfinite(vals), vals >= wet_mm, False)
    known = np.isfinite(vals)

    dry_run = 0          # dry days immediately before the current position (missing days reset it)
    i, n = 0, len(vals)
    while i < n:
        if not known[i]:
            dry_run = 0
            i += 1
            continue
        if not wet[i]:
            dry_run += 1
            i += 1
            continue
        j = i
        while j + 1 < n and known[j + 1] and wet[j + 1]:
            j += 1
        seg = vals[i:j + 1]
        total, peak = float(np.nansum(seg)), float(np.nanmax(seg))
        heavy = peak >= heavy_mm
        flush = total >= flush_mm and dry_run >= dry_days
        if heavy or flush:
            k = int(np.nanargmax(seg))
            out["events"].append({
                "start": days[i].strftime("%Y-%m-%d"), "end": days[j].strftime("%Y-%m-%d"),
                "days": int(j - i + 1), "total": _r(total, 1), "peak": _r(peak, 1),
                "peak_date": days[i + k].strftime("%Y-%m-%d"),
                "kind": "first_flush" if flush else "heavy", "heavy": bool(heavy),
                "dry_before": int(dry_run)})
        dry_run = 0
        i = j + 1

    # where things stand on the last day with data
    last_i = int(np.where(known)[0][-1])
    since = 0
    for k in range(last_i, -1, -1):
        if not known[k] or wet[k]:
            break
        since += 1
    last_wet = next((days[k].strftime("%Y-%m-%d") for k in range(last_i, -1, -1) if known[k] and wet[k]), None)
    out["status"] = {"date": days[last_i].strftime("%Y-%m-%d"), "dry_days": int(since), "last_wet_day": last_wet}
    return out


def attach_triggers(bloom_events: list, triggers: dict, lookback_days: int = 10, overlap_days: int = 2):
    """Add `triggers` to each bloom: events of other kinds that ended within `lookback_days` before the
    bloom started (or overlapped its first days). triggers = {kind: [{start, end, ...}]}."""
    for ev in bloom_events:
        b0 = pd.Timestamp(ev["start"])
        found = []
        for kind, items in triggers.items():
            for t in items or []:
                t0, t1 = pd.Timestamp(t["start"]), pd.Timestamp(t["end"])
                if t0 <= b0 + pd.Timedelta(days=overlap_days) and t1 >= b0 - pd.Timedelta(days=lookback_days):
                    found.append({"type": kind, "start": t["start"], "end": t["end"]})
        ev["triggers"] = found
    return bloom_events


# ------------------------------------------------------------------------------------------------
# Moon
# ------------------------------------------------------------------------------------------------
SYNODIC = 29.530588853


def moon_elongation(when) -> float:
    """Angle between sun and moon in degrees: 0 at new moon, 180 at full.

    Low-precision ecliptic longitudes (Meeus, Astronomical Algorithms), good to a few minutes on the
    phase times, which is far more than a sighting date needs.
    """
    t = pd.Timestamp(when)
    n = (t - pd.Timestamp("2000-01-01 12:00")).total_seconds() / 86400.0
    rad = np.pi / 180
    g = (357.5291092 + 0.98560028 * n) * rad
    lsun = 280.46646 + 0.98564736 * n + 1.914602 * np.sin(g) + 0.019993 * np.sin(2 * g) - 0.00569
    mm = (134.9633964 + 13.06499295 * n) * rad
    d = (297.8501921 + 12.19074912 * n) * rad
    f = (93.2720950 + 13.22935024 * n) * rad
    lmoon = (218.3164477 + 13.17639648 * n + 6.288774 * np.sin(mm) + 1.274027 * np.sin(2 * d - mm)
             + 0.658314 * np.sin(2 * d) + 0.213618 * np.sin(2 * mm) - 0.185116 * np.sin(g)
             - 0.114332 * np.sin(2 * f) + 0.058793 * np.sin(2 * d - 2 * mm)
             + 0.057066 * np.sin(2 * d - g - mm) + 0.053322 * np.sin(2 * d + mm)
             + 0.045758 * np.sin(2 * d - g) - 0.040923 * np.sin(g - mm) - 0.034720 * np.sin(d)
             - 0.030383 * np.sin(g + mm))
    return float((lmoon - lsun) % 360.0)


def moon_illumination(when) -> float:
    """Lit fraction of the moon's disc, 0 at new and 1 at full."""
    return float((1 - np.cos(np.radians(moon_elongation(when)))) / 2)


def moon_phase_name(when) -> str:
    e = moon_elongation(when)
    near = lambda x: min(abs(e - x), 360 - abs(e - x)) <= 12
    if near(0):
        return "New moon"
    if near(90):
        return "First quarter"
    if near(180):
        return "Full moon"
    if near(270):
        return "Last quarter"
    return ("Waxing crescent" if e < 90 else "Waxing gibbous" if e < 180
            else "Waning gibbous" if e < 270 else "Waning crescent")


# ------------------------------------------------------------------------------------------------
# Tides
# ------------------------------------------------------------------------------------------------
# Constituent speeds in degrees per hour.
CONSTITUENTS = {
    "M2": 28.9841042, "S2": 30.0000000, "N2": 28.4397295, "K2": 30.0821373,
    "K1": 15.0410686, "O1": 13.9430356, "P1": 14.9589314, "Q1": 13.3986609,
    "2N2": 27.8953548, "MU2": 27.9682084, "NU2": 28.5125831, "L2": 29.5284789,
    "M4": 57.9682084, "MS4": 58.9841042, "MN4": 57.4238337, "M6": 86.9523127,
    "MK3": 44.0251729, "2MS6": 87.9682084,
}
EPOCH = pd.Timestamp("2000-01-01")


def _hours(index) -> np.ndarray:
    return (pd.DatetimeIndex(index) - EPOCH) / pd.Timedelta(hours=1)


PRIORITY = ["M2", "S2", "K1", "O1", "N2", "K2", "P1", "Q1", "M4", "MS4", "MN4",
            "2N2", "NU2", "MU2", "L2", "M6", "MK3", "2MS6"]


def usable_constituents(span_hours: float) -> list[str]:
    """Constituents in order of importance, skipping any that can't be separated from one already chosen
    over a record this long (Rayleigh criterion: frequency difference x span >= one cycle)."""
    keep = []
    for n in PRIORITY:
        if all(abs(CONSTITUENTS[n] - CONSTITUENTS[k]) * span_hours >= 360.0 for k in keep):
            keep.append(n)
    return keep


def fit_tide(series: pd.Series, constituents: list[str] | None = None) -> dict | None:
    """Least-squares harmonic fit of an hourly sea-level series. Returns mean, amplitudes and phases."""
    s = series.dropna()
    if len(s) < 24 * 30:
        return None
    t = _hours(s.index)
    span = t.max() - t.min()
    names = constituents or usable_constituents(span)
    cols = [np.ones_like(t)]
    for n in names:
        w = np.radians(CONSTITUENTS[n])
        cols += [np.cos(w * t), np.sin(w * t)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, s.values.astype(float), rcond=None)
    fit = {"mean": float(coef[0]), "start": str(s.index.min()), "end": str(s.index.max()), "constituents": {}}
    for i, n in enumerate(names):
        a, b = coef[1 + 2 * i], coef[2 + 2 * i]
        fit["constituents"][n] = {"amp": float(np.hypot(a, b)), "phase": float(np.degrees(np.arctan2(b, a)) % 360),
                                  "a": float(a), "b": float(b)}
    return fit


def predict_tide(fit: dict, index, include_mean: bool = True) -> pd.Series:
    t = _hours(index)
    y = np.full(len(t), fit["mean"] if include_mean else 0.0)
    for n, c in fit["constituents"].items():
        w = np.radians(CONSTITUENTS[n])
        y = y + c["a"] * np.cos(w * t) + c["b"] * np.sin(w * t)
    return pd.Series(y, index=pd.DatetimeIndex(index))


def tide_extremes(pred: pd.Series, min_separation_hours: float = 3.0):
    """High and low waters from a finely sampled prediction: [(time, 'high'|'low', height)]."""
    v = pred.values
    d = np.diff(v)
    out = []
    for i in range(1, len(v) - 1):
        if d[i - 1] > 0 and d[i] <= 0:
            kind = "high"
        elif d[i - 1] < 0 and d[i] >= 0:
            kind = "low"
        else:
            continue
        # refine with a parabola through the three samples
        y0, y1, y2 = v[i - 1], v[i], v[i + 1]
        denom = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / denom if denom else 0.0
        step = (pred.index[i + 1] - pred.index[i]).total_seconds()
        t = pred.index[i] + pd.Timedelta(seconds=off * step)
        h = y1 - 0.25 * (y0 - y2) * off
        if out and out[-1][1] == kind:
            if (kind == "high" and h > out[-1][2]) or (kind == "low" and h < out[-1][2]):
                out[-1] = (t, kind, h)
            continue
        if out and (t - out[-1][0]).total_seconds() / 3600 < min_separation_hours:
            continue
        out.append((t, kind, h))
    return out


def chart_datum_below_msl(fits: dict, step_minutes: int = 20) -> float | None:
    """Depth of chart datum below mean sea level, as the lowest astronomical tide in the record.

    Chart datum is the level tide tables measure from: the lowest tide expected under average weather
    (lowest astronomical tide). Each well-covered year is predicted from its own harmonic fit, with that
    year's mean level removed, and the lowest level reached across all those years is taken. Using every
    year covers the 18.6-year lunar cycle, which is what makes a tide low enough to set the datum.
    """
    if not fits:
        return None
    lows = []
    for year, f in fits.items():
        idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:59", freq=f"{step_minutes}min")
        lows.append(float(predict_tide(f, idx, include_mean=False).min()))
    return -min(lows) if lows else None


def drop_bad_days(hourly: pd.Series, max_offset: float = 0.5, window: int = 31, min_days: int = 5):
    """Remove whole days whose level sits far from its neighbours: sensor faults and datum shifts.

    A tide gauge can jump when the instrument is serviced or replaced. Daily medians are compared with
    a centred 31-day median of daily medians, and a day more than `max_offset` away is dropped.
    Returns (kept series, number of days dropped).
    """
    if hourly.empty:
        return hourly, 0
    daily = hourly.groupby(hourly.index.normalize()).median()
    neighbours = daily.rolling(window, center=True, min_periods=min_days).median()
    bad = daily.index[(daily - neighbours).abs() > max_offset]
    if len(bad) == 0:
        return hourly, 0
    return hourly[~hourly.index.normalize().isin(bad)], int(len(bad))


def high_pass(series: pd.Series, window: int = 31, min_days: int = 5) -> pd.Series:
    """Remove slow drift by subtracting a centred running median of daily medians.

    Storm surge lasts hours to days; a gauge's datum drifts over months. Taking the slow part out keeps
    the surge series about the weather rather than about the instrument.
    """
    if series.empty:
        return series
    daily = series.groupby(series.index.normalize()).median()
    slow = daily.rolling(window, center=True, min_periods=min_days).median().bfill().ffill()
    return series - slow.reindex(series.index.normalize()).values


def tide_analysis(hourly: pd.Series, now: datetime, predict_days: int = 7, fit_days: int = 365,
                  min_year_days: int = 300, sample_offset_minutes: float = 30.0,
                  datum: str = "chart", chart_datum_offset_m: float | None = None,
                  max_daily_offset_m: float = 0.5, level_baseline_days: int = 365):
    """Yearly harmonic fits -> residual (surge) for the whole record; latest fit -> predictions.

    `sample_offset_minutes` moves each reading to the middle of the period it averages. The harvester
    stores hourly means labelled at the start of the hour, so a mean labelled 08:00 covers 08:00-08:59
    and belongs at 08:30; without this the fitted tide, and every predicted high and low water, comes
    out half an hour early.

    Returns (daily_mean_sea_level, daily_mean_surge, tides_payload) or None if there is too little data.
    """
    s = hourly.dropna().astype(float).sort_index()
    s = s[~s.index.duplicated()]
    if sample_offset_minutes:
        s.index = s.index + pd.Timedelta(minutes=float(sample_offset_minutes))
    s, n_dropped = drop_bad_days(s, float(max_daily_offset_m))
    if len(s) < 24 * 60:
        return None
    last = s.index.max()
    recent = s[s.index > last - pd.Timedelta(days=fit_days)]
    latest_fit = fit_tide(recent)
    if latest_fit is None:
        return None

    # a fit per calendar year where the year is well covered (absorbs slow nodal changes and datum shifts)
    fits = {}
    for year, part in s.groupby(s.index.year):
        if part.index.normalize().nunique() >= min_year_days:
            f = fit_tide(part)
            if f:
                fits[year] = f
    residual_parts = []
    for year, part in s.groupby(s.index.year):
        if year in fits:
            f = fits[year]
            pred = predict_tide(f, part.index, include_mean=False) + part.mean()
        else:
            # nearest well-covered year's tide, with this stretch's own mean level
            if fits:
                near = min(fits, key=lambda y: abs(y - year))
                f = fits[near]
            else:
                f = latest_fit
            pred = predict_tide(f, part.index, include_mean=False)
            pred = pred + (part - pred).mean()
        residual_parts.append(part - pred)
    residual = high_pass(pd.concat(residual_parts).sort_index())    # weather, not gauge drift

    g = s.groupby(s.index.normalize())
    level_raw = g.mean().where(g.count() >= 20).dropna()
    # published as an anomaly against the past year, because the gauge's own zero has moved over the
    # record; a raw height chart would read as sea level change when it is really an instrument change
    base = level_raw.rolling(int(level_baseline_days), center=True, min_periods=90).median().bfill().ffill()
    daily_level = (level_raw - base).dropna()
    rg = residual.groupby(residual.index.normalize())
    daily_surge = rg.mean().where(rg.count() >= 20).dropna()

    # Reference level for published heights: chart datum (like tide tables) or mean sea level.
    z0 = chart_datum_offset_m if chart_datum_offset_m is not None else chart_datum_below_msl(fits)
    if datum == "chart" and z0 is None:
        z0 = float(-(predict_tide(latest_fit, pd.date_range(latest_fit["start"][:10], periods=366 * 72, freq="20min"),
                                  include_mean=False)).min())
    ref = latest_fit["mean"] - float(z0) if datum == "chart" else latest_fit["mean"]

    now = pd.Timestamp(now).floor("10min")
    fine = pd.date_range(now - pd.Timedelta(days=1), now + pd.Timedelta(days=predict_days), freq="10min")
    msl = ref
    pred_fine = predict_tide(latest_fit, fine) - msl
    extremes = [(t, k, h) for t, k, h in tide_extremes(pred_fine) if t >= now - pd.Timedelta(hours=12)]

    recent_obs = s[s.index > now - pd.Timedelta(days=3)]
    hourly_future = pd.date_range(now.floor("h") - pd.Timedelta(days=3), now.floor("h") + pd.Timedelta(days=3), freq="h")
    pred_hourly = predict_tide(latest_fit, hourly_future) - msl
    surge_now = residual[residual.index > last - pd.Timedelta(hours=24)]
    top = sorted(latest_fit["constituents"].items(), key=lambda kv: -kv[1]["amp"])[:8]
    payload = {
        "generated": now.strftime("%Y-%m-%dT%H:%MZ"),
        "datum": "chart datum" if datum == "chart" else "mean sea level over the last 12 months",
        "datum_note": ("chart datum, the level tide tables measure from (lowest astronomical tide, "
                       f"{_r(z0, 2)} m below mean sea level here)" if datum == "chart"
                       else "mean sea level over the last 12 months"),
        "chart_datum_below_msl": _r(z0, 2) if datum == "chart" else None,
        "mean_sea_level": _r(float(z0), 2) if datum == "chart" else 0.0,   # mean sea level above the reference
        "fit": {"start": latest_fit["start"][:10], "end": latest_fit["end"][:10],
                "constituents": [{"name": n, "amp": _r(c["amp"], 3), "phase": _r(c["phase"], 1)} for n, c in top]},
        "last_observation": last.strftime("%Y-%m-%dT%H:%MZ"),
        "days_dropped": n_dropped,
        "surge_last_24h": _r(surge_now.mean(), 3) if len(surge_now) else None,
        "extremes": [{"time": t.strftime("%Y-%m-%dT%H:%MZ"), "type": k, "height": _r(h, 2)} for t, k, h in extremes],
        "curve": [[t.strftime("%Y-%m-%dT%H:%MZ"), _r(p, 3),
                   _r(recent_obs.get(t, np.nan) - msl, 3) if t in recent_obs.index else None]
                  for t, p in pred_hourly.items()],
    }
    return daily_level, daily_surge, payload
