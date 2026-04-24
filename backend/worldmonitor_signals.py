#!/usr/bin/env python3
"""
Free, no-auth signals harvested for the PSX prediction overlay.

Sources (all keyless, all public):
- yfinance: ^VIX (CBOE volatility), ^HSI (Hang Seng), ^BSESN (Sensex), ^NSEI (Nifty 50)
- GDELT Doc API: tone + volume timelines for Pakistan-region conflict topics
- USGS: significant earthquakes near Pakistan (rare but tail-risk)

Design notes:
- Every fetcher returns a tz-naive DataFrame keyed on a `date` column so it
  composes with `external_features._to_naive_datetime` and `pd.merge_asof`.
- Every fetcher has a graceful empty-DataFrame fallback. Geo overlay must
  never crash because GDELT 429'd or yfinance hiccupped.
- All on-disk caches live under `data/external_cache/` with explicit TTLs.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False


CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "external_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _read_cache(name: str, max_age_seconds: int) -> Optional[dict]:
    p = CACHE_DIR / name
    if not p.exists():
        return None
    try:
        age = time.time() - p.stat().st_mtime
        if age > max_age_seconds:
            return None
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_cache(name: str, payload: dict) -> None:
    try:
        (CACHE_DIR / name).write_text(json.dumps(payload, default=str))
    except Exception:
        pass


def _http_get_json(url: str, timeout: int = 20, retries: int = 2, backoff: float = 1.5) -> Optional[dict]:
    """GET a URL with simple exponential backoff. Returns None on persistent failure."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PSXPredictor/1.0)"}
    delay = 0.0
    for attempt in range(retries + 1):
        try:
            if delay:
                time.sleep(delay)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception:
            delay = max(1.0, (delay or 1.0) * backoff)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Yahoo-sourced indices — VIX, Hang Seng, Sensex, Nifty
# ─────────────────────────────────────────────────────────────────────────────

def fetch_vix(start_date: Optional[str] = None, end_date: Optional[str] = None,
              period: str = "1y") -> pd.DataFrame:
    """CBOE Volatility Index (^VIX). Returns DataFrame with columns:
    date, vix_close, vix_change, vix_zscore_60d, vix_above_25, vix_above_30
    """
    if not YF_OK:
        return pd.DataFrame()
    try:
        kw = dict(progress=False, auto_adjust=True)
        if start_date and end_date:
            data = yf.download("^VIX", start=start_date, end=end_date, **kw)
        else:
            data = yf.download("^VIX", period=period, **kw)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty:
            return pd.DataFrame()
        close = data["Close"].astype(float)
        df = pd.DataFrame({
            "date": data.index,
            "vix_close": close.values,
            "vix_change": close.pct_change().values,
            "vix_zscore_60d": ((close - close.rolling(60).mean()) / close.rolling(60).std()).values,
            "vix_above_25": (close > 25).astype(int).values,
            "vix_above_30": (close > 30).astype(int).values,
        }).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_extended_asian_indices(start_date: Optional[str] = None, end_date: Optional[str] = None,
                                  period: str = "1y") -> pd.DataFrame:
    """Hang Seng (^HSI) + Sensex (^BSESN) + Nifty (^NSEI).

    Returns DataFrame with columns:
      date, hsi_close, hsi_change, hsi_open_gap,
            bsesn_close, bsesn_change, bsesn_open_gap,
            nsei_close, nsei_change, nsei_open_gap,
            extended_asian_avg_return, extended_asian_risk_off
    """
    if not YF_OK:
        return pd.DataFrame()
    tickers = {"^HSI": "hsi", "^BSESN": "bsesn", "^NSEI": "nsei"}
    raw: Dict[str, pd.DataFrame] = {}
    for tk in tickers:
        try:
            kw = dict(progress=False, auto_adjust=True)
            if start_date and end_date:
                d = yf.download(tk, start=start_date, end=end_date, **kw)
            else:
                d = yf.download(tk, period=period, **kw)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            raw[tk] = d
        except Exception:
            raw[tk] = pd.DataFrame()

    nonempty = [df for df in raw.values() if not df.empty]
    if not nonempty:
        return pd.DataFrame()

    base_index = max(nonempty, key=len).index
    out = pd.DataFrame({"date": base_index})
    for tk, slug in tickers.items():
        df = raw[tk]
        if df.empty:
            out[f"{slug}_close"] = np.nan
            out[f"{slug}_change"] = np.nan
            out[f"{slug}_open_gap"] = np.nan
            continue
        aligned = df.reindex(base_index, method="ffill")
        close = aligned["Close"].astype(float)
        open_ = aligned["Open"].astype(float)
        out[f"{slug}_close"] = close.values
        out[f"{slug}_change"] = close.pct_change().values
        out[f"{slug}_open_gap"] = ((open_ - close.shift(1)) / close.shift(1)).values

    chg_cols = [f"{slug}_change" for slug in tickers.values()]
    out["extended_asian_avg_return"] = out[chg_cols].mean(axis=1, skipna=True)
    drops = (out[chg_cols].fillna(0) < -0.02).sum(axis=1)
    out["extended_asian_risk_off"] = (drops / max(1, len(chg_cols))).clip(0.0, 1.0)
    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# GDELT — tone + volume for Pakistan-region conflict topics (KEYLESS)
# ─────────────────────────────────────────────────────────────────────────────

GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

# Topic packs designed for PSX-relevant geopolitical tone tracking.
# Each pack is a GDELT query string. Avoid super-broad terms that drown in noise.
GDELT_TOPIC_PACKS: Dict[str, str] = {
    "pk_conflict": "(pakistan OR kashmir OR balochistan) (attack OR strike OR clash OR military OR violence)",
    "regional_war": "(iran OR israel OR \"middle east\") (war OR strike OR escalation OR conflict)",
    "energy_crisis": "(\"strait of hormuz\" OR \"oil supply\" OR \"opec\" OR \"crude oil\") (disruption OR shortage OR surge OR crisis)",
    "global_riskoff": "(recession OR \"emerging markets\" OR \"capital flight\" OR sanctions OR \"trade war\")",
}


def fetch_gdelt_topic_timeline(topic: str, query: str, timespan: str = "30d",
                                cache_ttl_seconds: int = 3 * 3600) -> pd.DataFrame:
    """Fetch BOTH tone and volume daily timelines for a GDELT query.

    Returns DataFrame with columns: date, gdelt_{topic}_tone, gdelt_{topic}_vol
    Empty DataFrame on rate-limit or upstream failure.
    """
    cache_name = f"gdelt_{topic}_{timespan}.json"
    cached = _read_cache(cache_name, cache_ttl_seconds)
    if cached is not None:
        try:
            return pd.DataFrame(cached["rows"])
        except Exception:
            pass

    encoded_q = urllib.parse.quote(query)
    tone_url = f"{GDELT_BASE}?query={encoded_q}&mode=timelinetone&timespan={timespan}&format=json"
    vol_url = f"{GDELT_BASE}?query={encoded_q}&mode=timelinevol&timespan={timespan}&format=json"

    tone = _http_get_json(tone_url, retries=3, backoff=2.0)
    # GDELT rate-limits aggressively — pause between calls to the same host.
    time.sleep(2.5)
    vol = _http_get_json(vol_url, retries=3, backoff=2.0)

    def _series(payload: Optional[dict]) -> Dict[str, float]:
        if not payload or not payload.get("timeline"):
            return {}
        data = payload["timeline"][0].get("data", [])
        out = {}
        for pt in data:
            ds = pt.get("date", "")
            # GDELT format: 20260410T000000Z
            if not ds or len(ds) < 8:
                continue
            try:
                d = datetime.strptime(ds[:8], "%Y%m%d").date()
                out[d.isoformat()] = float(pt.get("value", 0.0) or 0.0)
            except Exception:
                continue
        return out

    tone_series = _series(tone)
    vol_series = _series(vol)
    all_dates = sorted(set(tone_series) | set(vol_series))

    rows = []
    for d in all_dates:
        rows.append({
            "date": d,
            f"gdelt_{topic}_tone": tone_series.get(d),
            f"gdelt_{topic}_vol": vol_series.get(d),
        })

    _write_cache(cache_name, {"fetched_at": datetime.utcnow().isoformat(), "rows": rows})

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch_all_gdelt_topics(timespan: str = "30d") -> pd.DataFrame:
    """Fetch all topic packs and merge into one date-indexed DataFrame.

    Inter-topic delay is intentional to stay under GDELT's per-IP rate limit.
    """
    merged: Optional[pd.DataFrame] = None
    for topic, query in GDELT_TOPIC_PACKS.items():
        df = fetch_gdelt_topic_timeline(topic, query, timespan=timespan)
        if df.empty:
            continue
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="date", how="outer")
        time.sleep(3.0)  # be kind to GDELT
    if merged is None:
        return pd.DataFrame()
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# USD/PKR — Pakistan currency weakness leads PSX crashes (Agent 2 finding:
# 4–12-week lead for EM-originated shocks, cleanest Pakistan-specific signal).
# ─────────────────────────────────────────────────────────────────────────────

def fetch_usd_pkr(start_date: Optional[str] = None, end_date: Optional[str] = None,
                  period: str = "1y") -> pd.DataFrame:
    """USD/PKR exchange rate (yfinance PKR=X). Higher = PKR weaker = bearish.

    Returns DataFrame columns: date, usdpkr_close, usdpkr_change,
    usdpkr_zscore_30d, usdpkr_zscore_90d, usdpkr_weakening_streak.
    """
    if not YF_OK:
        return pd.DataFrame()
    try:
        kw = dict(progress=False, auto_adjust=True)
        if start_date and end_date:
            data = yf.download("PKR=X", start=start_date, end=end_date, **kw)
        else:
            data = yf.download("PKR=X", period=period, **kw)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty:
            return pd.DataFrame()
        close = data["Close"].astype(float)
        change = close.pct_change()
        z30 = (close - close.rolling(30).mean()) / close.rolling(30).std()
        z90 = (close - close.rolling(90).mean()) / close.rolling(90).std()
        weakening = (change > 0).astype(int)
        streak = weakening.rolling(5).sum()  # how many of last 5 days PKR weakened
        df = pd.DataFrame({
            "date": data.index,
            "usdpkr_close": close.values,
            "usdpkr_change": change.values,
            "usdpkr_zscore_30d": z30.values,
            "usdpkr_zscore_90d": z90.values,
            "usdpkr_weakening_streak": streak.values,
        }).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Brent & WTI for Hormuz-shock confirmation
# Lightweight wrappers around Yahoo BZ=F / CL=F; keyed on date like everything else.
# ─────────────────────────────────────────────────────────────────────────────

def fetch_crude_prices(period: str = "6mo") -> pd.DataFrame:
    """Brent + WTI close prices with 1d / 5d returns, for Hormuz-shock confirmation."""
    if not YF_OK:
        return pd.DataFrame()
    out = {"date": None}
    for tk, slug in (("BZ=F", "brent"), ("CL=F", "wti")):
        try:
            d = yf.download(tk, period=period, progress=False, auto_adjust=True)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            if d.empty:
                continue
            close = d["Close"].astype(float)
            if out["date"] is None:
                out["date"] = d.index
            out[f"{slug}_close"] = close.reindex(out["date"], method="ffill").values
            out[f"{slug}_change_1d"] = close.reindex(out["date"]).pct_change().values
            out[f"{slug}_change_5d"] = (close.reindex(out["date"]) / close.reindex(out["date"]).shift(5) - 1).values
        except Exception:
            continue
    if out["date"] is None:
        return pd.DataFrame()
    return pd.DataFrame(out).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# USGS earthquakes — Pakistan region
# ─────────────────────────────────────────────────────────────────────────────

USGS_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def fetch_pakistan_region_quakes(min_magnitude: float = 5.0,
                                  lookback_days: int = 90,
                                  cache_ttl_seconds: int = 6 * 3600) -> pd.DataFrame:
    """Significant earthquakes (M>=min) within a bounding box covering Pakistan
    + immediate neighbors. Returns: date, quake_count, max_magnitude.
    """
    cache_name = f"usgs_pk_M{min_magnitude}_{lookback_days}d.json"
    cached = _read_cache(cache_name, cache_ttl_seconds)
    if cached is not None:
        try:
            return pd.DataFrame(cached["rows"])
        except Exception:
            pass

    end = datetime.utcnow().date()
    start = end - timedelta(days=lookback_days)
    params = {
        "format": "geojson",
        "starttime": start.isoformat(),
        "endtime": end.isoformat(),
        "minmagnitude": str(min_magnitude),
        # Pakistan + neighbors bounding box
        "minlatitude": "23",
        "maxlatitude": "37",
        "minlongitude": "60",
        "maxlongitude": "78",
    }
    url = f"{USGS_BASE}?{urllib.parse.urlencode(params)}"
    payload = _http_get_json(url, retries=2)
    if not payload or "features" not in payload:
        return pd.DataFrame()

    by_day: Dict[str, Dict[str, float]] = {}
    for feat in payload["features"]:
        try:
            props = feat.get("properties", {})
            ts_ms = props.get("time")
            mag = float(props.get("mag", 0.0))
            d = datetime.utcfromtimestamp(ts_ms / 1000.0).date().isoformat()
            slot = by_day.setdefault(d, {"quake_count": 0, "max_magnitude": 0.0})
            slot["quake_count"] += 1
            slot["max_magnitude"] = max(slot["max_magnitude"], mag)
        except Exception:
            continue

    rows = [{"date": d, **v} for d, v in sorted(by_day.items())]
    _write_cache(cache_name, {"fetched_at": datetime.utcnow().isoformat(), "rows": rows})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Public convenience: build a "world-monitor enrichment" DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def build_worldmonitor_features(start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  period: str = "1y") -> pd.DataFrame:
    """One-shot fetch of every keyless signal, merged on `date`.

    Used by `external_features.merge_external_features` as a single new merge
    block. Returns empty DataFrame if everything fails (graceful no-op).
    """
    # Lazy import to avoid circular dep at module load.
    from backend.external_features import _to_naive_datetime

    parts: List[pd.DataFrame] = []
    vix = fetch_vix(start_date=start_date, end_date=end_date, period=period)
    if not vix.empty:
        vix["date"] = _to_naive_datetime(vix["date"])
        parts.append(vix)

    asian = fetch_extended_asian_indices(start_date=start_date, end_date=end_date, period=period)
    if not asian.empty:
        asian["date"] = _to_naive_datetime(asian["date"])
        parts.append(asian)

    if not parts:
        return pd.DataFrame()

    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)
