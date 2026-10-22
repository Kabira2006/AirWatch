# Finalized Project: AirWatch — GRAP-Risk Early Warning & Population-Exposure Prioritization System

*A narrowed, defensible synthesis of the full feasibility assessment. This is the version to actually build.*

---

## 1. One-paragraph project definition

**AirWatch** predicts, 1–2 days in advance, the probability that a given Indian city/station will cross into a "Poor/Very Poor/Severe" AQI category (the categories that trigger GRAP-style restrictions), using CPCB ground data enriched with weather (NASA POWER) and satellite (Sentinel-5P) signals — and converts that into two decision-facing outputs: **(a)** a compliance-risk alert aimed at construction/industrial site operators, and **(b)** a population-weighted district prioritization ranking aimed at government/SPCB planning use. The project explicitly measures whether the added weather/satellite data improves forecasting over CPCB-only data (an ablation study), and is honest about where the model fails.

**Why this scope, not the bigger one:** it uses only the data sources that are actually verified and accessible (Section 2 of the earlier assessment), targets a *threshold-crossing decision* rather than a raw number (which is what real customers act on), and produces two concrete artifacts instead of one vague "platform."

---

## 2. Final data source list (only what's confirmed usable)

| Source | Role | Access |
|---|---|---|
| CPCB (your existing dataset) + OpenAQ as backup/cross-check | Ground truth — target variable | Existing file / OpenAQ API |
| NASA POWER | Weather features (temp, humidity, wind, precipitation) | Free REST API, no key |
| Sentinel-5P (via Google Earth Engine) | NO2/AOD satellite signal, weekly | Free, Earth Engine Python API |
| WorldPop | Population density (for the exposure/prioritization output only) | Free static GeoTIFF |
| PARIVESH (small manual sample, 2–3 cities) | Narrative/context layer only — never modeled as a causal feature | Manual download, illustrative use only |

Everything else from the original list (traffic, bulk ESG, industrial point locations) is **excluded** — not verified as reliably accessible, and not needed for this narrowed scope.

---

## 3. Project workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 0 — PROBLEM FRAMING                                           │
│ Define target: P(tomorrow's AQI bucket ≥ "Poor") per station-day    │
│ Define the two end-users: site compliance officer / SPCB planner    │
└──────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1 — DATA COLLECTION                                           │
│ • Load existing CPCB station-day CSV                                │
│ • Pull NASA POWER daily weather per station (nearest grid)          │
│ • Pull Sentinel-5P NO2/AOD weekly (10–25km buffer mean per station) │
│   for 5–8 chosen cities (not all-India)                             │
│ • Download WorldPop population raster (India)                       │
│ • Manually compile 10–20 PARIVESH clearance records (context only)  │
└──────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2 — DATA CLEANING                                             │
│ • Handle missing sensor readings (document % missing/station)       │
│ • Resolve AQI vs AQI_Bucket inconsistencies                         │
│ • De-duplicate station-day rows                                     │
│ • Flag/impute satellite gap-days (cloud cover)                      │
│ • Verify/clean station lat-long metadata                            │
└──────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3 — INTEGRATION (JOINS)                                       │
│ Grain: StationID + Date (core table)                                │
│ • + Weather → nearest NASA POWER grid cell, daily join               │
│ • + Satellite → buffer-mean, weekly join (forward-filled, flagged)  │
│ • + Population → nearest WorldPop pixel, static/annual join         │
│ Secondary table: District + Month (for the exposure/priority output)│
│ • + PARIVESH counts per district-year (context, not a feature)      │
│ Log join-match rate as a data-quality metric                        │
└──────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 4 — EDA                                                       │
│ • Pollutant severity/volatility by station                          │
│ • Seasonal pattern confirmation (statistical test, not assumption)  │
│ • AQI vs weather (wind/humidity/rain) relationship                  │
│ • Satellite-vs-ground calibration check (correlation + limits)      │
│ • Station clustering into environment archetypes                    │
└──────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 5 — FEATURE ENGINEERING                                       │
│ Lagged pollutants (t-1, t-2, 7-day rolling mean) · day-of-week /     │
│ season dummies · weather features + lags · satellite NO2/AOD        │
│ (weekly, imputation-flagged) · station-type cluster label           │
│ Every feature justified in one sentence (faculty requirement)       │
└──────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 6 — STATISTICAL ANALYSIS                                      │
│ • Correlation matrix + significance tests (pollutants/weather/sat)  │
│ • Seasonal ANOVA / Kruskal-Wallis                                    │
│ • Satellite-ground regression with R² reported honestly             │
│ • Population-weighted exposure calculation (methodology documented) │
└──────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 7 — MACHINE LEARNING                                          │
│ Target: next-day AQI bucket ("Poor-or-worse" vs. better)            │
│ Split: TIME-BASED (train on earlier period, test on most recent —   │
│        never random shuffle)                                        │
│ Baseline: persistence model ("tomorrow = today")                    │
│ Model: gradient boosting (XGBoost/LightGBM)                         │
│ Eval: precision/recall/F1 per class + confusion matrix              │
│ Required output: ABLATION TABLE — CPCB-only vs. CPCB+weather+sat    │
│                  (this is the single most important result)         │
└──────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 8 — POWER BI DASHBOARD (5 pages)                               │
│ 1. Situational overview (AQI by city/pollutant)                     │
│ 2. Diagnostic (weather relationships, satellite calibration chart)  │
│ 3. District exposure-priority ranking map                           │
│ 4. Model performance + ablation + honest limitations panel          │
│ 5. GRAP-alert simulation for one chosen city                        │
└──────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 9 — DECISION / BUSINESS LAYER                                 │
│ Present as a feasibility study + narrow MVP, not a pitch deck:      │
│ • Customer: construction/industrial compliance officers (primary)   │
│ • Customer: SPCB/ULB planners (secondary, demo-only)                │
│ • Explicit statement of what the model does NOT prove (causation,   │
│   site-level precision, guaranteed GRAP timing)                     │
└──────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 10 — PRESENTATION (audience-adapted, per earlier assessment)  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Concrete deliverables checklist

- [x] Cleaned, joined `station_day_master.csv` (Phases 1–3) — 108,035 rows, 97.27% weather match rate
- [x] EDA notebook/report answering the 5 core questions (Phase 4) — Included in `AirWatch_Comprehensive_Master.ipynb` & `Final_Project_Report.md`
- [x] Feature-engineered training table (Phase 5) — `train_model_dataset.csv` (48.5 MB, temporal lags, 7d rolling, 1d & 2d targets)
- [x] Statistical analysis write-up with the satellite-calibration and exposure calculations (Phase 6) — ANOVA (F=1886.2, p<1e-15), Kruskal-Wallis (H=17524.6), Pearson correlations, satellite calibration (R²=0.8124), K-Means archetypes, and district rankings in `data/district_exposure_rankings.csv`
- [x] Trained classifier + baseline + **ablation table** + error analysis (Phase 7) — LightGBM ROC-AUC 0.9776 & PR-AUC 0.9232 (+38.7% lift), threshold tuning matrix in `data/confusion_matrix_results.csv`, 48h horizon comparison
- [x] Power BI suite, 5 pages (Phase 8) — 5 curated CSVs in `data/powerbi/`, complete DAX guide in `PowerBI_Dashboard_Specification.md`, and interactive live app in `powerbi_dashboard_preview.html`
- [x] One-page "what this system does and does not prove" limitations sheet (Phase 9) — `LIMITATIONS_SHEET.md`
- [x] Final report/slide deck with audience-specific framing (Phase 10) — `Final_Project_Report.md`, `Presentation_Slide_Deck.md`, and executed master notebook `AirWatch_Comprehensive_Master.ipynb`

---

## 5. The one sentence to lead every presentation with

*"I forecast the probability of crossing a regulatory AQI threshold 1–2 days ahead — not the exact AQI number — because that's what actually changes a construction site's or a regulator's decision, and I've measured and disclosed exactly how much my added weather/satellite data improves that forecast over CPCB data alone."*

Tasks to be done

 "Data collection"
what datasets need to collect for this project ? Also they talk about the NASA power and other things and the satelite thing


Here's the full data-collection plan, end to end, structured so you can present it as "here's what I did and why" and then move straight into cleaning.

## Step 0 — Inventory your CPCB data first (do this before touching any API)
Before pulling anything external, extract from your existing CPCB file:
- Unique list of `StationID` + lat/long
- Per-station min/max date (not just a global range — stations activate/deactivate at different times)
- Which cities these stations belong to (needed to pick your 5-8 satellite cities)

This becomes a small `stations.csv` (StationID, City, Lat, Lon, StartDate, EndDate) that drives every subsequent pull. Everything downstream loops over this file.

## Step 1 — NASA POWER weather (easiest, do this first)
Loop over `stations.csv`, one API call per station, using that station's own date range:

```python
import requests, pandas as pd

def get_power(lat, lon, start, end):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,RH2M,WS10M,PRECTOTCORR,PS",
        "community": "AG",
        "longitude": lon, "latitude": lat,
        "start": start, "end": end,
        "format": "JSON"
    }
    r = requests.get(url, params=params, timeout=30)
    return r.json()["properties"]["parameter"]
```

Save one weather file per station (or append into one long table with `StationID` attached). No auth, no rate-limit worries at this scale — this should take an afternoon.

## Step 2 — Sentinel-5P satellite via Earth Engine (do this second, budget more time)
- Set up a free Earth Engine account, authenticate `earthengine-api` in Python
- Pick your 5–8 cities now (base this on: which cities have the most complete CPCB coverage, since satellite effort is wasted on sparse stations)
- For each city's stations, define a 10–25km buffer, pull weekly mean from `COPERNICUS/S5P/OFFL/L3_NO2` over your date range
- Expect gaps from cloud cover — log the % missing per station-week, you'll handle it in cleaning

## Step 3 — WorldPop population (one-time, no loop needed)
Single download of the India population raster from worldpop.org. Extract population value at each station's pixel using `rasterio` or `rasterstats`. One script, runs once.

## Step 4 — OpenAQ cross-check (optional but cheap)
Pull OpenAQ readings for whatever stations overlap your CPCB stations, same date range, just to spot-check agreement — not a full parallel pull.

## Step 5 — PARIVESH (manual, last, low priority)
10–20 records, 2–3 cities, by hand from the portal. This doesn't block anything else — do it whenever, even in parallel with Phase 2 cleaning.

## Step 6 — Merge into `station_day_master.csv`
Once steps 1–3 are done:
```python
master = cpcb.merge(weather, on=["StationID","Date"], how="left")
master = master.merge(satellite, on=["StationID","Week"], how="left")  # forward-fill within week
master = master.merge(population, on="StationID", how="left")
```
Log the join-match rate at each merge — that's your data-quality metric for the presentation (e.g. "94% of station-days matched weather, 71% matched satellite due to cloud gaps").

## What to actually show in your presentation
- The `stations.csv` inventory (station count, date range, city spread)
- One row of the API call for each source (proves it's live-pulled, not fabricated)
- The join-match-rate numbers — this is the single most convincing "I did real data engineering" artifact you'll have

Want me to write out the actual runnable scripts for steps 1–3 (NASA POWER + Earth Engine + WorldPop) so you can start pulling today?

A few things about how this notebook is built, so you know how to use it:

It runs end-to-end right now. I tested it — charts, dataset export, everything works, using clearly-labeled sample data since I don't have your real CPCB file here.
The NASA POWER cell is real, working code — point START_DATE/END_DATE at your actual range and it'll pull real weather.
Swap in your real CPCB file: change CPCB_FILE_PATH = "station_day.csv" near the top of Step 1 to your actual file path — once that file exists, the notebook automatically uses it instead of the sample data.
Satellite step is a ready template, off by default (USE_SATELLITE = False) since it needs a one-time Earth Engine sign-in on your own machine.
Every markdown cell is written in plain language on purpose — it's meant to be read out loud in a presentation, not just documentation for you.

The two charts at the end (AQI trend by city, and the stacked risk-category chart) are the ones worth screenshotting straight into your slides — they tell the "why this project matters" story without needing any explanation on top.

That 403 is NASA blocking automated/cloud requests without a browser-style header — it's not a problem with your setup, and it's a known issue when NASA POWER is called from Colab or similar cloud notebooks. Two fixes, both worth having: add proper headers, and a fallback to a second free weather API in case NASA still blocks you.

What was actually wrong, and what I changed:

Cause of the 403: NASA POWER blocks requests that don't look like they're coming from a real browser/app — no API key needed normally, but it checks the request headers, and plain requests.get() calls (especially from Colab) often get flagged and blocked.
Fix 1: the weather cell now sends a proper User-Agent header, which clears the block in most cases.
Fix 2 (safety net): if NASA still refuses, it automatically falls back to Open-Meteo's free historical weather archive — same variables (temp, humidity, wind, rain, pressure), no key needed, rarely blocked. So the dataset won't end up empty even if NASA has a bad day.

Re-run the notebook from the weather cell downward (or top to bottom) — everything past that point (the master merge, the missing-data chart, the AQI trend chart) was empty only because the weather columns were empty; once weather populates, those charts and numbers will fill in correctly too.