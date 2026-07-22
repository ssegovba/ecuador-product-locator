from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Lightweight, deploy-committed inputs the app consumes — built by
# utils/build_product_locator_app_data.py (slim REEM tables) and .../build_product_locator_trade.py
# (trade). Living under summary_tables/ keeps the app self-contained on Streamlit Cloud:
# no gitignored microdata, no data/raw/, no geopandas at runtime.
APP_DATA_DIR = DATA_DIR / "processed" / "summary_tables" / "product_locator"
REEM_PROCESSED_PATH = APP_DATA_DIR / "reem_empresas_slim.parquet"
REEM_ESTAB_PROCESSED_PATH = APP_DATA_DIR / "reem_establecimientos_slim.parquet"
PROVINCES_GEOJSON_PATH = APP_DATA_DIR / "provinces.geojson"

CIIU_PATH = DATA_DIR / "intermediate" / "classifications" / "ciiu_rev4dot1.csv"
HS_REFERENCE_PATH = DATA_DIR / "intermediate" / "classifications" / "hs_reference_comtrade.csv"
ISIC_DETAILED_JSON_PATH = DATA_DIR / "processed" / "isic" / "isic_rev4_detailed.json"

# HS edition (Comtrade naming: H0 = HS 1992 ... H6 = HS 2022) -> ISIC4 crosswalk.
# Restricted to the editions for which Atlas trade data exists (HS92/HS12/HS22),
# so the Page-1 export scaling always matches the selected classification.
CROSSWALK_DIR = DATA_DIR / "intermediate" / "crosswalk"
HS_EDITIONS: dict[str, Path] = {
    "2022": CROSSWALK_DIR / "HS22_ISIC4_all.csv",
    "2012": CROSSWALK_DIR / "HS12_ISIC4_all.csv",
    "1992": CROSSWALK_DIR / "HS92_ISIC4_all.csv",
}
DEFAULT_HS_EDITION = "2022"

# Compact Ecuador export tables (built by utils/build_product_locator_trade.py),
# one per HS edition, for the Page-1 export scaling.
EC_TRADE_DIR = APP_DATA_DIR
EC_TRADE_PATHS: dict[str, Path] = {
    edition: EC_TRADE_DIR / f"ec_trade_{edition}.csv" for edition in HS_EDITIONS
}

# Province profile (Page 3) summary tables.
SUMMARY_DIR = DATA_DIR / "processed" / "summary_tables"
MULTISOURCE_PATH = SUMMARY_DIR / "multisource" / "provincial_multisource_latest.csv"
MULTISOURCE_RANKED_PATH = SUMMARY_DIR / "multisource" / "provincial_multisource_latest_ranked.csv"
GVA_PROVINCIAL_PATH = SUMMARY_DIR / "gva" / "gva_provincial.csv"
GVA_PROVINCE_INDUSTRY_PATH = SUMMARY_DIR / "gva" / "gva_province_industry.csv"
CENSUS_SUMMARY_PATH = SUMMARY_DIR / "census" / "census_provincial_summary.csv"
ENEMDU_LABOR_SUMMARY_PATH = SUMMARY_DIR / "labor_market" / "enemdu_provincial_labor_summary.csv"

# Relatedness / economic-complexity exports (Page 4). The `relatedness_density`
# notebook is the single source of truth; the app reads its precomputed tables
# and never recomputes proximity/density from microdata.
RELATEDNESS_DIR = SUMMARY_DIR / "relatedness_density"
TRADABILITY_PATH = DATA_DIR / "intermediate" / "classifications" / "isic_tradability_ai.csv"
# Local-relevance decoration for Page-4 opportunity bars (industry_tradability notebook).
INDUSTRY_TRADABILITY_DIR = SUMMARY_DIR / "industry_tradability"
LOCAL_RELEVANCE_PATH = (
    INDUSTRY_TRADABILITY_DIR / "province_industry_mean_relevance_geo_province_concen_plazas.csv"
)
TRADABILITY_PCTILE_PATH = INDUSTRY_TRADABILITY_DIR / "industry_tradability_classification.csv"
DENSITY_BASE = "estab_count"  # primary base for the app (plazas is a notebook robustness base)
# Unclassified / special CIIU codes: dropped from every Page-4 surface.
EXCLUDED_CLASSES = {"0000", "9800", "9999"}
# Off-diagonal proximity mass (phi_mass_noself) below which a class's no-self density is
# noise (near-zero denominator). Flagged, never dropped (see the settled no-self decisions).
THIN_NEIGHBORHOOD_THRESHOLD = 0.05

APP_TITLE = "Ecuador Product Location Explorer"
APP_CAPTION = (
    "Identify where in Ecuador a product could plausibly be produced, "
    "based on the presence of related industries and economic indicators."
)
DEFAULT_EMPLOYMENT_BASE = "empleo_equiv"
EMPLOYMENT_BASE_LABELS = {
    "empleo_equiv": "Equivalent employment",
    "plazas_equiv": "Equivalent positions",
}

# Tile map settings (plotly MapLibre traces).
MAP_STYLE = "carto-positron"
MAP_CENTER = {"lat": -1.6, "lon": -82.5}  # midpoint keeping Galapagos in view
MAP_ZOOM = 5.0
MAP_HEIGHT = 650

# Page 2 choropleth metric registry: column -> (label, colorscale, is_share).
METRIC_OPTIONS: dict[str, dict[str, object]] = {
    "enterprises": {"label": "Enterprises", "colorscale": "OrRd", "is_share": False},
    "enterprises_share_country": {
        "label": "Share of enterprises (national)",
        "colorscale": "OrRd",
        "is_share": True,
    },
    "establishments": {
        "label": "Establishments",
        "colorscale": "Purples",
        "is_share": False,
    },
    "establishments_share_country": {
        "label": "Share of establishments (national)",
        "colorscale": "Purples",
        "is_share": True,
    },
    "exporting_enterprises": {
        "label": "Exporting enterprises (net)",
        "colorscale": "Greens",
        "is_share": False,
    },
    "exporters_share_country": {
        "label": "Share of exporting enterprises",
        "colorscale": "Greens",
        "is_share": True,
    },
    "employment": {"label": "Employment", "colorscale": "Blues", "is_share": False},
    "employment_share_country": {
        "label": "Share of employment (national)",
        "colorscale": "Blues",
        "is_share": True,
    },
}

# House style colors (see CLAUDE.md / explore_ecuador_boundaries.ipynb).
COLOR_NATIONAL = "#00008B"
COLOR_FOCUS = "#8B0000"
COLOR_OTHER_BAR = "#7F7F7F"
COLOR_OTHER_LINE = "#EDE7D9"
SECTOR_COLORS = {
    "Primario": "#2E7D32",
    "Secundario": "#B45309",
    "Terciario": "#1D4ED8",
}

# Page 4 relatedness "state" (RCA-defined margin) colors — the GL brand
# green/yellow/blue used by the relatedness notebook (good mutual contrast):
# specialized (core) / present-not-specialized (foothold) / absent (new entry).
STATE_COLORS = {
    "specialized": "#48c0a2",  # GL brand green (teal)
    "present_not_spec": "#e5bd4f",  # GL brand yellow
    "absent": "#6db5db",  # GL brand blue
}
STATE_LABELS = {
    "specialized": "Specialized (RCA ≥ 1, core)",
    "present_not_spec": "Present, not specialized",
    "absent": "Absent (new entry)",
}
# Sequential colorscale for the density choropleth.
DENSITY_COLORSCALE = "Blues"
# Outline for M=1 ("already present") provinces on the density choropleth — GL
# brand red, bright enough to stay legible over the dark end of the Blues scale.
COLOR_PRESENT_OUTLINE = "#ee3e4c"
