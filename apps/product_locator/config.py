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
ISIC_DETAILED_JSON_PATH = DATA_DIR / "processed" / "isic" / "isic_rev4_detailed.json"

SUMMARY_DIR = DATA_DIR / "processed" / "summary_tables"

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
# National attractiveness metrics per CIIU class (Page 4): average wage paid,
# average jobs per firm, total employment — exported by
# notebooks/industry_labor_profile.ipynb. National only: REEM books enterprise
# variables to the HQ province, so no provincial variant exists.
LABOR_PROFILE_PATH = SUMMARY_DIR / "industry_labor_profile" / "class_labor_profile.csv"
# Establishment-booked employment per geography x CIIU level (same notebook):
# the provincial counterpart of plazas_equiv_estab_nat — employment counted
# where units operate, so it is comparable with the national figure.
LABOR_COMPOSITION_PATH = SUMMARY_DIR / "industry_labor_profile" / "industry_labor_composition.csv"
# Unclassified / special CIIU codes, dropped by every loader that reads them.
# **Not the complete list of special codes**: the relatedness density export also carries
# `9900`, which is deliberately left out of this set. It reaches no surface anyway — every
# page joins density against `class_labor_profile.csv` with an inner merge and `9900` has no
# row there — so adding it would change the density loader's row count (413 -> 412) to no
# visible effect. Named here because the set's membership is otherwise easy to misread as
# exhaustive (review note, 2026-08-21).
EXCLUDED_CLASSES = {"0000", "9800", "9999"}
# Off-diagonal proximity mass (phi_mass_noself) below which a class's no-self density is
# noise (near-zero denominator). Flagged, never dropped (see the settled no-self decisions).
THIN_NEIGHBORHOOD_THRESHOLD = 0.05

# Bottom-up page (province diversification explorer).
DEFAULT_BOTTOM_UP_PROVINCE = "17"  # Pichincha; '99' (national) is offered but never the default

# ---------------------------------------------------------------------------
# The bottom-up lens's shared control defaults.
#
# Defined **once, here** rather than as literals in each view, because the lens has
# two surfaces that must agree: the explorer owns the widgets, and *Candidate
# industries x provinces* replays the explorer's ranking for all 24 provinces and
# therefore **inherits** whatever the reader set there (falling back to these when
# the reader has not opened the explorer yet). A second copy of the literals is
# exactly how the two pages came to disagree before v4 -- the explorer shipping the
# menu off and k = 5 while the builder, the grid and the committed CSVs all used the
# menu on and k = 10, with nothing on screen to say the difference was a default.
#
# `BU_DEFAULT_TOP_K = 10` is `build_province_candidates.py`'s `BU_TOP_K`: the pool the
# four committed CSVs are cut from.
BU_DEFAULT_TRADABLE = True
BU_DEFAULT_SUPPORT = True
BU_DEFAULT_CONNECTED = True
BU_DEFAULT_TOP_K = 10
BU_DEFAULT_BETA = 0.5
BU_DEFAULT_WEIGHT = 0.5
# The explorer's widget keys, in the order the grid's "inheriting from" caption names
# them. Keyed by the setting the grid reads, so the caption and the lookup cannot
# drift apart.
BU_CONTROL_KEYS = {
    "tradable": "bu-tradable",
    "support": "bu-support",
    "connected": "bu-connected",
    "top_k": "bu-k",
    "beta": "bu-beta",
    "weight": "bu-weight",
}
# Support guard for a (province, industry) cell, shared by **both lenses** — the
# **AND** of the two floors below. A cell survives on either enough units or enough
# workers and is dropped only when both are thin: one 1,630-FTE operation is
# evidence, and so are five one-person firms. See MIN_ESTAB / MIN_PLAZAS.
#
# The bottom-up lens used establishments alone until 2026-08-20; it now uses the
# same two-dimensional guard as the top-down lens so the two are comparable (and so
# the planned potential-gap analysis inherits one definition rather than two).
RANKING_MIN_ESTAB = 5

# The extractive carve-out of the bottom-up page's tradable **menu**. The menu is
# `tradable_regional == 'tradable'` in TRADABILITY_PATH minus these classes:
# regional tradability asks "can the output leave the province?", to which a barrel
# of crude or a truck of quarried stone says yes — but where it can be produced is
# geology, not capability, so recommending it as a diversification target is
# meaningless. The set is **all of CIIU section B** (mining and quarrying, 0510-0899
# plus the two support classes 0910/0990) **plus A0170** hunting and trapping.
#
# Deliberately kept in the menu (Santiago, 2026-08-16): **A0220 logging** and
# **A0311/A0312 capture fishing** — renewable-stock harvest with genuine Ecuadorian
# capability clusters behind it (balsa, the tuna fleet), unlike geology-bound mining.
#
# 15 codes here, 14 of them in the app's 412-class universe — 0892 (extraction of
# peat) has no row in any app export. Defined as the full section, not as the 14 that
# happen to be present, so a data refresh cannot silently re-admit peat mining.
EXTRACTIVE_CLASSES = {
    "0170",  # Hunting, trapping and related service activities
    "0510", "0520",  # Mining of hard coal / lignite
    "0610", "0620",  # Extraction of crude petroleum / natural gas
    "0710", "0721", "0729",  # Mining of iron / uranium-thorium / other non-ferrous ores
    "0810", "0891", "0892", "0893", "0899",  # Quarrying and other mining
    "0910", "0990",  # Support activities for extraction
}

# The `entry_regime` vocabulary (isic_tradability_ai.csv, added 2026-08-25). The axis
# asks whether a province can *act on* an industry, which regional tradability does
# not: a regional hospital's output travels, but its siting is a ministry decision.
#   open_market       - a private entrant can cause it to appear or grow there
#   state_allocated   - a state planning, budget or siting decision puts it there
#   regulated_network - entry is a national licence/charter/frequency, and provincial
#                       presence is a branch or network node of a national operator
# Only `open_market` survives into the bottom-up menu (an `.eq('open_market')`
# inclusion in data_loader.load_tradable_menu, so a typo'd label would silently
# drop its class from the menu). Held here as the single definition so the loader
# can fail loudly on any value outside the vocabulary instead of letting a data
# error quietly change what the app recommends.
# See data/intermediate/classifications/isic_entry_regime_review.md.
ENTRY_REGIME_VALUES = frozenset({"open_market", "state_allocated", "regulated_network"})

# --------------------------------------------------------------------------- #
# Top-down lens — the 50 HS4 diversification opportunities.                    #
# --------------------------------------------------------------------------- #

# The opportunity -> industry map (one row per opportunity x route x industry),
# built by utils/build_oportunidades_industrias.py from the upstream analysis.
OPORTUNIDADES_DIR = SUMMARY_DIR / "oportunidades"
OPPORTUNITY_INDUSTRIES_PATH = OPORTUNIDADES_DIR / "oportunidades_industrias.csv"
# Province x CIIU class presence on three bases (industry_space notebook). The
# single presence source for the top-down lens: 24 provinces, **no '99' row**.
PROVINCE_RCA_PATH = SUMMARY_DIR / "industry_space" / "province_class_rca_multibase.csv"

# Provinces this lens never *assigns* an industry to. Galápagos ('20') is one of
# the smallest and most remote provinces, and the parity rule gave it exactly ten
# industries, every one of them manufacturing (furniture, wood, precious metals,
# appliances, concrete, ceramics, metal products, prepared foods, fish processing,
# apparel) — not practical recommendations for an island economy. Its bottom-up
# ranking surfaces air and sea transport services, courier, warehousing and
# tourism-adjacent food instead, so the candidate list reaches it through the
# top-up rather than through here (external feedback, 2026-09-09).
#
# **Assignment only, never the arithmetic.** An excluded province keeps every
# observation it contributes: it stays a column of `metrics.presence_frame`, so it
# still counts in the national employment/establishment denominators and in the
# scale-view parity denominator (1/24). The single place it is applied is
# `metrics.qualifying_provinces` — see that docstring.
TD_EXCLUDED_PROVINCES = frozenset({"20"})

# The disclosure, in one place so Pages A and B cannot word it differently. It has
# to say all three things: that the province is not assigned, that its data is
# still in the measures, and where its candidates do come from.
#
# Shortened 2026-09-10 (Santiago): it no longer names the manufacturing-only result
# or says the map greys it out and the ranking drops it. Both were mechanism the
# reader can see for themselves on the figure in front of them, and the note is
# read on two pages that grey and drop differently. The *reason* it is excluded is
# documented in README.md and the as-built, which is where it belongs.
TD_EXCLUSION_NOTE = (
    "**Galápagos is not assigned industries by this lens.** Its size and remoteness "
    "make a product-based assignment unreliable there. Its establishments and "
    "employment still count in every measure on this page, including the national "
    "totals and the parity bar. Its candidate industries come from the province lens "
    "instead."
)
TD_EXCLUSION_NOTE_ES = (
    "**Este enfoque no asigna industrias a Galápagos.** Su tamaño y lejanía hacen que "
    "una asignación basada en productos no sea confiable ahí. Sus establecimientos y su "
    "empleo siguen contando en todas las medidas de esta página, incluidos los totales "
    "nacionales y la barra de paridad. Sus industrias candidatas provienen del enfoque "
    "de provincia."
)
# Appended to the province's name in the map hover, so a reader who goes looking for
# Galápagos on the map learns why it is grey without leaving the tooltip.
TD_EXCLUDED_MAP_SUFFIX = " (not assigned by this lens)"
TD_EXCLUDED_MAP_SUFFIX_ES = " (no asignada por este enfoque)"

# Card headings. MARGIN_COLORS is derived from STATE_COLORS further down.
# These are UI chrome, not data values: the data carries "intensivo"/"extensivo"
# and these label it, so they translate. Before the language toggle the app had
# only the Spanish pair, which meant the English UI showed "Margen extensivo".
MARGIN_LABELS = {"intensivo": "Margen intensivo", "extensivo": "Margen extensivo"}
MARGIN_LABELS_EN = {"intensivo": "Intensive margin", "extensivo": "Extensive margin"}
# Language-neutral short form for dense labels (selector rows, hover, badges).
# Replaces taking the last word of MARGIN_LABELS, which only produced INT/EXT
# because the label happened to be Spanish.
MARGIN_ABBR = {"intensivo": "INT", "extensivo": "EXT"}

# How much of the score comes from the industries that make the product itself
# vs the industries behind its anchors. Presets, not a slider: the weight moves
# the tail of the ranking rather than its head, so discrete settings read
# honestly as a sensitivity check instead of as a tuned parameter.
# Keys are stable codes, never display text: the preset is persisted in session
# state ("td-route"), so a translated label as the key would either break the
# lookup or orphan the user's pick when the language changes. Labels live in
# ROUTE_PRESET_LABELS and are resolved through a format_func at the widget --
# the same split every other control here uses ("lq"/"scale", "plazas"/"estab").
ROUTE_PRESETS = {"direct": 1.0, "balanced": 0.65, "anchors": 0.0}
DEFAULT_ROUTE_PRESET = "balanced"
ROUTE_PRESET_LABELS = {
    "direct": "Direct only",
    "balanced": "Balanced",
    "anchors": "Anchors only",
}
ROUTE_PRESET_LABELS_ES = {
    "direct": "Solo directa",
    "balanced": "Equilibrada",
    "anchors": "Solo anclas",
}

# Assignment rule: an opportunity goes to every province whose weighted presence
# reaches **parity** — the score a province posts when nothing about it is
# special — rather than to a fixed number of them. In the location-quotient view
# parity is 1.0, the national average, which makes the rule the project's own
# `RCA >= 1` test applied to the product's weighted industry mix. In the scale
# view parity is an equal share of the industry across the provinces (1/n),
# because a weighted average of national shares sums to 1 across all of them and
# so can never approach 1 for any single one. The multiplier below is a
# sensitivity dial on that parity point, not a tuned parameter.
PRESENCE_CUT_CHOICES = [0.5, 0.75, 1.0, 1.5, 2.0]
DEFAULT_PRESENCE_CUT = 1.0

# Support guard for a (province, industry) cell — the **AND** of the two floors.
# A cell survives on either enough units or enough workers, and is dropped only
# when both are thin; one 1,630-FTE operation is evidence, and so are five
# one-person firms. Deliberately not the export's shipped `low_support`
# (establishments alone) nor complexity_outlook's OR combination: those answer
# "is this class's average wage reliable?", which is unstable if *either*
# dimension is thin. Ours asks "is there real productive activity here?".
MIN_ESTAB = RANKING_MIN_ESTAB  # 5 — industry_space's LOW_SUPPORT_ESTAB
MIN_PLAZAS = 50  # complexity_outlook's WAGE_MIN_PLAZAS
# A guard-removed cell holding at least this share of its industry's national
# employment is named in the caption rather than only counted.
MATERIAL_SHARE = 0.10

# Crosswalk-weight floor for scoring and for the Page-B rows. Below it an
# industry is HS4-aggregation residue (aircraft manufacturing at weight 0.000055
# for seats); shown greyed with its weight on Page A, never counted.
MIN_CROSSWALK_WEIGHT = 0.05

# Sequential colorscales for the top-down figures.
PRESENCE_COLORSCALE = "Blues"
ASSIGNMENT_COLORSCALE = "Blues"

# Neutral chrome for the cards, badges and figure furniture. Not palette colors —
# greys and white — but they live here so no view or figure carries a hex literal.
COLOR_TEXT_MUTED = "#777777"  # figure annotations, secondary hover text
COLOR_TEXT_BODY = "#444444"
COLOR_TEXT_STRONG = "#333333"  # gl_design body text: zero lines, reference marks
COLOR_TEXT_ON_FILL = "#ffffff"  # also the marker outline on a filled dot
COLOR_BADGE_BG = "#f2f4f6"
COLOR_BADGE_RULE = "#999999"
# Subtle dark province boundaries: the carto basemap and the low end of every
# colorscale are both near-white, so white borders disappear.
COLOR_BOUNDARY = "#555555"
COLOR_MISSING = "#d9d9d9"  # the "no data" fill on a choropleth
# HQ-flow Sankey nodes: a muted focus red for the HQ province's own node, a grey
# for the "other provinces" residual.
COLOR_SANKEY_WITHIN = "#B07070"
COLOR_SANKEY_REST = "#BFBFBF"

# The app began as a product->province locator; the unit of the answer is now the
# *industry* a province could grow into, which is what both lenses converge on in
# the candidate industries list. The title and caption say so.
APP_TITLE_EN = "Ecuador Diversification Explorer"
APP_TITLE_ES = "Explorador de Diversificación del Ecuador"
APP_CAPTION_EN = (
    "Which industries could each province of Ecuador plausibly grow into? Two lenses "
    "— top-down from 50 identified product opportunities, bottom-up from what each "
    "province already does — meet in one candidate industry list per province."
)
APP_CAPTION_ES = (
    "¿A qué industrias podría expandirse cada provincia del Ecuador? Dos enfoques, uno "
    "descendente que parte de 50 oportunidades de producto ya identificadas y otro "
    "ascendente que parte de lo que cada provincia ya hace, convergen en una lista de "
    "industrias candidatas por provincia."
)

# --- Language ---------------------------------------------------------------
# The app ships in Spanish: it is shared publicly with Ecuadorian policymakers,
# and English stays one click away for Growth Lab colleagues. Only UI chrome
# translates — province names and CIIU descriptions are data and stay Spanish in
# both modes. See notes/reference/product-locator/language_toggle.md.
LANGS = ("es", "en")
DEFAULT_LANG = "es"
LANG_LABELS = {"es": "Español", "en": "English"}

# Tile map settings (plotly MapLibre traces).
MAP_STYLE = "carto-positron"
MAP_CENTER = {"lat": -1.6, "lon": -82.5}  # midpoint keeping Galapagos in view
MAP_ZOOM = 5.0
MAP_HEIGHT = 650

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
# The parenthetical shape matters: callers strip " (...)" to get the short form,
# so each Spanish label keeps its qualifier in parentheses exactly as here.
STATE_LABELS_ES = {
    "specialized": "Especializada (RCA ≥ 1, núcleo)",
    "present_not_spec": "Presente, no especializada",
    "absent": "Ausente (nueva entrada)",
}
# Opportunity margin colors — the presence-state palette, which already carries
# the meaning: intensive margin = deepen something Ecuador already exports
# competitively (the "specialized" green); extensive margin = a genuinely new
# entry (the "absent" blue).
MARGIN_COLORS = {
    "intensivo": STATE_COLORS["specialized"],
    "extensivo": STATE_COLORS["absent"],
}
# Broad-sector (gsector) colors for the Page-4 attractiveness figures — the
# project-wide mapping every notebook uses (defined in industry_space.ipynb from
# gl_design's metroverse.industry_groups palette; values copied here because the
# app does not load the notebook skill stack). Keys are the Spanish
# gsector_label values in class_labor_profile.csv.
GSECTOR_COLORS = {
    "Agricultura, ganadería, silvicultura y pesca": "#76C799",  # Natural resources
    "Explotación de minas y canteras": "#FFC034",  # Financial activities (no mining group)
    "Industrias manufactureras": "#498099",  # Manufacturing
    "Comercio": "#F28188",  # Trade & transportation
    "Construcción": "#A773BF",  # Construction
    "Servicios": "#d25262",  # Professional & business (the services mass)
    "Otros / No clasificado": "#6A6AAC",  # Other
}
# Sequential colorscale for the density choropleth.
DENSITY_COLORSCALE = "Blues"
# Outline for M=1 ("already present") provinces on the density choropleth — GL
# brand red, bright enough to stay legible over the dark end of the Blues scale.
COLOR_PRESENT_OUTLINE = "#ee3e4c"


# --------------------------------------------------------------------------- #
# The candidate list — the third lens ("Both lenses in action").               #
# --------------------------------------------------------------------------- #
#
# The page derives the list in-app (metrics.candidate_list) and reads **none** of the
# four committed CSVs in data/processed/summary_tables/province_candidates/, which is
# what makes the app deployable with no builder run. These are therefore the
# parameters the derivation runs at — the same values utils/build_province_candidates.py
# pins, held here once so the app and the builder cannot drift on a number nobody is
# looking at. The equality tripwire in the test suite is what proves they have not.
#
# Every one is a *pinned* app control, not a widget: the page's only control is the
# attractiveness metric.

# Top-down leg: the app's own defaults (see TD_PARAM_DEFAULTS in state.py, ROUTE_PRESETS
# and DEFAULT_PRESENCE_CUT above).
CL_TD_VIEW = "lq"          # presence as a location quotient (1 = the national average)
CL_TD_BASE = "plazas"      # counted on establishment-booked FTE
CL_TD_SUPPORT = True       # the 5-establishment AND 50-FTE guard, on
CL_TD_ROUTE_WEIGHT = ROUTE_PRESETS[DEFAULT_ROUTE_PRESET]  # "Balanced" — 0.65
CL_TD_CUT = DEFAULT_PRESENCE_CUT  # the assignment bar at x1.0 parity: the RCA >= 1 test

# Bottom-up leg: the explorer's shipped defaults, both guards on.
CL_BU_BETA = BU_DEFAULT_BETA
CL_BU_WEIGHT = BU_DEFAULT_WEIGHT
CL_BU_TOP_K = BU_DEFAULT_TOP_K

# The floor every province is filled to, and **not a control** (settled, 2026-08-27):
# ten is enough industries that a reader can look for *structure* in a list — three
# wood-manufacturing classes read as a cluster worth promoting together, where ten
# unrelated ones do not — while staying small enough to investigate seriously. It was
# chosen for how a person uses the list, not for a statistical property, so exposing it
# as a slider would invite tuning it as though it had one.
CL_MIN_INDUSTRIES = 10

# The two lenses' colors on the Section-1 bar. Matched to the rendered matplotlib
# figure the workstream already ships (figs/province_candidates/industrias_por_provincia_es.png):
# gl_design's ecuador.geography navy for the top-down base, its focus red for the
# bottom-up top-up. Named here rather than written as hex in the view, like every other
# color in this app.
LENS_COLORS = {"top_down": "#0c2347", "bottom_up": "#d25262"}
LENS_LABELS = {
    "top_down": "From an opportunity (top-down)",
    "bottom_up": "Topped up from the province ranking (bottom-up)",
}
LENS_LABELS_ES = {
    "top_down": "Desde una oportunidad (enfoque descendente)",
    "bottom_up": "Complemento del ordenamiento provincial (enfoque ascendente)",
}
