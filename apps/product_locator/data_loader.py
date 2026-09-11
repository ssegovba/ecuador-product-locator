from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    CIIU_PATH,
    DENSITY_BASE,
    ENTRY_REGIME_VALUES,
    EXCLUDED_CLASSES,
    EXTRACTIVE_CLASSES,
    ISIC_DETAILED_JSON_PATH,
    LABOR_COMPOSITION_PATH,
    LABOR_PROFILE_PATH,
    LOCAL_RELEVANCE_PATH,
    OPPORTUNITY_INDUSTRIES_PATH,
    PROVINCE_RCA_PATH,
    PROVINCES_GEOJSON_PATH,
    REEM_ESTAB_PROCESSED_PATH,
    REEM_PROCESSED_PATH,
    RELATEDNESS_DIR,
    TRADABILITY_PATH,
    TRADABILITY_PCTILE_PATH,
)


@st.cache_data(show_spinner=False)
def load_reem_data() -> pd.DataFrame:
    df = pd.read_parquet(
        REEM_PROCESSED_PATH,
        columns=[
            "id_empresa",
            "codigo_provincia",
            "provincia",
            "codigo_seccion",
            "codigo_division",
            "codigo_clase",
            "empleo_equiv",
            "plazas_equiv",
            "remuneraciones",
            "ventas_totales",
            "exportaciones",
            "importaciones",
            "exportaciones_netas",
        ],
    )

    df["codigo_provincia"] = df["codigo_provincia"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(2)
    df["provincia"] = df["provincia"].astype(str).str.strip().str.upper()
    df["codigo_seccion"] = df["codigo_seccion"].astype(str).str.strip().str.upper().str[:1]
    df["codigo_division"] = df["codigo_division"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(2).str[-2:]
    df["codigo_clase"] = df["codigo_clase"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(4).str[-4:]

    numeric_cols = [
        "empleo_equiv",
        "plazas_equiv",
        "remuneraciones",
        "ventas_totales",
        "exportaciones",
        "importaciones",
        "exportaciones_netas",
    ]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Net exporter flag: positive net exports (NaN counts as non-exporter).
    df["is_net_exporter"] = df["exportaciones_netas"].fillna(0) > 0

    return df


@st.cache_data(show_spinner=False)
def load_estab_data() -> pd.DataFrame:
    """Active REEM establishments (local units: matriz + sucursales)."""
    df = pd.read_parquet(
        REEM_ESTAB_PROCESSED_PATH,
        columns=[
            "id_empresa",
            "id_unidad_local",
            "codigo_provincia",
            "provincia",
            "codigo_clase",
            "plazas_equiv",
            "is_active",
        ],
    )
    df = df[df["is_active"]].drop(columns=["is_active"])
    # Re-assert string dtype. codigo_clase in the processed establishment file carries the CIIU
    # section-letter prefix (e.g. "A0141"); strip it to the bare 4-digit form so it matches the
    # ISIC4 codes used throughout the app (mirrors the enterprise loader above).
    df["codigo_provincia"] = df["codigo_provincia"].astype(str).str.zfill(2)
    df["provincia"] = df["provincia"].astype(str).str.strip().str.upper()
    df["codigo_clase"] = df["codigo_clase"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(4).str[-4:]
    df["plazas_equiv"] = pd.to_numeric(df["plazas_equiv"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_ciiu_reference() -> dict[str, object]:
    ciiu = pd.read_csv(CIIU_PATH)
    ciiu["ciiu_code"] = ciiu["ciiu_code"].astype(str).str.strip().str.upper()
    ciiu["ciiu_level"] = ciiu["ciiu_level"].astype(str).str.strip().str.lower()
    ciiu["ciiu_name"] = ciiu["ciiu_name"].astype(str).str.strip()

    sections = ciiu[ciiu["ciiu_level"] == "section"].copy()
    sections["section_code"] = sections["ciiu_code"].str[:1]
    section_map = dict(zip(sections["section_code"], sections["ciiu_name"]))

    classes = ciiu[ciiu["ciiu_level"] == "class"].copy()
    classes["class_code"] = classes["ciiu_code"].str.extract(r"(\d{4})", expand=False)
    class_name_lookup = (
        classes.dropna(subset=["class_code"])
        .drop_duplicates(subset=["class_code"])
        .set_index("class_code")["ciiu_name"]
        .to_dict()
    )

    return {
        "section_map": section_map,
        "class_name_lookup": class_name_lookup,
        "reference": ciiu,
    }


@st.cache_data(show_spinner=False)
def load_opportunities() -> pd.DataFrame:
    """The 50 HS4 diversification opportunities mapped to CIIU classes.

    One row per (opportunity × route × industry), from
    ``summary_tables/oportunidades/oportunidades_industrias.csv``. ``route``
    splits two genuinely different questions: **``producto``** rows are the
    industries that make the opportunity product itself; **``ancla``** rows are
    the industries behind its *anchors* — products Ecuador already exports
    competitively but which sit in sparse parts of the product space, so the jump
    leans on marginal-but-existing capabilities. Extensive-margin opportunities
    carry both; intensive-margin ones carry only ``producto``.

    ``crosswalk_weight_used`` is the allocation weight to use — the resolved
    fallback (Ecuador's own within-HS4 export mix where it clears the export's
    $1M support floor, else the US reference in ``crosswalk_weight``). It sums to
    **1 within (product, ``producto``)** but to **n_anchors within (product,
    ``ancla``)**, because each anchor carries its own weights summing to 1 —
    which is why the anchor route is *averaged* across anchors, never summed.

    All three HS/ISIC code columns are zero-padded strings; product and industry
    names are Spanish and English respectively, as shipped.
    """
    df = pd.read_csv(
        OPPORTUNITY_INDUSTRIES_PATH,
        dtype={"opportunity_hs4_code": str, "anchor_hs4_code": str, "isic4_code": str},
        encoding="utf-8-sig",
    )
    for column in ["opportunity_hs4_code", "anchor_hs4_code"]:
        df[column] = df[column].astype("string").str.extract(r"(\d+)", expand=False).str.zfill(4)
    df["isic4_code"] = (
        df["isic4_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(4)
    )
    numeric = [
        "rank", "opportunity_priority_score", "crosswalk_weight", "crosswalk_weight_ec_mix",
        "crosswalk_weight_used", "export_value_allocated", "anchor_to_candidate_proximity",
        "anchor_density_percentile",
        # Billions of current USD — the market the product could sell into, a
        # different quantity from export_value_allocated (what Ecuador ships
        # today). Tolerated as missing so the app still runs against an export
        # built before the column was added.
        "opportunity_accessible_market_busd",
    ]
    for column in numeric:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan

    # Tripwire (a): the allocation weights must reconstruct exactly one product
    # per route — 1 for `producto`, one per anchor for `ancla`. A drift here
    # would silently rescale every ranking on the page.
    direct = df[df["route"] == "producto"].groupby("opportunity_hs4_code")["crosswalk_weight_used"].sum()
    if not np.allclose(direct.to_numpy(), 1.0):
        bad = direct[~np.isclose(direct.to_numpy(), 1.0)]
        raise ValueError(f"crosswalk_weight_used does not sum to 1 within (product, producto): {bad.to_dict()}")
    per_anchor = (
        df[df["route"] == "ancla"]
        .groupby(["opportunity_hs4_code", "anchor_hs4_code"])["crosswalk_weight_used"].sum()
    )
    if len(per_anchor) and not np.allclose(per_anchor.to_numpy(), 1.0):
        bad = per_anchor[~np.isclose(per_anchor.to_numpy(), 1.0)]
        raise ValueError(f"crosswalk_weight_used does not sum to 1 within (product, anchor): {bad.to_dict()}")
    return df


@st.cache_data(show_spinner=False)
def load_province_rca() -> pd.DataFrame:
    """Province × CIIU class presence on three bases (industry_space notebook).

    The single presence source for the top-down lens. **24 provinces and no
    ``'99'`` row** — the national aggregate simply is not in this export, which
    is why it is preferred over the composition table for anything that ranks
    provinces. Dense: every (province, class) pair has a row, absent ones with
    zeros.

    ``rca_*`` is the location quotient (the province's share of its own
    employment/units in the class ÷ the class's national share; 1 = national
    average); ``share_national_*`` is the province's share of the class's
    national total, which sums to 1 across the 24 provinces. ``plazas_equiv`` is
    establishment-booked employment — registered formal FTE **positions** the
    industry holds today, counted where the unit operates, never jobs an entrant
    would create.

    The shipped ``low_support`` column (``num_establecimientos < 5`` alone) is
    loaded for reference but **not used**: the top-down lens computes its own
    two-dimensional AND guard from ``num_establecimientos`` and ``plazas_equiv``
    (see ``config.MIN_ESTAB`` / ``MIN_PLAZAS``).
    """
    df = pd.read_csv(
        PROVINCE_RCA_PATH,
        dtype={"id_code": str, "codigo_clase": str},
        encoding="utf-8-sig",
    )
    df["id_code"] = df["id_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(2)
    df["codigo_clase"] = (
        df["codigo_clase"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(4)
    )
    numeric = [
        "num_establecimientos", "plazas_equiv", "rca_estab_count", "rca_plazas",
        "share_national_estab_count", "share_national_plazas",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["low_support"] = df["low_support"].astype(bool)
    columns = [
        "id_code", "id_name", "id_abbr", "region", "codigo_clase", "class_name",
        "class_name_es", "gsector_label", *numeric, "low_support",
    ]
    df = df[~df["codigo_clase"].isin(EXCLUDED_CLASSES)]
    # Tripwire (b): the national row must never reach a provincial ranking.
    if (df["id_code"] == "99").any():
        raise ValueError("province_class_rca_multibase.csv unexpectedly carries a '99' row")
    return df[columns].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_proximity_matrix(firm_set: str = "multi_industry") -> pd.DataFrame:
    """Symmetric industry proximity Φ (φ ∈ [0, 1]) from the relatedness notebook.

    Square DataFrame indexed and columned by 4-digit ISIC class (zero-padded
    strings); the diagonal is 1 (self-proximity). ``EXCLUDED_CLASSES`` are
    dropped from both axes.
    """
    path = RELATEDNESS_DIR / f"industry_proximity_matrix_{firm_set}.csv"
    matrix = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    matrix.index = (
        matrix.index.astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(4)
    )
    matrix.columns = (
        pd.Index(matrix.columns).astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(4)
    )
    keep_rows = ~matrix.index.isin(EXCLUDED_CLASSES)
    keep_cols = ~matrix.columns.isin(EXCLUDED_CLASSES)
    matrix = matrix.loc[keep_rows, keep_cols]
    return matrix.apply(pd.to_numeric, errors="coerce")


@st.cache_data(show_spinner=False)
def load_province_density(base: str = DENSITY_BASE) -> pd.DataFrame:
    """Province × ISIC relatedness density (one RCA base). ``EXCLUDED_CLASSES`` dropped.

    Columns: id_code, id_name, codigo_clase, class_name, sector, intensity,
    density, density_cont, density_dir, M, rca, state.
    """
    path = RELATEDNESS_DIR / f"province_density_{base}.csv"
    df = pd.read_csv(path, dtype={"id_code": str, "codigo_clase": str}, encoding="utf-8-sig")
    df["id_code"] = df["id_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(2)
    df["codigo_clase"] = df["codigo_clase"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(4)
    numeric = ["intensity", "density", "density_cont", "density_noself",
               "density_cont_noself", "density_dir", "rca"]
    for column in numeric:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["M"] = pd.to_numeric(df["M"], errors="coerce").fillna(0).astype(int)
    df = df[~df["codigo_clase"].isin(EXCLUDED_CLASSES)].reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_density_support() -> pd.DataFrame:
    """Per-class industry-space support (constant across provinces): phi_mass,
    phi_mass_noself (off-diagonal proximity mass), n_neighbors_pos. Used to flag
    thin-neighborhood classes whose no-self density is unstable."""
    path = RELATEDNESS_DIR / "industry_density_support.csv"
    df = pd.read_csv(path, dtype={"codigo_clase": str}, encoding="utf-8-sig")
    df["codigo_clase"] = df["codigo_clase"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(4)
    for column in ["phi_mass", "phi_mass_noself", "n_neighbors_pos"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df[~df["codigo_clase"].isin(EXCLUDED_CLASSES)].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_labor_profile() -> pd.DataFrame:
    """National attractiveness metrics per CIIU class (2024) from the
    industry_labor_profile notebook: average wage paid (``wage_per_fte``),
    average jobs per firm (``fte_per_firm``) and total establishment-booked
    employment (``plazas_equiv_estab_nat``), each with national and
    within-broad-sector percentiles, plus ``n_firms`` / ``n_estab_nat`` and
    name / sector / gsector labels.

    Every value is **national** — REEM books enterprise variables to the HQ
    province, so no provincial variant of these metrics exists. Six classes
    carry no wage metric (no employing firm reports pay); they stay in the
    table with NaN (flag, don't filter). ``plazas_equiv_estab_nat`` is a
    stock (employment the industry holds), never a jobs-created flow.
    """
    df = pd.read_csv(LABOR_PROFILE_PATH, dtype={"codigo_clase": str}, encoding="utf-8-sig")
    df["codigo_clase"] = df["codigo_clase"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(4)
    numeric = [
        "plazas_equiv_estab_nat", "fte_share_nat", "plazas_equiv_estab_nat_pctile",
        "n_estab_nat", "wage_per_fte", "wage_premium", "wage_per_fte_pctile",
        "wage_per_fte_pctile_gsector", "fte_per_firm", "fte_per_firm_premium",
        "fte_per_firm_pctile", "fte_per_firm_pctile_gsector", "n_firms",
    ]
    for column in numeric:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df[~df["codigo_clase"].isin(EXCLUDED_CLASSES)].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_labor_composition() -> pd.DataFrame:
    """Establishment-booked industry composition per geography × CIIU level
    (2024), from the labor-profile notebook's composition export — employment
    and establishments counted **where units operate**, not at the HQ, so both
    are genuinely provincial (the export's ``*_nat`` ratio columns are not, and
    are deliberately left behind here).

    All three CIIU levels ship (``section`` / ``division`` / ``class``); ``code``
    is 1 letter / 2 digits / 4 digits accordingly and ``codigo_clase`` carries
    the 4-digit code on class rows only (empty elsewhere). Covers only present
    (geography, industry) combinations — an absent pair means zero, fill after
    the join. ``id_code`` '99' is the national aggregate.

    Returns id_code, id_name, level, code, codigo_clase, display_code, name_en,
    name_es, gsector_label, plazas_equiv, plazas_equiv_share, n_estab,
    n_estab_share.
    """
    df = pd.read_csv(
        LABOR_COMPOSITION_PATH, dtype={"id_code": str, "code": str}, encoding="utf-8-sig"
    )
    df["id_code"] = df["id_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(2)
    df["level"] = df["level"].astype(str).str.strip().str.lower()
    is_class = df["level"] == "class"
    df["code"] = df["code"].astype(str).str.strip()
    # Zero-pad the numeric levels; section codes are letters and stay as they are.
    df.loc[is_class, "code"] = df.loc[is_class, "code"].str.zfill(4)
    df.loc[df["level"] == "division", "code"] = df.loc[df["level"] == "division", "code"].str.zfill(2)
    df["codigo_clase"] = df["code"].where(is_class, "")
    for column in ["plazas_equiv", "plazas_equiv_share", "n_estab", "n_estab_share"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[~(is_class & df["code"].isin(EXCLUDED_CLASSES))]
    columns = [
        "id_code", "id_name", "level", "code", "codigo_clase", "display_code",
        "name_en", "name_es", "gsector_label",
        "plazas_equiv", "plazas_equiv_share", "n_estab", "n_estab_share",
    ]
    return df[columns].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_labor_class_employment() -> pd.DataFrame:
    """Establishment-booked employment per province × CIIU class (2024) — the
    class slice of :func:`load_labor_composition`, i.e. the provincial
    counterpart of ``plazas_equiv_estab_nat`` (same attribution: employment is
    counted where units operate, not at the HQ). Covers only present
    (province, class) combinations — an absent pair means zero, fill after the
    join. ``id_code`` '99' is the national aggregate.

    Returns id_code, codigo_clase, plazas_equiv, n_estab.
    """
    df = load_labor_composition()
    df = df[df["level"] == "class"]
    return df[["id_code", "codigo_clase", "plazas_equiv", "n_estab"]].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_tradability() -> pd.DataFrame:
    """AI-labeled ISIC industry labels, keyed on the bare 4-digit class.

    Three label axes ship in this file, all properties of the *industry* and none
    of them derived from Ecuadorian outcomes:

    ``tradable_atlas`` (can it cross a border?) and ``tradable_regional`` (can it
    leave the province it is produced in?) are strings ('tradable' /
    'non-tradable'); the regional definition is the strictly weaker one.
    ``confidence_regional`` and ``rationale`` travel with them.

    ``entry_regime`` answers the different question of whether a province can
    *act on* an industry at all — one of 'open_market' (a private entrant can
    cause it to appear), 'state_allocated' (a ministry, budget or siting decision
    puts it there) or 'regulated_network' (entry is a national licence or charter
    and provincial presence is a branch node). ``confidence_entry``,
    ``mixed_entry`` and ``rationale_entry`` travel with it, and are review
    machinery rather than user-facing signals. The menu filter consumes
    ``tradable_regional`` and ``entry_regime`` together.

    **Three rows are keyed with a section-letter prefix** in the source — the
    CIIU-only classes ``J6203``, ``K6613`` and ``K6614`` — so the code column is
    stripped to its digits before anything joins on it (without the strip, 6614
    silently falls out of every merge). ``codigo_clase`` is the canonical join key
    and ``isic_code`` is retained under its original name for the older callers.

    Not every class has a row, but the codes that are here are unique and cover
    the app's whole universe (both asserted).
    """
    df = pd.read_csv(TRADABILITY_PATH, dtype=str, encoding="utf-8-sig")
    df["isic_code"] = df["isic_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(4)
    df["codigo_clase"] = df["isic_code"]
    for column in ["tradable_atlas", "tradable_regional", "entry_regime"]:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip().str.lower()
    for column in ["confidence_atlas", "confidence_regional", "confidence_entry"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "mixed_entry" in df.columns:
        df["mixed_entry"] = df["mixed_entry"].astype(str).str.strip().str.lower().eq("true")
    # Tripwire: an entry_regime value outside the vocabulary must fail loudly. A
    # typo'd label would otherwise silently drop its class from the menu —
    # load_tradable_menu() keeps only ``entry_regime == 'open_market'`` — and a
    # data error that changes what the app recommends must not pass quietly.
    if "entry_regime" in df.columns:
        unknown = sorted(set(df["entry_regime"]) - ENTRY_REGIME_VALUES)
        if unknown:
            raise ValueError(
                f"isic_tradability_ai.csv carries entry_regime values outside the "
                f"vocabulary {sorted(ENTRY_REGIME_VALUES)}: {unknown}"
            )
    # Tripwire: a duplicated code would make the menu merge fan out rows, and a
    # missing 6614 is the exact failure the prefix strip above exists to prevent.
    duplicated = df.loc[df["codigo_clase"].duplicated(), "codigo_clase"].tolist()
    if duplicated:
        raise ValueError(f"isic_tradability_ai.csv carries duplicate class codes: {duplicated}")
    for expected in ["6203", "6613", "6614"]:
        if expected not in set(df["codigo_clase"]):
            raise ValueError(
                f"isic_tradability_ai.csv: class {expected} is missing after the "
                "section-letter strip — the prefixed CIIU-only codes did not parse"
            )
    return df


@st.cache_data(show_spinner=False)
def load_tradable_menu() -> frozenset[str]:
    """The bottom-up page's **menu** of recommendable industries, derived.

    ``tradable_regional == 'tradable'`` **and** ``entry_regime == 'open_market'``,
    minus :data:`config.EXTRACTIVE_CLASSES` — industries whose output can plausibly
    leave the province that makes it, that a private entrant could actually cause
    to appear there, less the ones whose location is geology rather than
    capability. Never hardcoded: the count moves whenever the labels are revised,
    and the page prints whatever this returns.

    The two label axes answer different questions and both have to pass. Regional
    tradability asks whether the output can leave the province, to which a regional
    hospital or a bank branch says yes — but a hospital is sited by the Ministerio
    de Salud and a branch is a node of a national charter, so neither is something
    a province can diversify *into*. ``entry_regime`` is that second test. Only
    ``open_market`` survives; ``mixed_entry`` deliberately does **not** enter the
    rule, because the label commits to a side even when the class straddles (see
    the review doc beside the data).

    This is a **choice set, not an information set**: it restricts what the page
    may recommend and never what counts as evidence. Density, RCA, φ and every
    share stay exactly as the notebooks exported them (recomputing density with
    neighbours restricted to the menu was measured at Spearman ≈ 0.999 with-self
    and 0.90-0.95 related-only, and was rejected — see the as-built).

    Returned unintersected with any universe; callers intersect with the classes
    they actually plot and caption the result.
    """
    trad = load_tradability()
    eligible = trad["tradable_regional"].eq("tradable") & trad["entry_regime"].eq("open_market")
    menu = set(trad.loc[eligible, "codigo_clase"])
    return frozenset(menu - EXTRACTIVE_CLASSES - EXCLUDED_CLASSES)


@st.cache_data(show_spinner=False)
def load_local_relevance() -> pd.DataFrame:
    """Per-province industry relevance (geographic concentration, plazas base).

    Covers only industries **present** in a province. ``codigo_clase`` is
    section-letter-prefixed in the source (e.g. 'C2391') and is stripped to the
    bare 4-digit form here for joining. Returns id_code, codigo_clase,
    mean_relevance, rank_in_province.
    """
    df = pd.read_csv(LOCAL_RELEVANCE_PATH, dtype={"id_code": str}, encoding="utf-8-sig")
    df["id_code"] = df["id_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(2)
    df["codigo_clase"] = df["codigo_clase"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(4)
    df["mean_relevance"] = pd.to_numeric(df["mean_relevance"], errors="coerce")
    df["rank_in_province"] = pd.to_numeric(df["rank_in_province"], errors="coerce")
    return df[["id_code", "codigo_clase", "mean_relevance", "rank_in_province"]]


@st.cache_data(show_spinner=False)
def load_tradability_pctile() -> pd.DataFrame:
    """National tradability percentile per industry (industry_tradability
    classification). ``codigo_clase`` is section-prefixed in the source and is
    stripped to 4-digit here. Returns codigo_clase, tradability_pctile."""
    df = pd.read_csv(TRADABILITY_PCTILE_PATH, dtype=str, encoding="utf-8-sig")
    df["codigo_clase"] = df["codigo_clase"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(4)
    df["tradability_pctile"] = pd.to_numeric(df["tradability_pctile"], errors="coerce")
    df = df[df["codigo_clase"].str.fullmatch(r"\d{4}")]
    return df.drop_duplicates("codigo_clase")[["codigo_clase", "tradability_pctile"]]


@st.cache_data(show_spinner=False)
def load_class_labels() -> dict[str, str]:
    """4-digit CIIU class -> display name, Spanish preferred (the app-wide
    convention: English ISIC names as the base, overridden by the Spanish CIIU
    names wherever they exist)."""
    isic_desc = load_isic_descriptions()
    ciiu_ref = load_ciiu_reference()
    labels = dict(zip(isic_desc["isic_code"], isic_desc["isic_name"]))
    labels.update(ciiu_ref["class_name_lookup"])
    return labels


@st.cache_data(show_spinner=False)
def load_isic_descriptions() -> pd.DataFrame:
    """Flatten isic_rev4_detailed.json into one row per 4-digit ISIC class."""
    payload = json.loads(ISIC_DETAILED_JSON_PATH.read_text(encoding="utf-8"))

    rows: list[dict[str, str]] = []

    def walk(node: dict) -> None:
        if str(node.get("hierarchy", "")).strip().lower() == "class":
            code = str(node.get("isic_code", "")).strip()
            if code.isdigit():
                rows.append(
                    {
                        "isic_code": code.zfill(4),
                        "isic_name": str(node.get("isic_name", "")).strip(),
                        "description": str(node.get("description") or "").strip(),
                    }
                )
        for child in node.get("breakdown", []) or []:
            if isinstance(child, dict):
                walk(child)

    nodes = payload if isinstance(payload, list) else [payload]
    for node in nodes:
        walk(node)

    df = pd.DataFrame(rows).drop_duplicates(subset=["isic_code"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_province_geojson() -> dict:
    """Simplified province polygons (EPSG:4326) with id_code/id_name/id_abbr/region in
    ``properties`` — precomputed by utils/build_product_locator_app_data.py, so the app
    needs no geopandas and no data/raw/ shapefile at runtime."""
    return json.loads(PROVINCES_GEOJSON_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_province_geodata() -> pd.DataFrame:
    """Province attribute table (id_code, id_name, id_abbr, region, province_label),
    read from the province GeoJSON's feature properties — a plain DataFrame (no geometry;
    the geometry lives in ``load_province_geojson``)."""
    features = load_province_geojson().get("features", [])
    df = pd.DataFrame([f.get("properties", {}) for f in features])
    df["id_code"] = df["id_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(2)
    for column in ["id_name", "id_abbr", "region", "province_label"]:
        if column in df.columns:
            df[column] = df[column].astype(str)
    return df
