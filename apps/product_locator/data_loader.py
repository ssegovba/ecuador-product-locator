from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from config import (
    CENSUS_SUMMARY_PATH,
    CIIU_PATH,
    DENSITY_BASE,
    ENEMDU_LABOR_SUMMARY_PATH,
    EXCLUDED_CLASSES,
    EC_TRADE_PATHS,
    GVA_PROVINCE_INDUSTRY_PATH,
    GVA_PROVINCIAL_PATH,
    HS_EDITIONS,
    HS_REFERENCE_PATH,
    ISIC_DETAILED_JSON_PATH,
    LOCAL_RELEVANCE_PATH,
    MULTISOURCE_PATH,
    MULTISOURCE_RANKED_PATH,
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
def load_hs_reference(edition: str) -> pd.DataFrame:
    """HS6 codes + descriptions for one HS edition (Comtrade reference)."""
    ref = pd.read_csv(HS_REFERENCE_PATH, dtype=str)
    ref = ref[ref["hs_edition"] == str(edition)].copy()
    ref["hs_code"] = ref["hs_code"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    ref = ref[ref["hs_code"].str.fullmatch(r"\d{6}")]
    ref["description"] = ref["description"].astype(str).str.strip()
    return ref.sort_values("hs_code").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_hs_isic_crosswalk(edition: str) -> pd.DataFrame:
    """Multiple-match HS->ISIC4 crosswalk for one HS edition."""
    path = HS_EDITIONS[str(edition)]
    cw = pd.read_csv(path, dtype=str)
    cw["hs_code"] = cw["hs_code"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    cw["isic4_code"] = cw["isic4_code"].astype(str).str.extract(r"(\d{4})", expand=False).fillna("").str.zfill(4)
    cw["match_weight"] = pd.to_numeric(cw.get("match_weight"), errors="coerce")
    cw = cw[(cw["hs_code"] != "") & (cw["isic4_code"] != "")]
    cw = cw.drop_duplicates(subset=["hs_code", "isic4_code"]).reset_index(drop=True)
    return cw


@st.cache_data(show_spinner=False)
def load_ec_trade(edition: str) -> pd.DataFrame:
    """Compact Ecuador export table (latest year) for one HS edition.

    Columns: hs_code (6-digit str), export_value, import_value,
    national_export_share, year.
    """
    df = pd.read_csv(EC_TRADE_PATHS[str(edition)], dtype={"hs_code": str}, encoding="utf-8-sig")
    df["hs_code"] = df["hs_code"].str.zfill(6)
    for column in ["export_value", "import_value", "national_export_share"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


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
def load_tradability() -> pd.DataFrame:
    """AI-labeled ISIC tradability. ``tradable_atlas``/``tradable_regional`` are
    strings ('tradable' / 'non-tradable'); not every class has a row."""
    df = pd.read_csv(TRADABILITY_PATH, dtype=str, encoding="utf-8-sig")
    df["isic_code"] = df["isic_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(4)
    for column in ["tradable_atlas", "tradable_regional"]:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip().str.lower()
    return df


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
def load_province_profile_tables() -> dict[str, pd.DataFrame]:
    """Summary tables backing the province profile page (Page 3)."""

    def read(path) -> pd.DataFrame:
        df = pd.read_csv(path, dtype={"id_code": "str"}, encoding="utf-8-sig")
        df["id_code"] = df["id_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(2)
        return df

    return {
        "multisource": read(MULTISOURCE_PATH),
        "multisource_ranked": read(MULTISOURCE_RANKED_PATH),
        "gva_provincial": read(GVA_PROVINCIAL_PATH),
        "gva_province_industry": read(GVA_PROVINCE_INDUSTRY_PATH),
        "census": read(CENSUS_SUMMARY_PATH),
        "enemdu": read(ENEMDU_LABOR_SUMMARY_PATH),
    }


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
