from __future__ import annotations

import numpy as np
import pandas as pd

# Relatedness "state" ordering: specialized (best) > present-not-specialized > absent.
_STATE_RANK = {"specialized": 2, "present_not_spec": 1, "absent": 0}
_RANK_STATE = {2: "specialized", 1: "present_not_spec", 0: "absent"}

PROVINCE_METRIC_COLUMNS = [
    "codigo_provincia",
    "provincia",
    "enterprises",
    "establishments",
    "employment",
    "exporting_enterprises",
    "remuneraciones",
    "sales",
    "jobs_per_enterprise",
    "enterprises_share_country",
    "establishments_share_country",
    "employment_share_country",
    "exporters_share_country",
    "enterprises_share_province",
    "establishments_share_province",
    "employment_share_province",
    "exporters_share_province",
]

_EMPTY_SUMMARY = {
    "isic_classes_matched": 0,
    "provinces_with_activity": 0,
    "enterprises": 0,
    "establishments": 0,
    "employment": 0.0,
    "exporting_enterprises": 0,
    "jobs_per_enterprise": np.nan,
    "enterprises_share_national": np.nan,
    "establishments_share_national": np.nan,
    "employment_share_national": np.nan,
    "exporters_share_national": np.nan,
}


def build_hq_flows(
    reem_df: pd.DataFrame,
    estab_df: pd.DataFrame,
    isic_codes: list[str] | None,
    hq_province: str,
) -> dict[str, object]:
    """Reach of the **multi-unit** enterprises headquartered in ``hq_province``.

    Firms are scoped to the HQ province (and to ``isic_codes`` when given).
    All flows/totals below cover only *multi-unit* firms (> 1 establishment) —
    the single-establishment firms carry no HQ→branch structure. ``n_firms`` is
    still the full in-scope HQ count, used as the denominator for the multi-unit
    share. Employment is establishment-level ``plazas_equiv`` (NaN → 0).
    """
    firms = reem_df
    if isic_codes is not None:
        codes = sorted({str(code).zfill(4) for code in isic_codes})
        firms = firms[firms["codigo_clase"].isin(codes)]
    hq_firm_ids = firms.loc[firms["codigo_provincia"] == hq_province, "id_empresa"]
    n_firms = int(hq_firm_ids.nunique())

    estabs_all = estab_df[estab_df["id_empresa"].isin(set(hq_firm_ids))]

    # Multi-unit firms and how far their establishments reach.
    per_firm = estabs_all.groupby("id_empresa").agg(
        n_estabs=("id_unidad_local", "size"),
        n_outside=("codigo_provincia", lambda s: int((s != hq_province).sum())),
    )
    multi = per_firm[per_firm["n_estabs"] > 1]
    multi_ids = set(multi.index)

    # Flows/totals restricted to multi-unit firms' establishments.
    estabs = estabs_all[estabs_all["id_empresa"].isin(multi_ids)]
    flows = (
        estabs.groupby(["codigo_provincia", "provincia"], as_index=False)
        .agg(
            establishments=("id_unidad_local", "size"),
            employment=("plazas_equiv", lambda s: s.fillna(0).sum()),
        )
        .sort_values("employment", ascending=False)
        .reset_index(drop=True)
    )

    total_employment = float(flows["employment"].sum())
    total_establishments = int(flows["establishments"].sum())
    inside = flows[flows["codigo_provincia"] == hq_province]
    inside_employment = float(inside["employment"].sum())
    inside_establishments = int(inside["establishments"].sum())
    outside = flows[flows["codigo_provincia"] != hq_province]

    stats = {
        "n_firms": n_firms,
        # Row 1 — firm counts (multi-unit; within_only + outside == n_multi).
        "n_multi": int(len(multi)),
        "n_multi_within_only": int((multi["n_outside"] == 0).sum()),
        "n_multi_outside": int((multi["n_outside"] > 0).sum()),
        "share_multi_of_firms": len(multi) / n_firms if n_firms else np.nan,
        # Row 2 — establishments controlled by multi-unit firms.
        "total_establishments": total_establishments,
        "establishments_within": inside_establishments,
        "establishments_outside": total_establishments - inside_establishments,
        "share_inside_estab": inside_establishments / total_establishments if total_establishments else np.nan,
        "share_outside_estab": 1 - inside_establishments / total_establishments if total_establishments else np.nan,
        # Row 3 — employment controlled by multi-unit firms.
        "total_employment": total_employment,
        "employment_within": inside_employment,
        "employment_outside": total_employment - inside_employment,
        "share_inside": inside_employment / total_employment if total_employment else np.nan,
        "share_outside": 1 - inside_employment / total_employment if total_employment else np.nan,
        "top_destinations": outside.head(3)[["provincia", "employment", "establishments"]],
    }
    return {"flows": flows, "stats": stats}


def build_hq_industry_mix(
    reem_df: pd.DataFrame,
    estab_df: pd.DataFrame,
    isic_codes: list[str] | None,
    hq_province: str,
) -> dict[str, object]:
    """Industry composition of the (non-focus) establishments controlled by
    enterprises headquartered in ``hq_province``.

    With ``isic_codes`` given, establishments whose own class is in the focus
    set are excluded — what remains is the firms' diversification footprint
    (value-chain linkages). With ``isic_codes=None`` nothing is excluded and
    the full controlled mix is returned. Shares are computed within each
    location group ('within' the HQ province vs 'outside').

    Restricted to **multi-unit** firms (matching ``build_hq_flows``), so
    ``n_controlled`` equals that section's "establishments controlled".
    """
    firms = reem_df
    focus: set[str] = set()
    if isic_codes is not None:
        focus = {str(code).zfill(4) for code in isic_codes}
        firms = firms[firms["codigo_clase"].isin(focus)]
    hq_firm_ids = set(firms.loc[firms["codigo_provincia"] == hq_province, "id_empresa"])

    estabs_all = estab_df[estab_df["id_empresa"].isin(hq_firm_ids)]
    per_firm = estabs_all.groupby("id_empresa")["id_unidad_local"].size()
    multi_ids = set(per_firm[per_firm > 1].index)
    estabs = estabs_all[estabs_all["id_empresa"].isin(multi_ids)].copy()
    n_controlled = int(len(estabs))
    estabs["plazas_filled"] = estabs["plazas_equiv"].fillna(0)
    total_employment = float(estabs["plazas_filled"].sum())

    nonfocus = estabs[~estabs["codigo_clase"].isin(focus)].copy()
    nonfocus["location"] = np.where(
        nonfocus["codigo_provincia"] == hq_province, "within", "outside"
    )

    mix = (
        nonfocus.groupby(["location", "codigo_clase"], as_index=False)
        .agg(
            establishments=("id_unidad_local", "size"),
            employment=("plazas_filled", "sum"),
        )
    )
    group_totals = mix.groupby("location")[["establishments", "employment"]].transform("sum")
    mix["share_establishments"] = mix["establishments"] / group_totals["establishments"]
    mix["share_employment"] = mix["employment"] / group_totals["employment"].replace({0: np.nan})

    nonfocus_employment = float(nonfocus["plazas_filled"].sum())
    stats = {
        "n_controlled": n_controlled,
        "n_nonfocus": int(len(nonfocus)),
        "share_nonfocus": len(nonfocus) / n_controlled if n_controlled else np.nan,
        "n_firms_diversified": int(nonfocus["id_empresa"].nunique()),
        "n_within": int((nonfocus["location"] == "within").sum()),
        "n_outside": int((nonfocus["location"] == "outside").sum()),
        "employment_nonfocus": nonfocus_employment,
        "share_employment_nonfocus": nonfocus_employment / total_employment if total_employment else np.nan,
        "employment_within": float(nonfocus.loc[nonfocus["location"] == "within", "plazas_filled"].sum()),
        "employment_outside": float(nonfocus.loc[nonfocus["location"] == "outside", "plazas_filled"].sum()),
    }
    return {"mix": mix, "stats": stats}


def build_selection_metrics(
    reem_df: pd.DataFrame,
    isic_codes: list[str],
    employment_base: str = "empleo_equiv",
    estab_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Aggregate REEM enterprises (and, when given, active establishments)
    for the selected ISIC classes.

    Returns province_metrics (one row per province with any activity —
    enterprise- or establishment-side), a summary dict of national
    totals/shares for the selection, and an industry breakdown (one row per
    selected ISIC class).
    """
    codes = sorted({str(code).zfill(4) for code in isic_codes})
    matched = reem_df[reem_df["codigo_clase"].isin(codes)]
    estab_matched = (
        estab_df[estab_df["codigo_clase"].isin(codes)] if estab_df is not None else None
    )

    national_totals = {
        "enterprises": int(len(reem_df)),
        "establishments": int(len(estab_df)) if estab_df is not None else 0,
        "employment": float(reem_df[employment_base].sum()),
        "exporting_enterprises": int(reem_df["is_net_exporter"].sum()),
    }

    no_enterprises = matched.empty
    no_establishments = estab_matched is None or estab_matched.empty
    if no_enterprises and no_establishments:
        return {
            "province_metrics": pd.DataFrame(columns=PROVINCE_METRIC_COLUMNS),
            "summary": dict(_EMPTY_SUMMARY),
            "industry_breakdown": pd.DataFrame(
                columns=["codigo_clase", "enterprises", "establishments", "employment", "exporting_enterprises"]
            ),
        }

    # Enterprise-side province aggregation (headquarters location).
    ent = (
        matched.groupby(["codigo_provincia", "provincia"], as_index=False)
        .agg(
            enterprises=("id_empresa", "size"),
            employment=(employment_base, "sum"),
            exporting_enterprises=("is_net_exporter", "sum"),
            remuneraciones=("remuneraciones", "sum"),
            sales=("ventas_totales", "sum"),
        )
    )

    # Establishment-side aggregation (where activity physically happens).
    if not no_establishments:
        est = (
            estab_matched.groupby(["codigo_provincia"], as_index=False)
            .agg(
                establishments=("id_unidad_local", "size"),
                provincia_est=("provincia", "first"),
            )
        )
    else:
        est = pd.DataFrame(columns=["codigo_provincia", "establishments", "provincia_est"])

    province_metrics = ent.merge(est, on="codigo_provincia", how="outer")
    province_metrics["provincia"] = province_metrics["provincia"].fillna(
        province_metrics["provincia_est"]
    )
    province_metrics = province_metrics.drop(columns=["provincia_est"])
    for column in ["enterprises", "establishments", "exporting_enterprises"]:
        province_metrics[column] = province_metrics[column].fillna(0).astype(int)
    for column in ["employment", "remuneraciones", "sales"]:
        province_metrics[column] = province_metrics[column].fillna(0.0)

    total_enterprises = int(province_metrics["enterprises"].sum())
    total_establishments = int(province_metrics["establishments"].sum())
    total_employment = float(province_metrics["employment"].sum())
    total_exporters = int(province_metrics["exporting_enterprises"].sum())

    province_metrics["jobs_per_enterprise"] = (
        province_metrics["employment"] / province_metrics["enterprises"].replace({0: np.nan})
    )
    province_metrics["enterprises_share_country"] = province_metrics["enterprises"] / (total_enterprises or np.nan)
    province_metrics["establishments_share_country"] = province_metrics["establishments"] / (total_establishments or np.nan)
    province_metrics["employment_share_country"] = province_metrics["employment"] / (total_employment or np.nan)
    province_metrics["exporters_share_country"] = province_metrics["exporting_enterprises"] / (total_exporters or np.nan)

    # Share of each province's full REEM activity represented by the selection.
    province_totals = (
        reem_df.groupby(["codigo_provincia"], as_index=False)
        .agg(
            province_enterprises=("id_empresa", "size"),
            province_employment=(employment_base, "sum"),
            province_exporters=("is_net_exporter", "sum"),
        )
    )
    if not no_establishments:
        prov_est = (
            estab_df.groupby("codigo_provincia", as_index=False)
            .agg(province_establishments=("id_unidad_local", "size"))
        )
        province_totals = province_totals.merge(prov_est, on="codigo_provincia", how="left")
    else:
        province_totals["province_establishments"] = np.nan

    province_metrics = province_metrics.merge(province_totals, on="codigo_provincia", how="left")
    province_metrics["enterprises_share_province"] = (
        province_metrics["enterprises"] / province_metrics["province_enterprises"].replace({0: np.nan})
    )
    province_metrics["establishments_share_province"] = (
        province_metrics["establishments"] / province_metrics["province_establishments"].replace({0: np.nan})
    )
    province_metrics["employment_share_province"] = (
        province_metrics["employment"] / province_metrics["province_employment"].replace({0: np.nan})
    )
    province_metrics["exporters_share_province"] = (
        province_metrics["exporting_enterprises"] / province_metrics["province_exporters"].replace({0: np.nan})
    )

    industry_breakdown = (
        matched.groupby("codigo_clase", as_index=False)
        .agg(
            enterprises=("id_empresa", "size"),
            employment=(employment_base, "sum"),
            exporting_enterprises=("is_net_exporter", "sum"),
        )
    )
    if not no_establishments:
        est_by_class = (
            estab_matched.groupby("codigo_clase", as_index=False)
            .agg(establishments=("id_unidad_local", "size"))
        )
        industry_breakdown = industry_breakdown.merge(est_by_class, on="codigo_clase", how="outer")
        for column in ["enterprises", "establishments", "exporting_enterprises"]:
            industry_breakdown[column] = industry_breakdown[column].fillna(0).astype(int)
        industry_breakdown["employment"] = industry_breakdown["employment"].fillna(0.0)
    else:
        industry_breakdown["establishments"] = 0
    industry_breakdown = industry_breakdown.sort_values("employment", ascending=False)

    summary = {
        "isic_classes_matched": int(
            pd.concat(
                [
                    matched["codigo_clase"],
                    estab_matched["codigo_clase"] if not no_establishments else pd.Series(dtype=str),
                ]
            ).nunique()
        ),
        "provinces_with_activity": int(province_metrics["codigo_provincia"].nunique()),
        "enterprises": total_enterprises,
        "establishments": total_establishments,
        "employment": total_employment,
        "exporting_enterprises": total_exporters,
        "jobs_per_enterprise": total_employment / total_enterprises if total_enterprises else np.nan,
        "enterprises_share_national": total_enterprises / national_totals["enterprises"] if national_totals["enterprises"] else np.nan,
        "establishments_share_national": total_establishments / national_totals["establishments"] if national_totals["establishments"] else np.nan,
        "employment_share_national": total_employment / national_totals["employment"] if national_totals["employment"] else np.nan,
        "exporters_share_national": total_exporters / national_totals["exporting_enterprises"] if national_totals["exporting_enterprises"] else np.nan,
    }

    province_metrics = province_metrics.sort_values("codigo_provincia").reset_index(drop=True)

    return {
        "province_metrics": province_metrics,
        "summary": summary,
        "industry_breakdown": industry_breakdown.reset_index(drop=True),
    }


# --------------------------------------------------------------------------- #
# Page 4 — relatedness / opportunity metrics (arithmetic on notebook exports). #
# --------------------------------------------------------------------------- #


def top_neighbors(
    matrix: pd.DataFrame,
    focal: str,
    k: int = 12,
    exclude: set[str] | None = None,
) -> list[str]:
    """Top-``k`` proximity neighbors of ``focal`` (self and ``exclude`` removed).

    ``matrix`` is the symmetric Φ (already free of excluded classes). Returns
    class codes ordered by descending φ to the focal industry.
    """
    if focal not in matrix.index:
        return []
    row = matrix.loc[focal].drop(labels=[focal], errors="ignore")
    if exclude:
        row = row.drop(labels=[c for c in exclude if c in row.index], errors="ignore")
    row = row.dropna().sort_values(ascending=False)
    return list(row.head(k).index)


def aggregate_product_density(
    density_df: pd.DataFrame,
    selected_isic: list[str],
    value_col: str = "density_cont",
) -> pd.DataFrame:
    """Section 0: per-province mean density over the selected classes.

    ``value_col`` selects the density variant (``density_cont`` with-self, or
    ``density_cont_noself``). One row per province with the mean density (column
    ``value``) across the selected industries, the province's **best** relatedness
    state across those classes, and a count breakdown (specialized /
    present-not-specialized / absent). Sorted by ``value`` descending.
    """
    codes = sorted({str(c).zfill(4) for c in selected_isic})
    sub = density_df[density_df["codigo_clase"].isin(codes)].copy()
    columns = [
        "id_code", "id_name", "value", "best_state",
        "n_classes", "n_specialized", "n_present", "n_absent",
    ]
    if sub.empty:
        return pd.DataFrame(columns=columns)

    sub["state_rank"] = sub["state"].map(_STATE_RANK).fillna(0)
    agg = sub.groupby(["id_code", "id_name"], as_index=False).agg(
        value=(value_col, "mean"),
        best_rank=("state_rank", "max"),
        n_classes=("codigo_clase", "nunique"),
        n_specialized=("state", lambda s: int((s == "specialized").sum())),
        n_present=("state", lambda s: int((s == "present_not_spec").sum())),
        n_absent=("state", lambda s: int((s == "absent").sum())),
    )
    agg["best_state"] = agg["best_rank"].map(_RANK_STATE)
    agg = agg.drop(columns=["best_rank"])
    return agg.sort_values("value", ascending=False).reset_index(drop=True)[columns]


def build_opportunity_table(
    density_df: pd.DataFrame,
    tradability_df: pd.DataFrame,
    province: str,
    selected_isic: list[str],
    defn: str | None = "atlas",
    top_n: int = 15,
    value_col: str = "density_cont",
) -> pd.DataFrame:
    """Section B: the province's industries split into intensive / extensive margins.

    ``defn`` ∈ {"atlas", "regional", None} controls the tradability filter
    (``tradable_{defn} == "tradable"``), applied to **all** industries; ``None``
    applies no filter. Rows with no tradability record drop out when a filter is
    active. Within each RCA **state** (specialized / present_not_spec / absent)
    rows are ordered by ``value_col`` (``density_cont`` with-self, or
    ``density_cont_noself``) and truncated to ``top_n`` — the caller renders one
    bar chart per state. ``margin`` is retained for reference (intensive =
    present, extensive = absent). ``is_selected_product`` flags the product's own
    industries.
    """
    codes_sel = {str(c).zfill(4) for c in selected_isic}
    df = density_df[density_df["id_code"] == str(province).zfill(2)].copy()
    if df.empty:
        return df.assign(margin=[], is_selected_product=[])

    df["margin"] = np.where(df["state"] == "absent", "extensive", "intensive")
    trad_cols = [
        c for c in
        ["isic_code", "tradable_atlas", "tradable_regional", "confidence_atlas", "confidence_regional", "rationale"]
        if c in tradability_df.columns
    ]
    df = df.merge(
        tradability_df[trad_cols], left_on="codigo_clase", right_on="isic_code", how="left"
    )
    df["is_selected_product"] = df["codigo_clase"].isin(codes_sel)

    if defn in ("atlas", "regional"):
        column = f"tradable_{defn}"
        df = df[df[column].eq("tradable")].copy()

    df = df.sort_values(["state", value_col], ascending=[True, False])
    df = df.groupby("state", group_keys=False).head(top_n)
    return df.reset_index(drop=True)


def density_contributions(
    phi_row: pd.Series,
    province_df: pd.DataFrame,
    focal: str | None = None,
    top_n: int = 10,
    include_self: bool = True,
) -> pd.DataFrame:
    """Section A: decompose one (province, focal) density into its top
    ``φ_ij × M̃_j`` contributors.

    ``phi_row`` is the focal industry's proximity row (indexed by class code).
    ``province_df`` is the density table for that one province (needs
    ``codigo_clase`` and ``rca``). Continuous presence M̃ = rca / (1 + rca) —
    the same mapping the notebook's density uses, so the contribution shares
    reproduce the stored score exactly. With ``include_self=True`` the self-term
    (j = i, φ = 1) is kept and typically dominates; with ``include_self=False``
    (the no-self view) it is dropped from both the contributions and the
    normalization, so shares are over the *related* industries only. Shares sum
    to 1 across all contributors; the returned top-``top_n`` therefore sum to ≤ 1.
    """
    prov = province_df.copy()
    prov["codigo_clase"] = prov["codigo_clase"].astype(str).str.zfill(4)
    prov = prov.drop_duplicates(subset=["codigo_clase"]).set_index("codigo_clase")
    prov["m_tilde"] = prov["rca"] / (1.0 + prov["rca"])

    common = [c for c in phi_row.index if c in prov.index]
    if not include_self and focal is not None:
        common = [c for c in common if c != focal]
    phi = pd.to_numeric(phi_row.reindex(common), errors="coerce").fillna(0.0)
    m_tilde = prov.loc[common, "m_tilde"].fillna(0.0)
    numerator = phi.values * m_tilde.values
    total = float(numerator.sum())

    out = pd.DataFrame(
        {
            "codigo_clase": common,
            "phi": phi.values,
            "rca": prov.loc[common, "rca"].values,
            "m_tilde": m_tilde.values,
            "contribution": numerator,
            "contribution_share": numerator / total if total else 0.0,
        }
    )
    return out.sort_values("contribution", ascending=False).head(top_n).reset_index(drop=True)
