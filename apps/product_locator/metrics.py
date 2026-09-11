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


# --------------------------------------------------------------------------- #
# Bottom-up page — province composition / relative presence / opportunities.   #
# --------------------------------------------------------------------------- #


def relative_presence(
    comp: pd.DataFrame,
    province: str,
    level: str = "division",
    variable: str = "plazas_equiv",
    peer_codes: list[str] | None = None,
    rca_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Relative presence of every industry in ``province`` against a peer economy.

    ``relative presence = (share of the variable in the province) ÷ (share in the
    peer)`` — the location-quotient formula behind the RCA used elsewhere in the
    app. ``comp`` is :func:`data_loader.load_labor_composition` (all CIIU levels);
    ``variable`` is ``plazas_equiv`` (employment) or ``n_estab`` (establishments);
    shares are the within-geography shares shipped in that export, never
    recomputed.

    The peer is **Ecuador** (the export's ``'99'`` row) when ``peer_codes`` is
    ``None``, otherwise the aggregate of the given provinces — the caller passes
    the focal province's region **minus the focal province itself**, so a big
    province is never partly compared with itself. Peer shares are recomputed
    from the peer members' raw values (a sum of shares would be meaningless).

    ``rca_df`` (province × class rows of ``province_density_estab_count.csv``,
    columns ``codigo_clase`` + ``rca``) is consumed **only** at (class,
    ``n_estab``, Ecuador), the one combination for which a canonical RCA exists:
    there the shipped column replaces the composition-derived ratio so the chart
    is exactly the RCA behind the presence states. Every other combination has no
    canonical counterpart and is computed from the composition export.

    The industry universe is the peer's industry list, so industries **absent**
    from the province appear with ``value``/``n_estab``/``share_prov`` = 0 and
    ``rp`` = 0 (the caller decides how to show them; they cannot sit on a log
    axis). Industries present in the province but absent from the peer get
    ``rp = inf``.

    Both sides ship their **raw** counts as well as their shares
    (``value``/``n_estab`` for the province, ``value_peer``/``n_estab_peer`` for
    the peer): at class level a peer share rounds to 0.00% while the ratio is a
    perfectly ordinary ×3, and the counts are what stop that reading as a bug.

    Returns one row per industry: code, display_code, name_en, name_es,
    gsector_label, value, n_estab, ``plazas_equiv_guard``, share_prov, value_peer,
    n_estab_peer, share_peer, rp.

    ``plazas_equiv_guard`` is the province's FTE in the industry **regardless of
    which variable is being displayed**. The caller's support guard is the AND of an
    establishment floor and an employment floor (``config.MIN_ESTAB`` /
    ``MIN_PLAZAS``, the same guard the top-down lens applies), and on the
    establishments view ``value`` *is* ``n_estab`` — so the employment side of the
    guard needs its own column or the guard would silently change meaning when the
    user flips the display variable.
    """
    share_column = f"{variable}_share"
    frame = comp[comp["level"] == str(level)]

    label_columns = ["code", "display_code", "name_en", "name_es", "gsector_label"]
    peer_source = (
        frame[frame["id_code"] == "99"]
        if peer_codes is None
        else frame[frame["id_code"].isin([str(c).zfill(2) for c in peer_codes])]
    )
    if peer_codes is None:
        # Named aggregation, not a rename: on the establishments view
        # ``variable`` *is* ``n_estab`` and a rename would collapse the two.
        peer = peer_source[label_columns].copy()
        peer["value_peer"] = peer_source[variable].to_numpy()
        peer["n_estab_peer"] = peer_source["n_estab"].to_numpy()
        peer["share_peer"] = peer_source[share_column].to_numpy()
    else:
        # Aggregate the peer members, then take shares within that aggregate
        # (no shipped share exists for a region, unlike the '99' rows).
        totals = peer_source.groupby("code", as_index=False).agg(
            value_peer=(variable, "sum"), n_estab_peer=("n_estab", "sum")
        )
        peer_total = float(totals["value_peer"].sum())
        totals["share_peer"] = totals["value_peer"] / peer_total if peer_total else np.nan
        labels = (
            frame[frame["id_code"] == "99"][label_columns].drop_duplicates("code")
        )
        peer = labels.merge(totals, on="code", how="left")
        peer[["value_peer", "n_estab_peer", "share_peer"]] = peer[
            ["value_peer", "n_estab_peer", "share_peer"]
        ].fillna(0.0)

    # Built column by column: on the establishments view ``variable`` *is*
    # ``n_estab``, and a rename would collapse the two into one.
    prov_rows = frame[frame["id_code"] == str(province).zfill(2)]
    prov = pd.DataFrame(
        {
            "code": prov_rows["code"].to_numpy(),
            "value": prov_rows[variable].to_numpy(),
            "n_estab": prov_rows["n_estab"].to_numpy(),
            # Always the province's FTE, whichever variable is being displayed: the
            # support guard is the AND of an establishment floor and an employment
            # floor, and it must not change meaning when the user flips the display
            # variable (on the establishments view ``value`` *is* ``n_estab``).
            "plazas_equiv_guard": prov_rows["plazas_equiv"].to_numpy(),
            "share_prov": prov_rows[share_column].to_numpy(),
        }
    )

    out = peer.merge(prov, on="code", how="left")
    filled = ["value", "n_estab", "plazas_equiv_guard", "share_prov"]
    out[filled] = out[filled].fillna(0.0)
    out["rp"] = np.where(
        out["share_peer"] > 0,
        out["share_prov"] / out["share_peer"].replace({0.0: np.nan}),
        np.where(out["share_prov"] > 0, np.inf, np.nan),
    )

    canonical = str(level) == "class" and variable == "n_estab" and peer_codes is None
    if canonical and rca_df is not None:
        shipped = rca_df[["codigo_clase", "rca"]].drop_duplicates("codigo_clase")
        out = out.merge(shipped, left_on="code", right_on="codigo_clase", how="left")
        # The shipped RCA *is* the chart's value here — the same column, not an
        # approximation of it. Tripwire: the merge must cover every industry, or
        # the guarantee would silently degrade into the composition ratio.
        if out["rca"].isna().any():
            missing = out.loc[out["rca"].isna(), "code"].tolist()
            raise ValueError(
                "relative_presence: the shipped RCA does not cover every class of "
                f"province {province} ({len(missing)} missing, e.g. {missing[:5]})"
            )
        out["rp"] = out["rca"]
        out = out.drop(columns=["codigo_clase", "rca"])

    columns = [
        "code", "display_code", "name_en", "name_es", "gsector_label",
        "value", "n_estab", "plazas_equiv_guard", "share_prov",
        "value_peer", "n_estab_peer", "share_peer", "rp",
    ]
    return out[columns].reset_index(drop=True)


def capability_neighbors(
    phi_row: pd.Series,
    province_df: pd.DataFrame,
    focal: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """The industries most **similar** to ``focal``, with what the province has of each.

    Ranked by raw proximity φ — deliberately *not* by contribution to the density
    score. The two answer different questions: a contribution is
    ``φ × presence``, so a merely-similar industry the province happens to be big
    in outranks the industry that actually shares the capabilities. For "what do
    we already have that is like this?" the similarity ranking is the honest one,
    and the province's presence is then read off each row rather than baked into
    the order.

    ``province_df`` is one province's slice of the density table (needs
    ``codigo_clase``, ``rca``, ``state``, ``intensity``). Returns the top-``top_n``
    neighbours (self excluded): codigo_clase, phi, rca, state, intensity,
    ``has_it`` (the province has any establishments at all in it).
    """
    row = phi_row.drop(labels=[focal], errors="ignore").dropna().sort_values(ascending=False)
    out = pd.DataFrame({"codigo_clase": row.index.astype(str), "phi": row.to_numpy()})
    prov = province_df.drop_duplicates("codigo_clase").set_index("codigo_clase")
    out["rca"] = out["codigo_clase"].map(prov["rca"])
    out["state"] = out["codigo_clase"].map(prov["state"])
    out["intensity"] = pd.to_numeric(out["codigo_clase"].map(prov["intensity"]), errors="coerce")
    out["has_it"] = out["intensity"].fillna(0) > 0
    return out.head(int(top_n)).reset_index(drop=True)


def plotted_mask(plane: pd.DataFrame, var: str, density_col: str = "density") -> pd.Series:
    """The classes that can sit on the feasibility × attractiveness plane.

    One definition, shared by :func:`feasibility_mix` and
    :func:`opportunity_score` so the x-axis and the score are always ranked over
    the *same* set. A class needs both a density in this province and the chosen
    attractiveness metric; the six wage-less classes have neither metric and are
    therefore unplottable and unscorable on either.
    """
    return plane[density_col].notna() & plane[var].notna()


def feasibility_mix(
    plane: pd.DataFrame,
    var: str,
    beta: float = 0.5,
    rca_col: str = "rca",
    density_col: str = "density",
) -> pd.Series:
    """The page's **feasibility** definition: an explicit, user-owned mix of own
    presence and related capabilities.

    ``β · pctile(rca) + (1 − β) · pctile(related-only density)``, both percentile
    ranks over the plotted classes (:func:`plotted_mask`), so the result is in
    0-1 rank space and is directly comparable across β.

    **Why a mix rather than the with-self density.** With-self density already
    *is* a composite of the same two ingredients — a median 79% self-term for a
    specialized industry, Spearman 0.991 with RCA in Pichincha — but a hidden one
    whose ratio varies class by class with whatever share φ's diagonal happens to
    carry. The β mix makes that trade-off explicit, constant across industries and
    the user's to set. It also genuinely adds information: within Cotopaxi's
    presence states RCA and related-only density correlate at only 0.11-0.29.

    At β = 1 the ordering is RCA's, at β = 0 it is related-only density's. Within
    the **absent** state RCA is uniformly 0, so the mix degenerates to
    density-only ordering there — intended: an absent industry has no own presence
    to weigh, and β cannot reorder a constant.

    Returns a Series aligned to ``plane``'s index, NaN off the plotted set.
    """
    plotted = plotted_mask(plane, var, density_col)
    mix = pd.Series(np.nan, index=plane.index, dtype="float64")
    if not plotted.any():
        return mix
    rca_pctile = plane.loc[plotted, rca_col].rank(pct=True)
    density_pctile = plane.loc[plotted, density_col].rank(pct=True)
    mix.loc[plotted] = float(beta) * rca_pctile + (1.0 - float(beta)) * density_pctile
    return mix


def opportunity_score(
    plane: pd.DataFrame,
    var: str,
    weight: float = 0.5,
    feasibility_col: str = "x",
    density_col: str = "density",
) -> pd.Series:
    """Rank-space feasibility × attractiveness score, for **ranking only**.

    ``weight · feasibility + (1 − weight) · pctile(``var``, national)``, where
    ``feasibility`` is :func:`feasibility_mix`'s output (already a 0-1 rank-space
    quantity, so it is *not* re-ranked — re-ranking a convex combination of two
    percentile ranks would throw away the interpolation β bought) and the
    attractiveness percentile is taken over the plotted classes. Raw units are
    incommensurable (a density of 0.6 against $12,000 per FTE) and jobs per firm
    is log-skewed, so rank space is the only honest mix.

    The score follows every control the caller exposes — attractiveness metric, β,
    weight, and the menu filter that decides which classes are ranked at all — by
    design. Never a truth claim and never exported: it orders the highlighted set,
    the plane and the table show the underlying values. Returns a Series aligned
    to ``plane``'s index, NaN where either input is missing.
    """
    plotted = plotted_mask(plane, var, density_col)
    score = pd.Series(np.nan, index=plane.index, dtype="float64")
    if not plotted.any():
        return score
    attractiveness_pctile = plane.loc[plotted, var].rank(pct=True)
    score.loc[plotted] = (
        weight * plane.loc[plotted, feasibility_col] + (1.0 - weight) * attractiveness_pctile
    )
    return score


def support_eligible(
    state: pd.Series,
    n_estab: pd.Series,
    plazas: pd.Series,
    min_estab: int,
    min_plazas: float,
) -> pd.Series:
    """Is a (province, industry) cell supported well enough to be **recommended**?

    The bottom-up lens's shortlist guard, and the single implementation of it —
    consumed by :func:`province_shortlist_matrix` and by the explorer's Section 3
    alike, so the cross-province grid and the plane cannot drift apart (the role
    :func:`presence_frame` plays for the top-down lens).

    A cell is **thin** when it has *both* fewer than ``min_estab`` units *and*
    fewer than ``min_plazas`` FTE — the same AND floors as everywhere else in the
    app: a cell survives on either enough units or enough workers, and is noise
    only when both are thin. An **absent** cell is never gated: zero presence by
    construction is what an entry candidate *is*, and gating it would throw away
    the whole extensive margin.

    **Eligibility only, never a demotion.** The returned mask decides membership
    of the highlighted set and nothing else — ineligible rows keep their exported
    ``state``, stay on the plane as context dots and stay in the percentile-rank
    universe (:func:`plotted_mask`), so no feasibility, score or raw value of any
    *other* row moves when the guard is flipped. Rewriting ``state`` instead would
    contradict the page's "evidence stays as exported" rule *and* let a
    two-establishment industry re-enter as a fake entry candidate.

    Returns a plain-``bool`` Series aligned to ``state``'s index (True = may be
    recommended). Plain rather than nullable on purpose: it is used directly as a
    row mask, and a nullable mask raises on any NA it happens to carry.
    """
    states = pd.Series(state)
    estab = pd.to_numeric(pd.Series(n_estab), errors="coerce").fillna(0.0).to_numpy()
    fte = pd.to_numeric(pd.Series(plazas), errors="coerce").fillna(0.0).to_numpy()
    thin = (estab < float(min_estab)) & (fte < float(min_plazas))
    # An unrecognised or missing state counts as *not* absent, so a thin cell with a
    # broken label is held out rather than waved through as an entry candidate.
    absent = states.astype(str).str.strip().to_numpy() == "absent"
    return pd.Series(~(thin & ~absent), index=states.index, dtype=bool)


def connected_eligible(thin_neighborhood: pd.Series) -> pd.Series:
    """Is an industry's related-capability score measurable enough to **recommend** on?

    The bottom-up lens's second eligibility guard and the single implementation of it,
    the per-**industry** counterpart of :func:`support_eligible`'s per-cell rule. Reads
    the ``thin_neighborhood`` flag the caller derives from the shipped
    ``phi_mass_noself`` column (``< THIN_NEIGHBORHOOD_THRESHOLD``) — it performs no
    arithmetic of its own, and nothing here recomputes proximity.

    Why it exists: related-only density is the only density the lens carries, so for a
    class with almost no off-diagonal proximity mass that tiny mass is the *whole*
    denominator — a ratio of two very small numbers that moves on almost nothing.

    **Connectedness is a property of the industry, not of the (province, industry)
    cell** — φ is measured nationally — which is why this takes one flag column rather
    than a province's counts, and why :func:`province_shortlist_matrix` applies it as a
    class-level exclusion.

    A **missing** flag counts as *connected*: ``NaN < threshold`` is False, so a class
    the industry-space export does not cover keeps the reading the flag has always
    given it. Treating unknown as ineligible would let the guard move membership for a
    reason that has nothing to do with connectedness.

    **Eligibility only, never a demotion** — the same contract as
    :func:`support_eligible`, and it holds for the same reason: the mask is applied
    after scoring and before the top-k cut, so no survivor's feasibility, score or raw
    value moves when it is flipped.

    Returns a plain-``bool`` Series aligned to the input's index (True = may be
    recommended).
    """
    # Via the nullable dtype: a plain ``fillna`` on an object column warns about a
    # silent downcast, and the real column is already boolean.
    return ~pd.Series(thin_neighborhood).astype("boolean").fillna(False).astype(bool)


def province_shortlist_matrix(
    profile: pd.DataFrame,
    density: pd.DataFrame,
    provinces: list[str],
    var: str,
    beta: float = 0.5,
    weight: float = 0.5,
    top_k: int = 5,
    universe: set[str] | frozenset[str] | None = None,
    *,
    emp: pd.DataFrame,
    min_estab: int,
    min_plazas: float,
    apply_support: bool = True,
    ineligible_classes: set[str] | frozenset[str] | None = None,
) -> dict[str, object]:
    """Every province's shortlist at once — the bottom-up lens's cross-province summary.

    Replays the explorer's ranking (:func:`feasibility_mix` → :func:`opportunity_score` →
    top-``top_k`` per presence state) for each of ``provinces`` under one set of parameters, and
    returns the union as an industry × province matrix.

    **A membership matrix, not a count matrix.** The top-down sibling counts, per (industry,
    province) cell, how many of the 50 opportunities assign that province — a number
    that can exceed 1. There are no products here, and an industry has exactly one presence state
    per province, so a cell is either a candidate or not. The cell therefore carries the **state**
    (specialized 2 / present-not-specialized 1 / absent 0, NaN where the industry is not on that
    province's shortlist), which is strictly more informative than a binary mark: measured on the
    real data at the lens's shipped defaults (k = 10, the tradable menu and both guards on),
    **65% of candidate industries appear in more than one state across provinces** (71 of 109;
    it was 44% at the pre-v4 k = 5, so the finding survives the settings) — an entry
    opportunity in one province and a deepening play in another, and the finding a single-province
    page structurally cannot show.

    **And there are no blank columns**, unlike the sibling. Every province gets a shortlist by
    construction, so "the provinces no ranking reaches" — the point of the top-down heatmap — has
    no counterpart here. The readings that replace it are **row breadth** (how many provinces an
    industry is a candidate in, separating ubiquitous from place-specific) and the state mixing
    above. A province contributes fewer than ``3 × top_k`` marks only where a state has fewer than
    ``top_k`` scorable members, or none at all.

    ``universe`` restricts the industries that may be recommended (the tradable menu); it is
    applied per province **before** the percentile ranks, exactly as the explorer does, so the
    ranks run over the plotted set.

    ``ineligible_classes`` is the opposite in every respect and the distinction matters: it is a
    set of classes that may not be recommended **anywhere**, applied per province **after** the
    percentile ranks and before the top-``top_k`` cut — the same position as the support guard, and
    the reason no survivor's feasibility or score moves when it is passed. It carries the
    connectedness guard (:func:`connected_eligible`), which is a property of the *industry* rather
    than of the cell, so the caller resolves it once. Passing it through ``universe`` instead would
    move the percentile ranks and make this grid disagree with the explorer's plane about the very
    cells it exists to replay. Stated as an **exclusion** rather than an allow-list on purpose: a
    class missing from the industry-space export is then left alone rather than silently dropped,
    matching :func:`connected_eligible`'s NaN semantics exactly.

    **Scorable is not the same as recommendable.** ``emp`` (``id_code`` × ``codigo_clase`` →
    ``plazas_equiv``, ``n_estab`` — :func:`data_loader.load_labor_class_employment`) carries the
    province's own activity in each class, and with ``apply_support`` the support guard
    (:func:`support_eligible`) drops thinly-supported non-absent cells from the pick **between
    scoring and the top-``top_k`` cut**. Everything ranked stays ranked: the guard removes
    candidates, never percentile-rank members, so every survivor's feasibility and score are
    identical either way. Without it a two-establishment industry could be a province's headline
    "specialization" — the defect this parameter exists to close, and the reason the floors are
    mandatory arguments rather than defaults duplicated out of ``config``.

    Returns ``marks`` (long form: one row per (industry, province) candidate with its state,
    score and rank within its state there), ``state_matrix`` (industries × provinces of state
    rank), ``breadth``, ``state_mix`` (distinct states per industry) and ``per_province`` (marks
    per province). ``rank_in_state`` is the row's position among the province's **eligible**
    classes in that state, so a shortlist's ranks run 1…``top_k`` without gaps.
    """
    columns = ["codigo_clase", "rca", "state", "density_cont_noself"]
    emp_columns = ["codigo_clase", "plazas_equiv", "n_estab"]
    rows: list[pd.DataFrame] = []
    for province in provinces:
        slice_ = density.loc[density["id_code"] == str(province), columns].rename(
            columns={"density_cont_noself": "density"}
        )
        plane = profile.merge(slice_, on="codigo_clase", how="inner")
        if universe is not None:
            plane = plane[plane["codigo_clase"].isin(universe)].copy()
        # The province's own activity, for the support guard. The employment export
        # carries present combinations only, so an unmatched class is a genuine zero.
        here = emp.loc[emp["id_code"] == str(province), emp_columns].rename(
            columns={"plazas_equiv": "plazas_equiv_prov", "n_estab": "n_estab_prov"}
        )
        plane = plane.merge(here, on="codigo_clase", how="left")
        plane[["plazas_equiv_prov", "n_estab_prov"]] = plane[
            ["plazas_equiv_prov", "n_estab_prov"]
        ].fillna(0.0)
        plane["x"] = feasibility_mix(plane, var, beta)
        plane["score"] = opportunity_score(plane, var, weight, feasibility_col="x")
        scored = plane.dropna(subset=["score"]).sort_values("score", ascending=False)
        # Eligibility, applied after scoring and before the cut: the ranks above were
        # taken over every scorable class (the display-only guarantee), and what the
        # guard removes is only the right to be recommended.
        if apply_support:
            scored = scored[
                support_eligible(
                    scored["state"], scored["n_estab_prov"], scored["plazas_equiv_prov"],
                    min_estab, min_plazas,
                )
            ]
        # The second guard, at the same point in the pipeline for the same reason.
        if ineligible_classes:
            scored = scored[~scored["codigo_clase"].isin(ineligible_classes)]
        if scored.empty:
            continue
        scored = scored.copy()
        scored["rank_in_state"] = scored.groupby("state").cumcount() + 1
        picks = scored.groupby("state", group_keys=False).head(int(top_k)).copy()
        picks["id_code"] = str(province)
        rows.append(
            picks[[
                "id_code", "codigo_clase", "display_code", "class_name", "gsector_label",
                "state", "score", "x", "rca", "density", "rank_in_state",
            ]]
        )

    marks = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["id_code", "codigo_clase", "state", "score", "rank_in_state"])
    )
    if marks.empty:
        return {
            "marks": marks, "state_matrix": pd.DataFrame(),
            "breadth": pd.Series(dtype="int64"), "state_mix": pd.Series(dtype="int64"),
            "per_province": pd.Series(dtype="int64"),
        }

    marks["state_rank"] = marks["state"].map(_STATE_RANK)
    state_matrix = marks.pivot(index="codigo_clase", columns="id_code", values="state_rank")
    state_matrix = state_matrix.reindex(columns=[str(p) for p in provinces])
    return {
        "marks": marks,
        "state_matrix": state_matrix,
        "breadth": marks.groupby("codigo_clase")["id_code"].nunique(),
        "state_mix": marks.groupby("codigo_clase")["state"].nunique(),
        "per_province": marks.groupby("id_code")["codigo_clase"].nunique(),
    }


# --------------------------------------------------------------------------- #
# The potential / gap benchmark — Ecuador's own frontier as the yardstick.      #
# --------------------------------------------------------------------------- #


def province_employment_totals(rca: pd.DataFrame) -> pd.Series:
    """Each province's total establishment-booked FTE over the app's class universe.

    The denominator of every share-of-economy figure in the benchmark below, and
    the multiplier that turns a leader's share back into a headcount for another
    province. One implementation so the two can never be computed off different
    universes: ``rca`` is :func:`data_loader.load_province_rca`, which has already
    dropped ``EXCLUDED_CLASSES``, so this total is the app's economy rather than
    the registry's.
    """
    return rca.groupby("id_code")["plazas_equiv"].sum()


def leader_benchmark(rca: pd.DataFrame, min_plazas: float) -> pd.DataFrame:
    """The province that holds the largest share of its own economy in each industry.

    An **existence proof**, not a forecast: for industry *i* the leader is the
    province where *i* accounts for the biggest slice of local employment, so
    ``leader_share`` is the highest intensity any Ecuadorian province has actually
    reached. Multiplied by another province's total FTE it answers "what would
    this industry weigh here if we were as specialized in it as the best province
    in the country?" — a benchmark built entirely from observed Ecuadorian
    outcomes, with no cross-country concordance in it.

    **The pool is an employment floor, not the app's AND guard** (``min_plazas``
    FTE in the cell, ``config.MIN_PLAZAS``). The quantity being benchmarked *is*
    an employment share, and the AND guard admits cells that clear the
    establishment floor on almost no employment: under it the malt-liquors
    frontier is Galápagos at 7.5 FTE across 9 craft breweries, and 16 of 401
    leaders stand on two establishments or fewer. The floor costs coverage —
    **366 of 412 classes** have a qualifying cell — and the classes it leaves out
    get no tile and a counting caption, rather than a "frontier" the rest of the
    app would call noise.

    A province that is **its own leader** has a gap of exactly 0 — it *is* the
    national frontier, and the caller names those rows rather than rendering an
    empty tile.

    **The leader is not guaranteed to be specialized in the industry.** The
    argument that it must be — a maximum exceeds the employment-weighted mean, so
    the winner's location quotient is ≥ 1 — holds for the argmax over *all* 24
    provinces, and the floor takes the argmax over the pool instead. Measured
    2026-08-24: it holds in **365 of the 366** covered classes; the exception is
    `C1623` wooden containers, where **only Guayas clears the floor** (50.25 FTE)
    while Los Ríos runs six times the intensity on 36.5 FTE, leaving the pool
    winner at a plazas-base LQ of 0.74. ``n_qualifying`` is returned so a caller
    can see when a "frontier" rests on a single qualifying province.

    ``rca`` is :func:`data_loader.load_province_rca` (dense province × class, 24
    provinces, no ``'99'`` row) and must carry ``plazas_equiv`` and
    ``num_establecimientos``. Returns one row per class **with a qualifying
    cell**, indexed by ``codigo_clase``: ``leader_id_code``, ``leader_id_name``,
    ``leader_share``, ``leader_plazas``, ``leader_estab``, ``n_qualifying``.
    """
    totals = province_employment_totals(rca)
    pool = rca.loc[pd.to_numeric(rca["plazas_equiv"], errors="coerce") >= float(min_plazas)].copy()
    columns = ["leader_id_code", "leader_id_name", "leader_share", "leader_plazas",
               "leader_estab", "n_qualifying"]
    if pool.empty:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="codigo_clase"))
    pool["share_econ"] = pool["plazas_equiv"] / pool["id_code"].map(totals)
    winners = pool.loc[pool.groupby("codigo_clase")["share_econ"].idxmax()]
    out = pd.DataFrame(
        {
            "leader_id_code": winners["id_code"].to_numpy(),
            "leader_id_name": winners["id_name"].astype(str).str.strip().to_numpy(),
            "leader_share": winners["share_econ"].to_numpy(),
            "leader_plazas": winners["plazas_equiv"].to_numpy(),
            "leader_estab": winners["num_establecimientos"].to_numpy(),
        },
        index=pd.Index(winners["codigo_clase"].to_numpy(), name="codigo_clase"),
    )
    out["n_qualifying"] = pool.groupby("codigo_clase")["id_code"].nunique().reindex(out.index)
    # Tripwire: no benchmark may stand on less employment than the floor it was
    # selected under — the whole credibility of the "existence proof" rests on it.
    if float(out["leader_plazas"].min()) < float(min_plazas):
        raise ValueError(
            "leader_benchmark: a leader stands on "
            f"{float(out['leader_plazas'].min()):,.1f} FTE, under the {min_plazas:,.0f} floor"
        )
    return out[columns].sort_index()


def potential_gap(
    benchmark: pd.DataFrame,
    codes: list[str],
    province: str,
    province_total: float,
    current: pd.Series,
) -> pd.DataFrame:
    """``potential = leader_share × T(province)``; ``gap = max(0, potential − current)``.

    The arithmetic behind the potential/gap treemap, kept pure so it can be tested
    without a data bundle. ``codes`` is the shortlist being priced,
    ``province_total`` is the province's own total FTE
    (:func:`province_employment_totals`) and ``current`` maps class → the FTE the
    province holds today.

    **The clamp is not cosmetic, and it means something specific.** A province can
    hold *more* FTE than ``potential`` — a negative raw gap — and the only way it
    happens is that **its own cell was below the benchmark's employment floor**, so
    it never entered the pool the maximum was taken over and its intensity is free
    to exceed the pool winner's. Measured 2026-08-24: 146 such (province, class)
    cells nationally, every one of them under 50 FTE (the largest is 49.75). The
    reading is "more concentrated in this than the frontier province, on too little
    employment to *be* the frontier" — not "already the frontier", which is what
    ``is_self`` means. Both are flagged (``at_frontier`` covers either), because a
    negative gap left unclamped would render as a missing tile that reads like no
    opportunity at all.

    Classes with **no qualifying benchmark are absent from the result** — no
    fallback below the floor the rest of the app calls noise; the caller counts
    and names them. Returns one row per priced class, indexed by
    ``codigo_clase``: the ``leader_*`` columns, ``current``, ``potential``,
    ``gap``, ``is_self`` and ``at_frontier``.
    """
    wanted = [str(code) for code in codes]
    priced = benchmark.reindex([c for c in wanted if c in benchmark.index]).copy()
    if priced.empty:
        for column in ["current", "potential", "gap", "is_self", "at_frontier"]:
            priced[column] = pd.Series(dtype="float64" if column != "is_self" else "bool")
        return priced
    here = pd.to_numeric(pd.Series(current), errors="coerce")
    priced["current"] = pd.Series(priced.index.map(here), index=priced.index).astype("float64")
    priced["current"] = priced["current"].fillna(0.0)
    priced["potential"] = priced["leader_share"].astype("float64") * float(province_total)
    priced["gap"] = (priced["potential"] - priced["current"]).clip(lower=0.0)
    priced["is_self"] = priced["leader_id_code"].astype(str) == str(province)
    priced["at_frontier"] = priced["is_self"] | (priced["gap"] <= 0.0)
    return priced


# --------------------------------------------------------------------------- #
# Top-down lens — opportunity -> its industries -> the provinces that have them #
# --------------------------------------------------------------------------- #

# The 4-way (view, base) -> column mapping, in one place. "lq" is the location
# quotient (1 = the national average), "scale" the province's share of the
# industry's national total (sums to 1 across the 24 provinces).
PRESENCE_COLUMNS: dict[tuple[str, str], str] = {
    ("lq", "plazas"): "rca_plazas",
    ("lq", "estab"): "rca_estab_count",
    ("scale", "plazas"): "share_national_plazas",
    ("scale", "estab"): "share_national_estab_count",
}


def presence_column(view: str, base: str) -> str:
    """Column of the province × class export that carries one presence measure."""
    try:
        return PRESENCE_COLUMNS[(str(view), str(base))]
    except KeyError:  # pragma: no cover - guarded by the UI's fixed options
        raise ValueError(f"unknown presence view/base: {view!r}/{base!r}") from None


def presence_frame(
    rca: pd.DataFrame,
    view: str,
    base: str,
    apply_support: bool,
    min_estab: int,
    min_plazas: float,
) -> dict[str, pd.DataFrame]:
    """Wide class × province presence, with the support guard already applied.

    The single implementation of the guard, shared by the per-product ranking and
    the all-products matrix so the two can never drift apart. A cell is **thin**
    when it has *both* fewer than ``min_estab`` units *and* fewer than
    ``min_plazas`` FTE — an AND, not the export's establishment-only
    ``low_support`` and not the OR used for wage reliability elsewhere in the
    repo: a cell survives on either enough units or enough workers, and is noise
    only when both are thin.

    Returns the guarded ``presence`` frame (thin cells set to 0, so they
    contribute nothing to any weighted sum), the unguarded ``raw`` frame, the
    boolean ``thin`` mask, and the underlying ``plazas`` / ``estab`` counts —
    all indexed by 4-digit class, columned by 2-digit province code.
    """
    column = presence_column(view, base)
    wide = {
        name: rca.pivot(index="codigo_clase", columns="id_code", values=source).fillna(0.0)
        for name, source in [
            ("raw", column), ("plazas", "plazas_equiv"), ("estab", "num_establecimientos")
        ]
    }
    thin = (wide["estab"] < float(min_estab)) & (wide["plazas"] < float(min_plazas))
    presence = wide["raw"].where(~thin, 0.0) if apply_support else wide["raw"].copy()
    return {"presence": presence, "raw": wide["raw"], "thin": thin,
            "plazas": wide["plazas"], "estab": wide["estab"]}


def product_presence(
    opps: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    hs4: str,
    route: str,
    min_weight: float,
    material_share: float = 0.10,
) -> dict[str, object]:
    """Per-province weighted presence of one opportunity's industries, one route.

    ``P_route(province) = Σ_i w_i · presence(industry_i, province)`` with
    ``w_i = crosswalk_weight_used``. Weighting is not optional: unweighted
    averaging lets a 1.6%-weight industry vote as loudly as a 95% one.

    **The anchor route is averaged across anchors, never summed.** The weights
    sum to 1 within (product, ``producto``) but to ``n_anchors`` within (product,
    ``ancla``) — each anchor carries its own weights summing to 1 — so summing
    would inflate the anchor route by the anchor count, which is bookkeeping
    rather than evidence. Each anchor's own weighted vector is built first, then
    the anchors are averaged with equal weight. (Proximity-weighting the anchors
    was tested and gives an almost identical answer; the simple average is the
    settled choice.)

    Industries below ``min_weight`` are excluded from the arithmetic — they are
    HS4-aggregation residue — but are returned in ``industries`` flagged
    ``below_floor`` so the page can show them greyed rather than hide them.
    An absent or guard-filtered industry contributes **0 and the weights are not
    renormalized**: a province that lacks one of the product's industries
    genuinely has less of the capability.

    ``frames`` is :func:`presence_frame`'s output. Returns the province vector
    plus everything the page has to disclose: the industry list, the
    province × industry detail, how many populated cells the guard removed, and
    which of those held at least ``material_share`` of their industry's national
    employment.
    """
    provinces = list(frames["presence"].columns)
    is_anchor_route = str(route) == "ancla"
    rows = opps[(opps["opportunity_hs4_code"] == str(hs4)) & (opps["route"] == str(route))]
    empty: dict[str, object] = {
        "presence": pd.Series(0.0, index=provinces, dtype="float64"),
        "industries": pd.DataFrame(), "detail": pd.DataFrame(), "n_removed": 0,
        "material_removed": pd.DataFrame(), "n_observable": 0, "n_anchors": 0,
    }
    if rows.empty:
        return empty

    kept = rows[rows["crosswalk_weight_used"] >= float(min_weight)]
    n_anchors = int(rows["anchor_hs4_code"].nunique()) if is_anchor_route else 0

    # Effective weight per industry — the coefficient it actually carries in the
    # route vector. Averaging the anchors is linear, so
    #   (1/n) · Σ_a Σ_i w_{a,i}·p_i  ==  Σ_i [(1/n) · Σ_a w_{a,i}] · p_i,
    # i.e. dividing each industry's summed weight by the anchor count is the same
    # arithmetic as building each anchor's vector and averaging them. The divisor
    # is the anchors the product *has*, not the ones that survived the weight
    # floor: an anchor whose industries are all residue contributes nothing,
    # exactly like an absent industry, and is not renormalized away.
    weights = kept.groupby("isic4_code")["crosswalk_weight_used"].sum()
    if is_anchor_route and n_anchors:
        weights = weights / n_anchors

    total = pd.Series(0.0, index=provinces, dtype="float64")
    detail_rows: list[pd.DataFrame] = []
    removed, material = 0, []
    national_plazas = frames["plazas"].sum(axis=1)
    for code, weight in weights.items():
        if code not in frames["presence"].index:
            continue
        used = frames["presence"].loc[code]
        total += float(weight) * used
        plazas, estab, thin = (frames[key].loc[code] for key in ("plazas", "estab", "thin"))
        detail_rows.append(
            pd.DataFrame({
                "id_code": provinces,
                "isic4_code": code,
                "route": str(route),
                "weight": float(weight),
                "presence_raw": frames["raw"].loc[code].to_numpy(),
                "presence_used": used.to_numpy(),
                "contribution": (float(weight) * used).to_numpy(),
                "plazas_equiv": plazas.to_numpy(),
                "num_establecimientos": estab.to_numpy(),
                "thin": thin.to_numpy(),
            })
        )
        # Only populated cells count as "removed" — an empty cell was never an
        # observation the guard could take away.
        cut = thin & ((plazas > 0) | (estab > 0))
        removed += int(cut.sum())
        national = float(national_plazas.get(code, 0.0))
        if national > 0:
            share = plazas / national
            flagged = cut & (share >= float(material_share))
            for province in flagged.index[flagged.to_numpy()]:
                material.append({
                    "id_code": province, "isic4_code": code,
                    "plazas_equiv": float(plazas[province]),
                    "num_establecimientos": float(estab[province]),
                    "share_national": float(share[province]),
                })

    # One row per (anchor, industry) so the anchor block can name its products;
    # `effective_weight` is what that industry actually contributes to the route.
    industries = rows.drop_duplicates(["anchor_hs4_code", "isic4_code"]).copy()
    industries["below_floor"] = industries["crosswalk_weight_used"] < float(min_weight)
    industries["effective_weight"] = industries["isic4_code"].map(weights).fillna(0.0)
    industries["national_plazas"] = industries["isic4_code"].map(national_plazas).fillna(0.0)
    industries["national_estab"] = (
        industries["isic4_code"].map(frames["estab"].sum(axis=1)).fillna(0.0)
    )

    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    if not detail.empty:
        names = rows.drop_duplicates("isic4_code").set_index("isic4_code")["isic4_name"]
        detail["isic4_name"] = detail["isic4_code"].map(names)
    material_frame = pd.DataFrame(material)
    if not material_frame.empty:
        names = rows.drop_duplicates("isic4_code").set_index("isic4_code")["isic4_name"]
        material_frame["isic4_name"] = material_frame["isic4_code"].map(names)

    return {
        "presence": total,
        "industries": industries,
        "detail": detail,
        "n_removed": removed,
        "material_removed": material_frame,
        "n_observable": int((total > 0).sum()),
        "n_anchors": n_anchors,
    }


def blend_routes(direct: pd.Series, anchor: pd.Series, w: float) -> dict[str, object]:
    """``w · P_producto + (1 − w) · P_ancla`` — a straight weighted average.

    **Neither route is renormalized, in either view.** Both presence measures are
    already scale-free (LQ because 1 is the national average; the national shares
    because each industry's shares sum to 1 across the provinces, so any weighted
    combination of industries does too). With the guard on, the two routes do
    carry unequal mass and the *effective* weight drifts from the nominal one —
    that is intended, not drift to correct: where the guard removed a route's
    cells we genuinely cannot observe it there, so it should count for less, and
    renormalizing would inflate the survivors to imply coverage we do not have.
    The caller surfaces ``effective_w`` and the observable-province counts rather
    than correcting them.
    """
    weight = float(w)
    score = weight * direct.astype(float) + (1.0 - weight) * anchor.astype(float)
    mass_direct = float(direct.sum())
    mass_anchor = float(anchor.sum())
    blended_mass = weight * mass_direct + (1.0 - weight) * mass_anchor
    return {
        "score": score,
        "n_observable_direct": int((direct > 0).sum()),
        "n_observable_anchor": int((anchor > 0).sum()),
        "mass_direct": mass_direct,
        "mass_anchor": mass_anchor,
        "effective_w": (weight * mass_direct / blended_mass) if blended_mass else np.nan,
    }


def parity_threshold(view: str, n_provinces: int) -> float:
    """The score a province posts when nothing about it is special.

    The assignment bar, expressed in the units of whichever presence measure is
    on screen:

    * **lq** — ``1.0``. A weighted average of location quotients whose weights
      sum to 1 is itself a location quotient, so 1 is exactly "this province
      holds as much of the product's industry mix as the country does on
      average". That makes the rule the project's own ``RCA >= 1`` test.
    * **scale** — ``1 / n_provinces``. A weighted average of national shares
      sums to 1 across *all* provinces, so no single one can approach 1; the
      honest analogue is an equal share of the industry. (Under the support
      guard the vector sums to slightly under 1 — HS 8418 reaches 0.990 — so
      this bar is marginally strict, which is the safe direction.)
    """
    return 1.0 if str(view) == "lq" else 1.0 / max(int(n_provinces), 1)


def qualifying_provinces(
    score: pd.Series,
    threshold: float,
    size: pd.Series | None = None,
    excluded: frozenset[str] | set[str] | None = None,
) -> list[str]:
    """Every province whose weighted presence reaches ``threshold``, best first.

    Replaces a fixed top-k, so **how many provinces a product reaches is a
    finding rather than a setting** — 2 to 9 of them at the defaults, median 4,
    191 assignments across the 50 against the 150 a flat top-3 produced. The
    comparison is ``>=``, matching the repo's ``RCA ≥ 1`` convention; measured,
    ``>`` and ``>=`` differ for not a single (product, province) pair, because
    no score lands exactly on 1.

    A province scoring **0 or less is never assigned**, whatever the threshold:
    zero means no industry of the product is observable there, so admitting it
    would be an artifact of a loosened dial rather than evidence.

    Ordering is score-descending with the same deterministic tie-break as before
    — score, then ``size`` (the province's employment behind the product's
    industries), then ``id_code`` — because position still carries meaning: it is
    the rank the product grid prints in each cell.

    **This is the lens's single assignment gate**, and therefore the one place a
    province can be withheld from the top-down lens: ``excluded`` (the app passes
    ``config.TD_EXCLUDED_PROVINCES``, currently Galápagos) drops those provinces
    before the threshold test, so they are never assigned an opportunity on Page A,
    in :func:`assignment_matrix`, in :func:`top_down_candidates`, or in the
    committed candidate list the builder writes.

    The exclusion is **assignment only, never the arithmetic**. An excluded
    province keeps its column in :func:`presence_frame` and so goes on counting in
    the national employment/establishment denominators
    (:func:`product_presence`) and in the scale-view parity denominator
    (:func:`parity_threshold`) — it is removed from the answer, not from the
    evidence. Dropping it upstream instead would silently re-base every "national"
    figure the lens prints.
    """
    if excluded:
        score = score.drop(index=[c for c in excluded if c in score.index])
    frame = pd.DataFrame({"score": pd.to_numeric(score, errors="coerce").fillna(0.0)})
    frame["size"] = (
        pd.to_numeric(size, errors="coerce").reindex(frame.index).fillna(0.0)
        if size is not None
        else 0.0
    )
    frame = frame[(frame["score"] >= float(threshold)) & (frame["score"] > 0)]
    frame = frame.sort_index().sort_values(
        ["score", "size"], ascending=[False, False], kind="mergesort"
    )
    return list(frame.index)


def assignment_matrix(
    opps: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    w: float,
    threshold: float,
    min_weight: float,
    excluded: frozenset[str] | set[str] | None = None,
) -> dict[str, object]:
    """Industry × province **counts** over all 50 opportunities.

    ``cell(i, p) = #`` of the 50 products for which industry ``i`` is mapped
    *and* province ``p`` **clears ``threshold``** on that product. A row can
    therefore carry many marks and a cell can exceed 1 — both meaningful ("this
    industry is a top pick for three different products here"). The per-product
    assignment is exactly the one Page A shows, replayed for all 50 under the
    same parameters; ``threshold`` is the parity point of the presence measure
    times the user's sensitivity dial (see :func:`parity_threshold`).

    Rows are the industries that clear ``min_weight`` for at least one product;
    the rest are HS4-aggregation residue that would otherwise earn a mark on a
    weight of 0.000055.

    Also returns ``cell_products``: ``(industry, province) -> [hs4, …]``, the
    opportunities *behind* each count. The matrix says how many; this says which,
    and it is the same pass — inverting it later would risk drifting from the
    counts it is supposed to explain.

    And ``product_matrix``: the **primitive** the industry matrix is derived
    from — one row per opportunity, the cell carrying the province's **rank**
    among the provinces that qualified (0 = not assigned). The number of marks
    per row now **varies by product**, which is the point of a threshold rule.
    Its column totals are the
    number of **opportunities** a province wins, which is a different and much
    smaller quantity than the industry matrix's column totals: an opportunity
    mapping to four industries contributes four industry marks but one
    opportunity. Reporting the industry total as "opportunities" would overstate
    it roughly two-fold (Pichincha: 70 industry marks, 29 opportunities).

    ``excluded`` is forwarded to :func:`qualifying_provinces` — the provinces the
    lens never assigns. They keep their **columns** here (all-zero) so the frames
    stay the same width as everything derived from them; the pages drop the
    column at draw time, since an all-zero column reads as "no opportunity reaches
    this province", which is a different statement.
    """
    provinces = list(frames["presence"].columns)
    products = sorted(opps["opportunity_hs4_code"].dropna().unique())
    codes = sorted(
        opps.loc[opps["crosswalk_weight_used"] >= float(min_weight), "isic4_code"].dropna().unique()
    )
    matrix = pd.DataFrame(0, index=codes, columns=provinces, dtype="int64")
    product_matrix = pd.DataFrame(0, index=products, columns=provinces, dtype="int64")

    assigned: dict[str, list[str]] = {}
    industries_by_product: dict[str, list[str]] = {}
    cell_products: dict[tuple[str, str], list[str]] = {}
    for hs4 in products:
        direct = product_presence(opps, frames, hs4, "producto", min_weight)
        anchor = product_presence(opps, frames, hs4, "ancla", min_weight)
        # Intensive-margin opportunities have no anchors: the route weight has
        # nothing to blend and the score is the direct route alone.
        score = (
            blend_routes(direct["presence"], anchor["presence"], w)["score"]
            if anchor["n_anchors"]
            else direct["presence"]
        )
        rows = opps[opps["opportunity_hs4_code"] == hs4]
        mapped = sorted(
            rows.loc[rows["crosswalk_weight_used"] >= float(min_weight), "isic4_code"].dropna().unique()
        )
        size = frames["plazas"].reindex(mapped).sum(axis=0).reindex(provinces).fillna(0.0)
        picks = qualifying_provinces(score, threshold, size, excluded)
        assigned[hs4] = picks
        industries_by_product[hs4] = mapped
        for position, province in enumerate(picks, start=1):
            product_matrix.loc[hs4, province] = position
        for code in mapped:
            if code in matrix.index:
                matrix.loc[code, picks] += 1
                for province in picks:
                    cell_products.setdefault((code, province), []).append(hs4)

    return {
        "matrix": matrix,
        "product_matrix": product_matrix,
        # Opportunities won per province — the honest "how many of the 50" figure.
        "opportunities_per_province": (product_matrix > 0).sum(axis=0),
        "assigned": assigned,
        "industries_by_product": industries_by_product,
        "cell_products": cell_products,
    }


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


# --------------------------------------------------------------------------- #
# The candidate list — the two lenses joined into one committed list.          #
# --------------------------------------------------------------------------- #
#
# The app's two lenses each answer half a question. The top-down assignment is the
# stronger evidentiary base but can only reach industries a province already visibly
# has, so it leaves 2 provinces with nothing and 12 more under 10; the bottom-up
# ranking covers every province by construction but is agnostic to the product
# analysis. The **list** is the top-down assignment **topped up** from the bottom-up
# ranking to a floor of ``min_industries``, with every row labelled ``lens`` for where
# it came from.
#
# The same list is also a committed analysis artifact —
# ``data/processed/summary_tables/province_candidates/``, built by
# ``utils/build_province_candidates.py``. The app derives it here rather than reading
# those files, which is what makes it deployable with no builder run and keeps the
# page consistent with the explorer and the grids *by construction*. The two are tied
# together by an **equality tripwire** in the test suite, not by a loader: drift
# becomes a red test at pin time rather than a wrong page at runtime.
#
# These functions therefore mirror the builder's ``build_top_down`` /
# ``build_bottom_up`` / ``build_final`` step for step, and two of those steps are easy
# to "improve" into a different answer:
#
#   * the bottom-up leg passes **no** ``ineligible_classes`` to
#     :func:`province_shortlist_matrix` and drops thin-neighbourhood classes only
#     *after* the top-``k`` cut, so a thin class still consumes a pool slot;
#   * ``n_estab`` / ``plazas_equiv`` on every finished row come from the **RCA**
#     export, overwriting the composition figures the bottom-up guard actually read
#     (the two genuinely disagree — see CLAUDE.md's note on the two
#     establishment-count conventions).
#
# Both are the builder's rules, and reproducing them is what lets the tripwire pass.

# The composition of the combined ``min_industries`` the fill aims at. Approached,
# never enforced — see :func:`deficit_fill`.
CANDIDATE_TARGET_MIX = {"specialized": 5, "present_not_spec": 3, "absent": 2}

# The finished list's schema, in order — the same column set the committed CSVs ship.
CANDIDATE_LIST_COLUMNS = [
    "attractiveness_metric", "id_code", "id_name", "region", "codigo_clase",
    "display_code", "class_name", "gsector_label", "lens", "state", "score",
    "n_opportunities", "hs4_codes", "accessible_market_busd", "rca", "n_estab",
    "plazas_equiv",
    "current_plazas", "potential_plazas", "gap_plazas", "leader_province",
    "wage_per_fte", "fte_per_firm", "no_growth", "both_bases", "thin_neighborhood",
    "entry_regime", "rank_in_province",
]

_ATTRACTIVENESS_METRICS = ("wage_per_fte", "fte_per_firm")


def top_down_candidates(
    opportunities: pd.DataFrame,
    rca: pd.DataFrame,
    density: pd.DataFrame,
    *,
    view: str,
    base: str,
    apply_support: bool,
    min_estab: int,
    min_plazas: float,
    route_weight: float,
    cut: float,
    min_crosswalk_weight: float,
    excluded: frozenset[str] | set[str] | None = None,
) -> pd.DataFrame:
    """The complete top-down assignment as one row per (province, industry).

    A province's top-down industries are the rows of :func:`assignment_matrix` with a
    positive count in its column — the industries that at least one of the 50
    opportunities put there. ``n_opportunities`` is that count and ``hs4_codes`` names
    the opportunities behind it, both taken from the same pass so the two cannot
    disagree.

    **Metric-independent**: nothing here reads an attractiveness figure, which is why
    both metrics' lists carry an identical top-down block.

    Returns ``id_code``, ``codigo_clase``, ``n_opportunities``, ``hs4_codes``,
    ``accessible_market_busd`` (the opportunities' accessible market summed over
    ``hs4_codes``, billions of current USD, from the opportunities export — a world
    market size, **not** anything Ecuador ships or this province would capture) plus the
    ``state`` and ``rca`` the establishment-base density export gives the cell — the
    same measure that defines a presence state everywhere else in the app, so the two
    lenses' ``state`` columns are comparable.

    ``excluded`` provinces (see :func:`qualifying_provinces`) contribute no rows: they
    are never assigned, so the candidate list reaches them through the bottom-up top-up
    instead. They still count in every denominator the assignment uses.
    """
    frames = presence_frame(rca, view, base, apply_support, min_estab, min_plazas)
    threshold = parity_threshold(view, len(frames["presence"].columns)) * float(cut)
    result = assignment_matrix(
        opportunities, frames, float(route_weight), threshold, min_crosswalk_weight,
        excluded,
    )
    matrix: pd.DataFrame = result["matrix"]
    cell_products: dict = result["cell_products"]

    # Accessible market per opportunity, in billions of current USD, straight off the
    # export — an opportunity-level attribute repeated on all of that opportunity's
    # rows, so one value per HS4 after de-duplication.
    market = (
        opportunities.drop_duplicates("opportunity_hs4_code")
        .set_index("opportunity_hs4_code")["opportunity_accessible_market_busd"]
    )

    rows = []
    for province in matrix.columns:
        column = matrix[province]
        for code in column.index[column > 0]:
            products = cell_products.get((code, province), [])
            # Built in the same pass; if the two ever disagree the matrix is no longer
            # explained by what it ships as evidence.
            if len(products) != int(column[code]):
                raise ValueError(
                    f"top-down: {province}/{code} counts {int(column[code])} "
                    f"opportunities but names {len(products)}"
                )
            rows.append({
                "id_code": str(province),
                "codigo_clase": str(code),
                "n_opportunities": int(column[code]),
                "hs4_codes": ";".join(sorted(products)),
                # Summed over the opportunities named in `hs4_codes`, each of which
                # appears at most once in this cell — so this is a sum over *distinct*
                # opportunities, never one counted twice. It can still overstate the
                # distinct market when two of them address overlapping world demand
                # (HS 9401 seats + HS 9403 other furniture), which is a property of the
                # upstream figure; the page discloses it rather than adjusting it here.
                "accessible_market_busd": float(
                    market.reindex(sorted(products)).sum()
                ),
            })
    frame = pd.DataFrame(
        rows,
        columns=["id_code", "codigo_clase", "n_opportunities", "hs4_codes",
                 "accessible_market_busd"],
    )
    states = density[["id_code", "codigo_clase", "state", "rca"]]
    return frame.merge(states, on=["id_code", "codigo_clase"], how="left")


def deficit_fill(
    have: dict[str, int],
    pool: pd.DataFrame,
    target_mix: dict[str, int] | None = None,
    min_industries: int = 10,
) -> list:
    """Fill a province to ``min_industries``, closing the largest gap to the mix first.

    ``have`` is the province's top-down composition by presence state; ``pool`` is its
    usable bottom-up candidates, **already sorted best-score-first**. At each step the
    state furthest below its target contributes its best remaining candidate; ties on
    the deficit go to the higher presence state (specialized > present > absent).

    A plain round-robin was rejected upstream because its *starting* state decides the
    whole outcome for a province needing one or two slots (spec-first and absent-first
    give 29/23/17 and 17/23/29 from identical inputs).

    The target is **approached, never enforced**: a top-down leg that already
    overshoots a state cannot be un-picked, so a province can land 5/4/1 or 4/2/4. If
    the pool runs out before the floor, the picks made so far are returned rather than
    padded — the caller reports the shortfall.

    Returns the index labels of ``pool`` that were picked, in pick order.
    """
    mix = dict(target_mix or CANDIDATE_TARGET_MIX)
    counts = {state: int(have.get(state, 0)) for state in mix}
    total = sum(counts.values())
    queues = {state: list(pool.index[pool["state"] == state]) for state in mix}
    picked: list = []
    while total < int(min_industries):
        order = sorted(
            mix, key=lambda state: (-(mix[state] - counts[state]), -_STATE_RANK[state])
        )
        for state in order:
            if queues[state]:
                picked.append(queues[state].pop(0))
                counts[state] += 1
                total += 1
                break
        else:  # the pool is exhausted before the floor — report, never pad silently
            return picked
    return picked


def _candidate_identity(
    frame: pd.DataFrame,
    province_meta: pd.DataFrame,
    display_codes: dict[str, str],
    class_labels: dict[str, str],
    gsectors: dict[str, str],
) -> pd.DataFrame:
    """The shared identity block: province metadata and class labels."""
    out = frame.drop(columns=[c for c in ("id_name", "region") if c in frame.columns])
    out = out.merge(province_meta[["id_code", "id_name", "region"]], on="id_code", how="left")
    out["display_code"] = out["codigo_clase"].map(display_codes)
    out["class_name"] = out["codigo_clase"].map(class_labels)
    out["gsector_label"] = out["codigo_clase"].map(gsectors)
    return out


def _candidate_both_bases(frame: pd.DataFrame, density_plazas: pd.DataFrame) -> pd.DataFrame:
    """``both_bases``: does the presence state survive switching to the employment base?

    A disclosure column and nothing else — every state, score and rank on the list stays
    establishment-based (the lens's own base ruling). A cell with no employment-base row
    compares unequal and so reads False, exactly as on the explorer.
    """
    plazas = density_plazas[["id_code", "codigo_clase", "state"]].rename(
        columns={"state": "_state_plazas"}
    )
    out = frame.merge(plazas, on=["id_code", "codigo_clase"], how="left")
    out["both_bases"] = out["state"] == out["_state_plazas"]
    return out.drop(columns=["_state_plazas"])


def _candidate_benchmark(
    frame: pd.DataFrame, rca: pd.DataFrame, min_plazas: float
) -> pd.DataFrame:
    """Price every row against Ecuador's own frontier, province by province.

    ``potential = leader_share × T(province)`` (:func:`leader_benchmark` /
    :func:`potential_gap`) — an existence proof from Ecuadorian data, a stock and never
    a forecast. Classes with no qualifying frontier get no benchmark rather than a
    fallback below the floor the rest of the app calls noise.
    """
    benchmark = leader_benchmark(rca, min_plazas)
    totals = province_employment_totals(rca)
    pieces = []
    for province, group in frame.groupby("id_code", sort=False):
        here_fte = (
            rca.loc[rca["id_code"] == province]
            .drop_duplicates("codigo_clase")
            .set_index("codigo_clase")["plazas_equiv"]
        )
        priced = potential_gap(
            benchmark, group["codigo_clase"].tolist(), province,
            float(totals.get(province, 0.0)), here_fte,
        )
        pieces.append(
            group.merge(
                priced[["current", "potential", "gap", "leader_id_name", "at_frontier"]],
                left_on="codigo_clase", right_index=True, how="left",
            )
        )
    out = pd.concat(pieces, ignore_index=True).rename(columns={
        "current": "current_plazas", "potential": "potential_plazas",
        "gap": "gap_plazas", "leader_id_name": "leader_province",
        "at_frontier": "no_growth",
    })
    # An unbenchmarkable class is not "no growth", it is "no yardstick" — so the flag
    # stays blank there rather than becoming a False that reads as a measurement.
    out["no_growth"] = pd.Series(
        [pd.NA if pd.isna(value) else bool(value) for value in out["no_growth"]],
        index=out.index, dtype="boolean",
    )
    return out


def _rank_in_province(frame: pd.DataFrame) -> pd.DataFrame:
    """1…n within each province: the top-down rows first, then the top-ups.

    Top-down first because it is the stronger evidentiary base, ordered by how many of
    the 50 opportunities put the industry there and then by observed presence; the
    top-ups follow by their own score. One ordering is never taken across both, since
    the two lenses' scores are not comparable — a composite sort would let ``rca``
    (which every row has) decide the top-up order and leave ``score`` (which only the
    top-ups have) as a tie-break.
    """
    top_down = frame[frame["lens"] == "top_down"].sort_values(
        ["id_code", "n_opportunities", "rca", "codigo_clase"],
        ascending=[True, False, False, True], kind="mergesort",
    )
    top_up = frame[frame["lens"] == "bottom_up"].sort_values(
        ["id_code", "score", "codigo_clase"],
        ascending=[True, False, True], kind="mergesort",
    )
    # A stable sort on the province alone, so the concat's block order survives it.
    out = pd.concat([top_down, top_up], ignore_index=True).sort_values(
        "id_code", kind="mergesort"
    )
    out["rank_in_province"] = out.groupby("id_code").cumcount() + 1
    return out


def candidate_list(
    *,
    var: str,
    provinces: list[str],
    opportunities: pd.DataFrame,
    rca: pd.DataFrame,
    density: pd.DataFrame,
    density_plazas: pd.DataFrame,
    profile: pd.DataFrame,
    emp: pd.DataFrame,
    support: pd.DataFrame,
    menu: set[str] | frozenset[str],
    class_labels: dict[str, str],
    entry_regime: dict[str, str],
    min_estab: int,
    min_plazas: float,
    min_crosswalk_weight: float,
    thin_threshold: float,
    td_view: str,
    td_base: str,
    td_support: bool,
    td_route_weight: float,
    td_cut: float,
    beta: float,
    weight: float,
    top_k: int,
    min_industries: int = 10,
    target_mix: dict[str, int] | None = None,
    top_down: pd.DataFrame | None = None,
    td_excluded: frozenset[str] | set[str] | None = None,
) -> pd.DataFrame:
    """Every province's candidate list for one attractiveness metric.

    The top-down assignment (:func:`top_down_candidates`) is the base; any province it
    leaves under ``min_industries`` is topped up from that province's bottom-up ranking
    (:func:`province_shortlist_matrix`) by :func:`deficit_fill`, after dropping
    thin-neighbourhood classes and deduplicating against the top-down set. Every row is
    then priced against Ecuador's own frontier and ranked within its province.

    ``top_down`` may be passed in when the caller already has it: the leg is
    metric-independent, so deriving it once and sharing it across both metrics is free
    and cannot produce a different answer.

    ``provinces`` stays the **full** province list even when ``td_excluded`` is set: the
    exclusion applies only inside the top-down leg, so an excluded province arrives with
    no top-down rows and is filled to ``min_industries`` from its own bottom-up ranking.
    That is the point of the exclusion — it changes which lens answers for that province,
    not whether it gets an answer.

    Returns :data:`CANDIDATE_LIST_COLUMNS` sorted by province then ``rank_in_province``
    — the schema of the committed ``province_candidates_final_<metric>.csv``.
    """
    province_meta = rca[["id_code", "id_name", "region"]].drop_duplicates("id_code")
    display_codes = dict(zip(profile["codigo_clase"], profile["display_code"]))
    gsectors = dict(zip(profile["codigo_clase"], profile["gsector_label"]))
    attractiveness = profile.set_index("codigo_clase")[list(_ATTRACTIVENESS_METRICS)]
    phi_mass = support.set_index("codigo_clase")["phi_mass_noself"]

    def thin_flag(codes: pd.Series) -> pd.Series:
        # A class with no support row is NaN and therefore not flagged — the app's own
        # comparison semantics (see :func:`connected_eligible`).
        return codes.map(phi_mass) < float(thin_threshold)

    def identity(frame: pd.DataFrame) -> pd.DataFrame:
        return _candidate_identity(
            frame, province_meta, display_codes, class_labels, gsectors
        )

    # ---- the top-down leg ------------------------------------------------- #
    if top_down is None:
        top_down = top_down_candidates(
            opportunities, rca, density,
            view=td_view, base=td_base, apply_support=td_support,
            min_estab=min_estab, min_plazas=min_plazas,
            route_weight=td_route_weight, cut=td_cut,
            min_crosswalk_weight=min_crosswalk_weight,
            excluded=td_excluded,
        )
    base_rows = top_down.copy()
    base_rows["lens"] = "top_down"
    # The two lenses' scores are not comparable, so the top-down rows carry none.
    base_rows["score"] = np.nan
    base_rows = identity(base_rows)
    base_rows = _candidate_both_bases(base_rows, density_plazas)
    base_rows["thin_neighborhood"] = thin_flag(base_rows["codigo_clase"])
    top_down_pairs = set(zip(base_rows["id_code"], base_rows["codigo_clase"]))

    # ---- the bottom-up pool ----------------------------------------------- #
    # No ``ineligible_classes`` here, and the thin drop happens *after* the top-k cut:
    # the builder's rule, so a thin class still consumes a pool slot. Routing the flag
    # through ``ineligible_classes`` instead would give a larger — and different — pool.
    result = province_shortlist_matrix(
        profile, density, list(provinces), var,
        beta=float(beta), weight=float(weight), top_k=int(top_k),
        universe=set(menu),
        emp=emp,
        min_estab=min_estab,
        min_plazas=min_plazas,
        apply_support=True,
    )
    pool: pd.DataFrame = result["marks"].copy()
    pool["thin_neighborhood"] = thin_flag(pool["codigo_clase"])
    pool["overlaps_top_down"] = [
        (row.id_code, row.codigo_clase) in top_down_pairs for row in pool.itertuples()
    ]
    pool = identity(pool)
    pool = _candidate_both_bases(pool, density_plazas)

    # ---- the top-up ------------------------------------------------------- #
    fills: list[pd.DataFrame] = []
    for province in provinces:
        here = base_rows[base_rows["id_code"] == province]
        if len(here) >= int(min_industries):
            continue
        usable = pool[
            (pool["id_code"] == province)
            & (~pool["thin_neighborhood"])
            & (~pool["overlaps_top_down"])
        ].sort_values(["score", "codigo_clase"], ascending=[False, True], kind="mergesort")
        chosen = deficit_fill(
            here["state"].value_counts().to_dict(), usable, target_mix, int(min_industries)
        )
        fills.append(pool.loc[chosen])

    top_up = pd.concat(fills) if fills else pool.iloc[0:0]
    top_up = top_up.copy()
    top_up["lens"] = "bottom_up"
    # Top-down-only columns, carried empty so the two lenses share one schema. The
    # accessible market belongs to an *opportunity*; a top-up row was never reached by
    # one, so it has no market to report — blank, not zero.
    top_up["n_opportunities"] = np.nan
    top_up["hs4_codes"] = ""
    top_up["accessible_market_busd"] = np.nan

    shared = [
        "id_code", "id_name", "region", "codigo_clase", "display_code", "class_name",
        "gsector_label", "lens", "state", "score", "n_opportunities", "hs4_codes",
        "accessible_market_busd", "rca", "both_bases", "thin_neighborhood",
    ]
    final = pd.concat([base_rows[shared], top_up[shared]], ignore_index=True)
    final["attractiveness_metric"] = var

    # Activity from **one** export for every row (see the section note above): the RCA
    # export, which is also where the neighbouring current/potential figures come from,
    # and the dense one, so an absent cell is a true zero rather than a missing key.
    activity = rca[
        ["id_code", "codigo_clase", "num_establecimientos", "plazas_equiv"]
    ].rename(columns={"num_establecimientos": "n_estab"})
    final = final.merge(activity, on=["id_code", "codigo_clase"], how="left")
    for column in _ATTRACTIVENESS_METRICS:
        final[column] = final["codigo_clase"].map(attractiveness[column])
    final = _candidate_benchmark(final, rca, min_plazas)
    final["entry_regime"] = final["codigo_clase"].map(entry_regime)
    final = _rank_in_province(final)
    return final[CANDIDATE_LIST_COLUMNS].sort_values(
        ["id_code", "rank_in_province"], kind="mergesort"
    ).reset_index(drop=True)
