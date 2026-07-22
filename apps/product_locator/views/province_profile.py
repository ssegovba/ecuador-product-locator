"""Page 3: profile of a single province — demographics, labor market, economy."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import state
from data_loader import (
    load_ciiu_reference,
    load_estab_data,
    load_isic_descriptions,
    load_province_profile_tables,
    load_reem_data,
)
from metrics import (
    build_hq_flows,
    build_hq_industry_mix,
    build_selection_metrics,
)
from viz import (
    fmt_number,
    fmt_percent,
    make_cagr_by_province_bar,
    make_gva_trend_figure,
    make_hq_industry_mix_bar,
    make_hq_sankey,
    make_industry_share_bar,
    make_province_scatter,
    make_ranked_indicator_bar,
    make_sector_cagr_figure,
)

# Indicators that get a value-based rank (1 = highest value among provinces).
RANKED_KPIS = [
    "gva_real",
    "gva_per_worker",
    "num_empresas",
    "avg_sales_per_firm_M",
    "formal_wage_per_fte",
    "tasa_desempleo",
    "tasa_informalidad",
    "tasa_subempleo",
    "share_nbi_poor",
    "malnutrition",
    "total_pop",
    "share_working_age",
    "avg_schooling",
    "share_rural",
    "net_migration_rate",
    "share_indigenous",
]

# (column, label, is_share) options for the "Position among provinces" bar.
INDICATOR_OPTIONS = [
    ("gva_per_worker", "GVA per worker (USD)", False),
    ("formal_wage_per_fte", "Formal wage per FTE (USD)", False),
    ("avg_schooling", "Average schooling (years)", False),
    ("tasa_desempleo", "Unemployment rate", True),
    ("tasa_informalidad", "Informality rate", True),
    ("tasa_subempleo", "Underemployment rate", True),
    ("share_nbi_poor", "NBI poverty rate", True),
    ("malnutrition", "Child malnutrition", True),
    ("total_pop", "Population (2022)", False),
    ("share_working_age", "Working-age share", True),
    ("share_rural", "Rural population share", True),
    ("share_indigenous", "Indigenous population share", True),
]


def _build_profile_frame(ranked: pd.DataFrame, census: pd.DataFrame) -> pd.DataFrame:
    """Merge census demographics into the multisource table and compute
    uniform value-descending ranks (1 = highest value)."""
    profile = ranked.merge(
        census[["id_code", "total_pop", "share_indigenous"]], on="id_code", how="left"
    )
    for col in RANKED_KPIS:
        values = pd.to_numeric(profile[col], errors="coerce")
        profile[f"vrank_{col}"] = values.rank(method="min", ascending=False)
        profile[f"vrank_n_{col}"] = values.notna().sum()
    return profile


def _rank_caption(row: pd.Series, column: str) -> str:
    rank = row.get(f"vrank_{column}")
    n = row.get(f"vrank_n_{column}")
    if rank is None or pd.isna(rank):
        return ""
    return f"rank {int(rank)}/{int(n)}"


def render() -> None:
    state.init_state()
    state.require_selection()

    tables = load_province_profile_tables()
    profile = _build_profile_frame(tables["multisource_ranked"], tables["census"])

    st.title("Province profile")
    st.caption(
        "Ranks order the provinces by value (1 = highest value, "
        f"{len(profile)} provinces ranked). The arrow next to each rank is decorative — "
        "it does not indicate a change over time."
    )

    reem_df = load_reem_data()
    estab_df = load_estab_data()
    selection_metrics = build_selection_metrics(
        reem_df,
        st.session_state["selected_isic"],
        employment_base=st.session_state["employment_base"],
        estab_df=estab_df,
    )
    province_metrics = selection_metrics["province_metrics"]

    # Default to the province chosen earlier, else the one with the most
    # selected-industry employment.
    province_options = profile.sort_values("id_code")["id_code"].tolist()
    name_lookup = profile.set_index("id_code")
    default_code = st.session_state.get("selected_province")
    if default_code not in province_options and not province_metrics.empty:
        default_code = province_metrics.sort_values("employment", ascending=False)[
            "codigo_provincia"
        ].iloc[0]

    selected_code = st.selectbox(
        "Province",
        province_options,
        index=province_options.index(default_code) if default_code in province_options else 0,
        format_func=lambda code: f"{name_lookup.loc[code, 'id_name']} ({name_lookup.loc[code, 'region']})",
    )
    st.session_state["selected_province"] = selected_code

    row = name_lookup.loc[selected_code]
    st.header(f"{row['id_name']}")
    st.caption(
        f"Region: **{row['region']}** · GVA {int(row['year_gva'])} · "
        f"labor market {int(row['year_enemdu'])} · business registry {int(row['year_reem'])} · "
        f"census {int(row['year_census'])}"
    )

    st.subheader("Economy")
    eco_cols = st.columns(5)
    eco_cols[0].metric(
        "Real GVA",
        f"${fmt_number(row['gva_real'])} M",
        delta=_rank_caption(row, "gva_real"),
        delta_color="off",
    )
    eco_cols[1].metric(
        "GVA per worker",
        f"${fmt_number(row['gva_per_worker'])}",
        delta=_rank_caption(row, "gva_per_worker"),
        delta_color="off",
    )
    eco_cols[2].metric(
        "Enterprises",
        fmt_number(row["num_empresas"]),
        delta=_rank_caption(row, "num_empresas"),
        delta_color="off",
    )
    eco_cols[3].metric(
        "Avg. sales per firm",
        f"${fmt_number(row['avg_sales_per_firm_M'], 2)} M",
        delta=_rank_caption(row, "avg_sales_per_firm_M"),
        delta_color="off",
    )
    eco_cols[4].metric(
        "Formal wage per FTE",
        f"${fmt_number(row['formal_wage_per_fte'])}",
        delta=_rank_caption(row, "formal_wage_per_fte"),
        delta_color="off",
    )

    st.subheader("Labor market & social conditions")
    lab_cols = st.columns(5)
    lab_cols[0].metric(
        "Unemployment",
        fmt_percent(row["tasa_desempleo"]),
        delta=_rank_caption(row, "tasa_desempleo"),
        delta_color="off",
    )
    lab_cols[1].metric(
        "Informality",
        fmt_percent(row["tasa_informalidad"]),
        delta=_rank_caption(row, "tasa_informalidad"),
        delta_color="off",
    )
    lab_cols[2].metric(
        "Underemployment",
        fmt_percent(row["tasa_subempleo"]),
        delta=_rank_caption(row, "tasa_subempleo"),
        delta_color="off",
    )
    lab_cols[3].metric(
        "NBI poverty",
        fmt_percent(row["share_nbi_poor"]),
        delta=_rank_caption(row, "share_nbi_poor"),
        delta_color="off",
    )
    lab_cols[4].metric(
        "Child malnutrition",
        fmt_percent(row["malnutrition"]),
        delta=_rank_caption(row, "malnutrition"),
        delta_color="off",
    )

    st.subheader("Demographics")
    dem_cols = st.columns(6)
    dem_cols[0].metric(
        "Population (2022)",
        fmt_number(row["total_pop"]),
        delta=_rank_caption(row, "total_pop"),
        delta_color="off",
    )
    dem_cols[1].metric(
        "Working-age share",
        fmt_percent(row["share_working_age"]),
        delta=_rank_caption(row, "share_working_age"),
        delta_color="off",
    )
    dem_cols[2].metric(
        "Avg. schooling (years)",
        fmt_number(row["avg_schooling"], 1),
        delta=_rank_caption(row, "avg_schooling"),
        delta_color="off",
    )
    dem_cols[3].metric(
        "Rural population",
        fmt_percent(row["share_rural"]),
        delta=_rank_caption(row, "share_rural"),
        delta_color="off",
    )
    dem_cols[4].metric(
        "Net migration rate",
        fmt_percent(row["net_migration_rate"]),
        delta=_rank_caption(row, "net_migration_rate"),
        delta_color="off",
    )
    dem_cols[5].metric(
        "Indigenous population",
        fmt_percent(row["share_indigenous"]),
        delta=_rank_caption(row, "share_indigenous"),
        delta_color="off",
    )

    st.divider()

    trend_col, cagr_col = st.columns(2)
    with trend_col:
        st.subheader("Real GVA trend")
        st.plotly_chart(
            make_gva_trend_figure(tables["gva_provincial"], selected_code, row["id_name"]),
            width="stretch",
            key=f"gva-trend-{selected_code}",
        )
        st.caption(
            "Selected province in dark red; all other provinces in light grey. "
            "USD millions, constant prices."
        )
    with cagr_col:
        cagr_fig, cagr_year = make_cagr_by_province_bar(tables["gva_provincial"], selected_code)
        st.subheader(f"Real GVA CAGR (10-year period ending {cagr_year})")
        st.plotly_chart(cagr_fig, width="stretch", key=f"cagr-prov-{selected_code}")
        st.caption(
            "Annualized 10-year growth of real GVA. Selected province in dark red; "
            "Ecuador (national) in dark blue."
        )

    structure_col, sector_cagr_col = st.columns(2)
    with structure_col:
        structure_fig, structure_year = make_industry_share_bar(
            tables["gva_province_industry"], selected_code
        )
        st.subheader(f"Economic structure ({structure_year})")
        st.plotly_chart(
            structure_fig,
            width="stretch",
            key=f"structure-{selected_code}",
        )
        st.caption(
            "Share of provincial GVA by industry. Green = primary, "
            "orange = secondary, blue = tertiary sector."
        )
    with sector_cagr_col:
        sector_fig, sec_start, sec_end = make_sector_cagr_figure(
            tables["gva_province_industry"], selected_code
        )
        st.subheader(f"GVA CAGR by sector ({sec_start}–{sec_end})")
        st.plotly_chart(sector_fig, width="stretch", key=f"sector-cagr-{selected_code}")
        st.caption(
            "Bars: annualized GVA growth of each industry in the selected province "
            "(colored by macro-sector). Diamonds: the same industry's growth for "
            "Ecuador as a whole — bars beyond the diamond grow faster than the country."
        )

    st.subheader("Position among provinces")
    scatter_col, bar_col = st.columns(2)
    with scatter_col:
        st.plotly_chart(
            make_province_scatter(
                profile,
                x_column="tasa_informalidad",
                y_column="gva_per_worker",
                selected_id_code=selected_code,
                x_label="Informality rate",
                y_label="GVA per worker (USD)",
                x_is_share=True,
            ),
            width="stretch",
            key=f"scatter-{selected_code}",
        )
        st.caption(
            "Productivity vs labor informality across provinces. The most "
            "attractive locations combine high GVA per worker with lower informality."
        )
    with bar_col:
        indicator = st.selectbox(
            "Indicator",
            options=INDICATOR_OPTIONS,
            format_func=lambda option: option[1],
        )
        st.plotly_chart(
            make_ranked_indicator_bar(
                profile,
                column=indicator[0],
                selected_id_code=selected_code,
                label=indicator[1],
                is_share=indicator[2],
            ),
            width="stretch",
            key=f"ranked-{selected_code}-{indicator[0]}",
        )
        st.caption("Dashed line marks the average across provinces.")

    st.divider()
    st.subheader("Your selected industries in this province")
    hs_code = st.session_state["hs_code"]
    st.caption(
        f"Activity of the industries matched to HS {hs_code} within "
        f"{row['id_name']}, from the 2024 business registry (REEM)."
    )
    province_row = province_metrics[province_metrics["codigo_provincia"] == selected_code]
    if province_row.empty:
        st.info("The selected industries have no registered enterprises in this province.")
    else:
        province_row = province_row.iloc[0]
        # Row A — absolute values with share of the national selection.
        sel_cols = st.columns(4)
        sel_cols[0].metric(
            "Enterprises",
            fmt_number(province_row["enterprises"]),
            delta=f"{fmt_percent(province_row['enterprises_share_country'])} of national selection",
            delta_color="off",
        )
        sel_cols[1].metric(
            "Establishments",
            fmt_number(province_row["establishments"]),
            delta=f"{fmt_percent(province_row['establishments_share_country'])} of national selection",
            delta_color="off",
            help="Active local units (headquarters and branches) located in this province.",
        )
        sel_cols[2].metric(
            "Employment",
            fmt_number(province_row["employment"], 1),
            delta=f"{fmt_percent(province_row['employment_share_country'])} of national selection",
            delta_color="off",
        )
        sel_cols[3].metric(
            "Exporting enterprises (net)",
            fmt_number(province_row["exporting_enterprises"]),
            delta=f"{fmt_percent(province_row['exporters_share_country'])} of national selection",
            delta_color="off",
        )
        # Row B — share of the province's total (all-industry) REEM activity.
        st.caption(
            "Share of the province's total REEM activity (all industries) accounted for "
            "by the selected industries:"
        )
        prov_cols = st.columns(4)
        prov_cols[0].metric(
            "% of province enterprises",
            fmt_percent(province_row["enterprises_share_province"]),
        )
        prov_cols[1].metric(
            "% of province establishments",
            fmt_percent(province_row["establishments_share_province"]),
        )
        prov_cols[2].metric(
            "% of province employment",
            fmt_percent(province_row["employment_share_province"]),
        )
        prov_cols[3].metric(
            "% of province exporters",
            fmt_percent(province_row["exporters_share_province"]),
        )

    st.divider()
    st.subheader("Headquarters reach")

    hq_controls = st.columns([2, 2, 3])
    with hq_controls[0]:
        scope_choice = st.segmented_control(
            "Industry scope",
            ["Selected industries", "All industries"],
            default="Selected industries",
            key=f"hq-scope-{selected_code}",
        )
    with hq_controls[1]:
        flow_choice = st.segmented_control(
            "Flow width",
            ["Employment", "Establishments"],
            default="Employment",
            key=f"hq-flow-{selected_code}",
        )
    scope_codes = st.session_state["selected_isic"] if scope_choice == "Selected industries" else None
    value_column = "employment" if flow_choice == "Employment" else "establishments"

    hq = build_hq_flows(reem_df, estab_df, scope_codes, selected_code)
    flows, hq_stats = hq["flows"], hq["stats"]

    n_firms = hq_stats["n_firms"]
    n_multi = hq_stats["n_multi"]
    if n_firms == 0:
        st.info(
            f"No enterprises in scope are headquartered in {row['id_name']}. "
            "Switch to 'All industries' to see the structural pattern."
        )
    elif n_multi == 0 or flows.empty:
        st.info(
            f"The {fmt_number(n_firms)} enterprises headquartered in {row['id_name']} in "
            "this scope are all single-establishment firms — none control subsidiaries."
        )
    else:
        st.caption(
            f"How many of the {fmt_number(n_firms)} enterprises headquartered in "
            f"{row['id_name']} have subsidiaries, and where do those operate? Flows trace "
            f"the establishments (and employment) controlled by the {fmt_number(n_multi)} "
            "multi-unit firms among them."
        )

        top_dest = hq_stats["top_destinations"]
        top_row = None if top_dest.empty else top_dest.iloc[0]

        # Row 1 — firms (multi-unit; within-only + reaching-other == HQs with branches).
        firm_kpis = st.columns(3)
        firm_kpis[0].metric(
            "HQs with branches (multi-unit firms)",
            fmt_number(n_multi),
            delta=f"{fmt_percent(hq_stats['share_multi_of_firms'])} of province enterprises",
            delta_color="off",
            help="Enterprises headquartered here with more than one establishment (count of firms).",
        )
        firm_kpis[1].metric(
            "Expanding only within province",
            fmt_number(hq_stats["n_multi_within_only"]),
            help="Of those, firms whose establishments are all within this province (count of firms).",
        )
        firm_kpis[2].metric(
            "Expanding to other provinces",
            fmt_number(hq_stats["n_multi_outside"]),
            help="Of those, firms with at least one establishment in another province (count of firms).",
        )

        # Row 2 — establishments controlled by multi-unit firms.
        est_kpis = st.columns(4)
        est_kpis[0].metric(
            "Establishments controlled",
            fmt_number(hq_stats["total_establishments"]),
            help="All establishments (matriz + branches) of the multi-unit firms above.",
        )
        est_kpis[1].metric("Located within province", fmt_percent(hq_stats["share_inside_estab"]))
        est_kpis[2].metric("Located outside province", fmt_percent(hq_stats["share_outside_estab"]))
        if top_row is None:
            est_kpis[3].metric("Top outside destination", "None")
        else:
            est_kpis[3].metric(
                "Top outside destination",
                str(top_row["provincia"]).title(),
                delta=f"{fmt_number(top_row['establishments'])} establishments",
                delta_color="off",
            )

        # Row 3 — employment controlled by multi-unit firms.
        emp_kpis = st.columns(4)
        emp_kpis[0].metric(
            "Employment controlled",
            fmt_number(hq_stats["total_employment"]),
            help="Registered positions (plazas) in the establishments of the multi-unit firms above.",
        )
        emp_kpis[1].metric("Located within province", fmt_percent(hq_stats["share_inside"]))
        emp_kpis[2].metric("Located outside province", fmt_percent(hq_stats["share_outside"]))
        if top_row is None:
            emp_kpis[3].metric("Top outside destination", "None")
        else:
            emp_kpis[3].metric(
                "Top outside destination",
                str(top_row["provincia"]).title(),
                delta=f"{fmt_number(top_row['employment'])} positions",
                delta_color="off",
            )

        source_label = f"{row['id_name']} — {fmt_number(n_multi)} multi-unit HQs"
        st.plotly_chart(
            make_hq_sankey(
                flows, selected_code, row["id_name"], value_column, source_label=source_label
            ),
            width="stretch",
            key=f"hq-sankey-{selected_code}-{value_column}-{scope_choice}",
        )
        st.caption(
            f"Flow width = {flow_choice.lower()} of the multi-unit firms' establishments. "
            "The source node's size reflects the summed flows; its label shows the number "
            "of controlling HQs. Employment is registered full-time-equivalent positions "
            "(plazas), the measure available at establishment level."
        )

        # Industry mix — only under the selected-industry scope (skipped for "All industries").
        if scope_choice == "Selected industries":
            st.subheader("What else do these firms do?")
            mix_result = build_hq_industry_mix(reem_df, estab_df, scope_codes, selected_code)
            mix, mix_stats = mix_result["mix"], mix_result["stats"]
            if mix.empty:
                st.info(
                    "All establishments controlled from this province operate within the "
                    "selected industries — no diversification footprint to show."
                )
            else:
                st.caption(
                    f"{fmt_number(mix_stats['n_nonfocus'])} of "
                    f"{fmt_number(mix_stats['n_controlled'])} controlled establishments "
                    f"({fmt_percent(mix_stats['share_nonfocus'])}) operate **outside** the "
                    f"selected industries — held by "
                    f"{fmt_number(mix_stats['n_firms_diversified'])} firms and accounting "
                    f"for {fmt_percent(mix_stats['share_employment_nonfocus'])} of the "
                    "employment these headquarters control."
                )
                ciiu_ref = load_ciiu_reference()
                isic_desc = load_isic_descriptions()
                class_labels = dict(zip(isic_desc["isic_code"], isic_desc["isic_name"]))
                class_labels.update(ciiu_ref["class_name_lookup"])  # prefer Spanish names
                section_lookup = dict(
                    zip(reem_df["codigo_clase"], reem_df["codigo_seccion"])
                )
                st.plotly_chart(
                    make_hq_industry_mix_bar(
                        mix, class_labels, value_basis=value_column, section_lookup=section_lookup
                    ),
                    width="stretch",
                    key=f"hq-mix-{selected_code}-{value_column}-{scope_choice}",
                )
                st.caption(
                    f"Share of each industry in the group's total ({flow_choice.lower()}): "
                    f"establishments located within {row['id_name']} "
                    f"(n = {fmt_number(mix_stats['n_within'])}) vs in all other provinces "
                    f"(n = {fmt_number(mix_stats['n_outside'])}). Wholesale and retail "
                    "branches are common ancillary activities across all industries; "
                    "production-side classes are stronger signals of value-chain integration. "
                    "Employment = registered positions (plazas)."
                )

    st.divider()
    next_cols = st.columns([3, 1])
    next_cols[0].caption(
        "Which related industries could this province move into? Continue to the "
        "relatedness & opportunity analysis."
    )
    if next_cols[1].button("Related industries →", type="primary", width="stretch"):
        import nav  # deferred: nav imports this module

        st.switch_page(nav.PAGE_RELATEDNESS)
