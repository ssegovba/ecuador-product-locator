"""Page 2: map-centric geographic analysis of the selected industries."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import state
from config import EMPLOYMENT_BASE_LABELS, METRIC_OPTIONS
from data_loader import (
    load_estab_data,
    load_province_geodata,
    load_province_geojson,
    load_reem_data,
)
from metrics import build_selection_metrics
from viz import fmt_number, fmt_percent, make_province_tile_choropleth

HOVER_LABELS = {
    "provincia": "Province",
    "id_abbr": "Code",
    "enterprises_display": "Enterprises",
    "establishments_display": "Establishments",
    "exporters_display": "Exporting enterprises (net)",
    "employment_display": "Employment",
    "enterprises_share_display": "Enterprise share (Ecuador)",
    "establishments_share_display": "Establishment share (Ecuador)",
    "exporters_share_display": "Exporter share (Ecuador)",
    "employment_share_display": "Employment share (Ecuador)",
    "jobs_per_enterprise_display": "Jobs per enterprise",
}


def render() -> None:
    state.init_state()
    state.require_selection()

    st.title("Geographic analysis")
    hs_code = st.session_state["hs_code"]
    hs_description = st.session_state.get("hs_description") or ""
    st.caption(
        f"Product: **HS {hs_code}** ({st.session_state['hs_edition']} edition) — {hs_description}"
    )

    matched = pd.DataFrame(st.session_state["matched_isic"])
    selected_codes = st.session_state["selected_isic"]
    label_lookup = dict(zip(matched["codigo_clase"], matched["isic_name"]))

    control_cols = st.columns([3, 2, 1])
    with control_cols[0]:
        filtered_codes = st.multiselect(
            "Industries included",
            options=selected_codes,
            default=selected_codes,
            format_func=lambda code: f"{code} — {label_lookup.get(code, code)}",
            help="Subset the industries selected on the previous page without going back.",
        )
    with control_cols[1]:
        metric_column = st.selectbox(
            "Map metric",
            options=list(METRIC_OPTIONS.keys()),
            format_func=lambda col: METRIC_OPTIONS[col]["label"],
        )
    with control_cols[2]:
        with st.popover("Options"):
            employment_base = st.selectbox(
                "Employment base",
                options=list(EMPLOYMENT_BASE_LABELS.keys()),
                index=list(EMPLOYMENT_BASE_LABELS.keys()).index(st.session_state["employment_base"]),
                format_func=EMPLOYMENT_BASE_LABELS.get,
            )
            st.session_state["employment_base"] = employment_base

    if not filtered_codes:
        st.warning("Include at least one industry to display the analysis.")
        st.stop()

    reem_df = load_reem_data()
    estab_df = load_estab_data()
    tables = build_selection_metrics(
        reem_df, filtered_codes, employment_base=employment_base, estab_df=estab_df
    )
    province_metrics = tables["province_metrics"]
    summary = tables["summary"]

    if province_metrics.empty:
        st.warning(
            "The included industries have no registered enterprises in the 2024 business "
            "registry (REEM). Go back and select industries with registered activity."
        )
        st.stop()

    kpi_cols = st.columns(6)
    kpi_cols[0].metric(
        "Industries (ISIC classes)",
        f"{summary['isic_classes_matched']} / {len(selected_codes)}",
        help="Industries with REEM activity / industries included in the filter.",
    )
    kpi_cols[1].metric("Provinces with activity", f"{summary['provinces_with_activity']} / 24")
    kpi_cols[2].metric(
        "Enterprises",
        fmt_number(summary["enterprises"]),
        delta=f"{fmt_percent(summary['enterprises_share_national'])} of Ecuador",
        delta_color="off",
        help="Each enterprise is counted once, at its headquarters province.",
    )
    kpi_cols[3].metric(
        "Establishments",
        fmt_number(summary["establishments"]),
        delta=f"{fmt_percent(summary['establishments_share_national'])} of Ecuador",
        delta_color="off",
        help=(
            "Active local units (headquarters and branches). Unlike enterprises, "
            "establishments are counted where activity physically happens, which "
            "corrects the headquarter effect."
        ),
    )
    kpi_cols[4].metric(
        "Employment",
        fmt_number(summary["employment"], 0),
        delta=f"{fmt_percent(summary['employment_share_national'])} of Ecuador",
        delta_color="off",
    )
    kpi_cols[5].metric(
        "Exporting enterprises (net)",
        fmt_number(summary["exporting_enterprises"]),
        delta=f"{fmt_percent(summary['exporters_share_national'])} of Ecuador",
        delta_color="off",
        help="Enterprises with positive net exports (exports minus imports) in 2024.",
    )

    province_gdf = load_province_geodata()
    province_lookup = (
        province_gdf[["id_code", "id_name", "id_abbr"]]
        .drop_duplicates()
        .rename(columns={"id_code": "codigo_provincia"})
    )
    map_base = province_lookup.merge(province_metrics, on="codigo_provincia", how="left")
    map_base["provincia"] = map_base["provincia"].fillna(map_base["id_name"])

    map_base["enterprises_display"] = map_base["enterprises"].apply(fmt_number)
    map_base["establishments_display"] = map_base["establishments"].apply(fmt_number)
    map_base["establishments_share_display"] = map_base["establishments_share_country"].apply(fmt_percent)
    map_base["exporters_display"] = map_base["exporting_enterprises"].apply(fmt_number)
    map_base["employment_display"] = map_base["employment"].apply(lambda v: fmt_number(v, 1))
    map_base["enterprises_share_display"] = map_base["enterprises_share_country"].apply(fmt_percent)
    map_base["exporters_share_display"] = map_base["exporters_share_country"].apply(fmt_percent)
    map_base["employment_share_display"] = map_base["employment_share_country"].apply(fmt_percent)
    map_base["jobs_per_enterprise_display"] = map_base["jobs_per_enterprise"].apply(
        lambda v: fmt_number(v, 2)
    )

    metric_config = METRIC_OPTIONS[metric_column]
    fig = make_province_tile_choropleth(
        province_geojson=load_province_geojson(),
        plot_df=map_base,
        value_column=metric_column,
        colorbar_title=metric_config["label"],
        color_scale=metric_config["colorscale"],
        hover_columns=list(HOVER_LABELS.keys()),
        hover_labels=HOVER_LABELS,
    )
    st.plotly_chart(fig, width="stretch", key=f"map-{metric_column}")
    st.caption(
        "Industry presence indicates productive capability, not production feasibility — "
        "use it as a starting point for where a product could plausibly be made."
    )

    st.subheader("Province detail")
    detail = province_metrics[
        [
            "codigo_provincia",
            "provincia",
            "enterprises",
            "enterprises_share_country",
            "establishments",
            "establishments_share_country",
            "exporting_enterprises",
            "exporters_share_country",
            "employment",
            "employment_share_country",
            "jobs_per_enterprise",
        ]
    ].sort_values("employment", ascending=False)
    st.dataframe(
        detail,
        width="stretch",
        hide_index=True,
        column_config={
            "codigo_provincia": st.column_config.TextColumn("Code"),
            "provincia": st.column_config.TextColumn("Province"),
            "enterprises": st.column_config.NumberColumn("Enterprises", format="localized"),
            "enterprises_share_country": st.column_config.NumberColumn(
                "Enterprise share", format="percent"
            ),
            "establishments": st.column_config.NumberColumn("Establishments", format="localized"),
            "establishments_share_country": st.column_config.NumberColumn(
                "Establishment share", format="percent"
            ),
            "exporting_enterprises": st.column_config.NumberColumn(
                "Exporters (net)", format="localized"
            ),
            "exporters_share_country": st.column_config.NumberColumn(
                "Exporter share", format="percent"
            ),
            "employment": st.column_config.NumberColumn("Employment", format="localized"),
            "employment_share_country": st.column_config.NumberColumn(
                "Employment share", format="percent"
            ),
            "jobs_per_enterprise": st.column_config.NumberColumn(
                "Jobs per enterprise", format="%.2f"
            ),
        },
    )
    st.download_button(
        "Download province metrics CSV",
        data=detail.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"hs{hs_code}_province_metrics.csv",
        mime="text/csv",
    )

    st.divider()
    next_cols = st.columns([3, 1])
    next_cols[0].caption(
        "Found an interesting province? Continue to its full profile — demographics, "
        "labor market and economic structure."
    )
    if next_cols[1].button("Province profile →", type="primary", width="stretch"):
        import nav  # deferred: nav imports this module

        st.switch_page(nav.PAGE_PROVINCE)
