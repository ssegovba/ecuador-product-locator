from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import (
    COLOR_FOCUS,
    COLOR_NATIONAL,
    COLOR_OTHER_BAR,
    COLOR_OTHER_LINE,
    MAP_CENTER,
    MAP_HEIGHT,
    MAP_STYLE,
    MAP_ZOOM,
    SECTOR_COLORS,
)


# Subtle dark province boundaries: the carto basemap and the low end of the
# colorscales are both near-white, so white borders disappear.
BOUNDARY_COLOR = "#555555"


def fmt_number(value: float | int | None, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "No data"
    return f"{value:,.{decimals}f}"


def fmt_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "No data"
    return f"{value:.1%}"


def make_province_tile_choropleth(
    province_geojson: dict,
    plot_df: pd.DataFrame,
    value_column: str,
    colorbar_title: str,
    color_scale: str,
    hover_columns: list[str],
    hover_labels: dict[str, str],
    locations_column: str = "codigo_provincia",
    missing_color: str = "#d9d9d9",
    outline_ids: list[str] | None = None,
    outline_color: str = COLOR_FOCUS,
    outline_width: float = 3.5,
) -> go.Figure:
    """Choropleth on a MapLibre tile basemap (plotly >= 5.24 'map' traces).

    ``outline_ids`` draws an overlay of those ``id_code`` provinces with a
    transparent fill and a colored border (e.g. to flag M=1 "already present"
    provinces on a density map); its hover is skipped so hover falls through to
    the value trace beneath.
    """
    plot_df = plot_df.copy()
    plot_df[value_column] = pd.to_numeric(plot_df[value_column], errors="coerce")

    hover_lines = []
    for index, column in enumerate(hover_columns):
        label = hover_labels.get(column, column.replace("_", " ").title())
        hover_lines.append(f"{label}: %{{customdata[{index}]}}")
    hover_lines.append("<extra></extra>")

    fig = go.Figure()

    missing = plot_df[plot_df[value_column].isna()]
    if not missing.empty:
        fig.add_trace(
            go.Choroplethmap(
                geojson=province_geojson,
                locations=missing[locations_column],
                z=[0] * len(missing),
                featureidkey="properties.id_code",
                colorscale=[[0, missing_color], [1, missing_color]],
                showscale=False,
                marker_opacity=0.55,
                marker_line_color=BOUNDARY_COLOR,
                marker_line_width=0.7,
                customdata=missing[hover_columns],
                hovertemplate="<br>".join(hover_lines),
                name="",
            )
        )

    present = plot_df[plot_df[value_column].notna()]
    fig.add_trace(
        go.Choroplethmap(
            geojson=province_geojson,
            locations=present[locations_column],
            z=present[value_column],
            featureidkey="properties.id_code",
            colorscale=color_scale,
            colorbar=dict(title=colorbar_title, thickness=14, x=1.0),
            marker_opacity=0.8,
            marker_line_color=BOUNDARY_COLOR,
            marker_line_width=0.7,
            customdata=present[hover_columns],
            hovertemplate="<br>".join(hover_lines),
            name="",
        )
    )

    if outline_ids:
        outline = [str(code) for code in outline_ids]
        fig.add_trace(
            go.Choroplethmap(
                geojson=province_geojson,
                locations=outline,
                z=[0] * len(outline),
                featureidkey="properties.id_code",
                colorscale=[[0, outline_color], [1, outline_color]],
                showscale=False,
                marker_opacity=0.0,  # transparent fill; only the border shows
                marker_line_color=outline_color,
                marker_line_width=outline_width,
                hoverinfo="skip",
                name="",
            )
        )

    fig.update_layout(
        map=dict(style=MAP_STYLE, center=MAP_CENTER, zoom=MAP_ZOOM),
        margin=dict(l=0, r=0, t=0, b=0),
        height=MAP_HEIGHT,
    )
    return fig


def make_gva_trend_figure(
    gva_provincial: pd.DataFrame,
    selected_id_code: str,
    selected_name: str,
) -> go.Figure:
    """Real GVA trend: selected province highlighted vs all other provinces."""
    df = gva_provincial.copy()
    df = df[df["id_code"] != "99"]  # drop the national aggregate; it dwarfs the provinces
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["gva_real"] = pd.to_numeric(df["gva_real"], errors="coerce")
    df = df.dropna(subset=["year", "gva_real"])

    fig = go.Figure()
    others = df[df["id_code"] != selected_id_code]
    for _, group in others.groupby("id_code"):
        fig.add_trace(
            go.Scatter(
                x=group["year"],
                y=group["gva_real"],
                mode="lines",
                line=dict(color=COLOR_OTHER_LINE, width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    selected = df[df["id_code"] == selected_id_code].sort_values("year")
    fig.add_trace(
        go.Scatter(
            x=selected["year"],
            y=selected["gva_real"],
            mode="lines+markers",
            line=dict(color=COLOR_FOCUS, width=3),
            marker=dict(size=5),
            name=selected_name,
            hovertemplate="%{x}: %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="Real GVA (USD millions)",
        xaxis_title=None,
        showlegend=False,
    )
    return fig


def make_industry_share_bar(
    gva_industry: pd.DataFrame,
    selected_id_code: str,
) -> tuple[go.Figure, int]:
    """Horizontal bar of GVA shares by CIIU industry for one province
    (latest year, all sectors). Returns (figure, latest_year)."""
    df = gva_industry[gva_industry["id_code"] == selected_id_code].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    latest_year = int(df["year"].max())
    df = df[df["year"] == latest_year]
    # ciiu_share is stored as percent (0-100) in the CSV; convert to fraction
    # so the % tickformat does not double-scale it.
    df["ciiu_share"] = pd.to_numeric(df["ciiu_share"], errors="coerce") / 100.0
    df = df.dropna(subset=["ciiu_share"]).sort_values("ciiu_share", ascending=True)
    df["label"] = df["ciiu_name"] + " (" + df["ciiu_code"] + ")"

    colors = [SECTOR_COLORS.get(str(sector), COLOR_OTHER_BAR) for sector in df.get("sector", "")]
    fig = go.Figure(
        go.Bar(
            x=df["ciiu_share"],
            y=df["label"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(360, 24 * len(df) + 80),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_tickformat=".0%",
        xaxis_title="Share of provincial GVA",
        yaxis_title=None,
    )
    return fig, latest_year


def make_cagr_by_province_bar(
    gva_provincial: pd.DataFrame,
    selected_id_code: str,
) -> tuple[go.Figure, int]:
    """10-year real-GVA CAGR by province (latest year), selected highlighted.

    The `cagr` column is already in percent. Returns (figure, latest_year).
    """
    df = gva_provincial.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["cagr"] = pd.to_numeric(df["cagr"], errors="coerce")
    df = df.dropna(subset=["year", "cagr"])
    latest_year = int(df["year"].max())
    df = df[df["year"] == latest_year].sort_values("cagr")

    colors = []
    for code in df["id_code"]:
        if code == selected_id_code:
            colors.append(COLOR_FOCUS)
        elif code == "99":
            colors.append(COLOR_NATIONAL)
        else:
            colors.append(COLOR_OTHER_BAR)

    fig = go.Figure(
        go.Bar(
            x=df["cagr"],
            y=df["id_abbr"],
            orientation="h",
            marker_color=colors,
            customdata=df[["id_name"]],
            hovertemplate="%{customdata[0]}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#333333")
    fig.update_layout(
        height=max(420, 18 * len(df) + 60),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="CAGR (%, 10-year)",
        yaxis_title=None,
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig, latest_year


def make_sector_cagr_figure(
    gva_industry: pd.DataFrame,
    selected_id_code: str,
) -> tuple[go.Figure, int, int]:
    """Sector GVA CAGR: bars = selected province, diamonds = national.

    Computed from `gva_province_industry.csv` between its first and last
    available year per sector. Returns (figure, start_year, end_year).
    """
    df = gva_industry.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["gva_real"] = pd.to_numeric(df["gva_real"], errors="coerce")
    start_year = int(df["year"].min())
    end_year = int(df["year"].max())
    n_years = end_year - start_year

    def sector_cagr(frame: pd.DataFrame) -> pd.DataFrame:
        start = frame[frame["year"] == start_year].set_index("ciiu_code")["gva_real"]
        end = frame[frame["year"] == end_year].set_index("ciiu_code")["gva_real"]
        names = frame.drop_duplicates("ciiu_code").set_index("ciiu_code")[["ciiu_name", "sector"]]
        out = names.copy()
        out["cagr"] = np.where(
            start.reindex(out.index) > 0,
            ((end.reindex(out.index) / start.reindex(out.index)) ** (1 / n_years) - 1) * 100,
            np.nan,
        )
        return out.reset_index()

    province = sector_cagr(df[df["id_code"] == selected_id_code])
    national = sector_cagr(df[df["id_code"] == "99"]).rename(columns={"cagr": "cagr_national"})
    merged = province.merge(
        national[["ciiu_code", "cagr_national"]], on="ciiu_code", how="left"
    )
    merged["label"] = merged["ciiu_name"] + " (" + merged["ciiu_code"] + ")"
    merged = merged.dropna(subset=["cagr"]).sort_values("cagr")

    colors = [SECTOR_COLORS.get(str(sector), COLOR_OTHER_BAR) for sector in merged["sector"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=merged["cagr"],
            y=merged["label"],
            orientation="h",
            marker_color=colors,
            name="Province",
            customdata=merged[["cagr_national"]],
            hovertemplate=(
                "%{y}<br>Province: %{x:.2f}%<br>"
                "Ecuador: %{customdata[0]:.2f}%<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=merged["cagr_national"],
            y=merged["label"],
            mode="markers",
            marker=dict(symbol="diamond", size=9, color="#333333"),
            name="Ecuador",
            hovertemplate="%{y}<br>Ecuador: %{x:.2f}%<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#333333")
    fig.update_layout(
        height=max(420, 24 * len(merged) + 80),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title=f"CAGR (%, {n_years}-year)",
        yaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        showlegend=True,
    )
    return fig, start_year, end_year


def make_hq_sankey(
    flows: pd.DataFrame,
    hq_code: str,
    hq_name: str,
    value_column: str,
    top_n: int = 10,
    source_label: str | None = None,
) -> go.Figure:
    """Sankey from the HQ province to the provinces hosting its enterprises'
    establishments. ``value_column`` is 'employment' or 'establishments'.
    ``source_label`` overrides the left node's label (e.g. to carry the count
    of multi-unit HQs, whose node size necessarily reflects the summed flows)."""
    df = flows[flows[value_column] > 0].sort_values(value_column, ascending=False).copy()
    top = df.head(top_n)
    rest = df.iloc[top_n:]

    labels = [source_label or f"{hq_name} (HQ)"]
    values: list[float] = []
    link_colors: list[str] = []
    node_colors = [COLOR_FOCUS]
    for row in top.itertuples():
        is_inside = row.codigo_provincia == hq_code
        labels.append(f"{row.provincia} (within province)" if is_inside else row.provincia)
        node_colors.append("#B07070" if is_inside else COLOR_OTHER_BAR)
        values.append(getattr(row, value_column))
        link_colors.append("rgba(139,0,0,0.35)" if is_inside else "rgba(127,127,127,0.30)")
    if not rest.empty:
        labels.append(f"Other provinces ({len(rest)})")
        node_colors.append("#BFBFBF")
        values.append(float(rest[value_column].sum()))
        link_colors.append("rgba(127,127,127,0.20)")

    total = sum(values) or float("nan")
    customdata = [[value / total] for value in values]
    fig = go.Figure(
        go.Sankey(
            node=dict(
                label=labels,
                color=node_colors,
                pad=12,
                thickness=14,
                line=dict(color="white", width=0.5),
            ),
            link=dict(
                source=[0] * len(values),
                target=list(range(1, len(values) + 1)),
                value=values,
                color=link_colors,
                customdata=customdata,
                hovertemplate=(
                    "%{target.label}<br>"
                    + ("%{value:,.0f}" if value_column == "establishments" else "%{value:,.1f}")
                    + " · %{customdata[0]:.1%} of total<extra></extra>"
                ),
            ),
        )
    )
    fig.update_layout(
        height=max(460, 38 * (len(values) + 1)),
        margin=dict(l=0, r=0, t=20, b=10),
        font=dict(size=13),
    )
    return fig


def make_hq_industry_mix_bar(
    mix: pd.DataFrame,
    class_label_lookup: dict[str, str],
    value_basis: str,
    top_n: int = 15,
    section_lookup: dict[str, str] | None = None,
) -> go.Figure:
    """Grouped horizontal bars: industry shares of the HQ firms' (non-focus)
    establishments, within the HQ province vs in all other provinces, on a
    shared industry ordering. ``value_basis`` is 'employment' or
    'establishments'. ``section_lookup`` (4-digit class -> section letter)
    prefixes the code, e.g. 'A0311'."""
    section_lookup = section_lookup or {}

    def code_label(code: str) -> str:
        display_code = f"{section_lookup.get(code, '')}{code}"
        return f"{class_label_lookup.get(code, code)} ({display_code})"

    share_column = f"share_{value_basis}"
    wide = mix.pivot_table(
        index="codigo_clase",
        columns="location",
        values=[value_basis, share_column],
        aggfunc="sum",
    )
    for location in ("within", "outside"):
        if (value_basis, location) not in wide.columns:
            wide[(value_basis, location)] = 0.0
            wide[(share_column, location)] = 0.0
    wide = wide.fillna(0.0)

    # Top classes by combined size; remainder aggregated into "Other".
    combined = wide[value_basis].sum(axis=1)
    top_codes = combined.sort_values(ascending=False).head(top_n).index
    top = wide.loc[top_codes].copy()
    rest = wide.drop(index=top_codes)

    rows = []
    for code in top.index:
        rows.append(
            {
                "label": code_label(code),
                "share_within": top.loc[code, (share_column, "within")],
                "share_outside": top.loc[code, (share_column, "outside")],
                "value_within": top.loc[code, (value_basis, "within")],
                "value_outside": top.loc[code, (value_basis, "outside")],
            }
        )
    # Sort named industries by within-share (ascending = bottom-to-top in the
    # chart); pin the aggregated "Other" row to the very bottom.
    plot_df = pd.DataFrame(rows).sort_values("share_within", ascending=True)
    if not rest.empty:
        other_row = pd.DataFrame(
            [
                {
                    "label": f"Other ({len(rest)} industries)",
                    "share_within": rest[(share_column, "within")].sum(),
                    "share_outside": rest[(share_column, "outside")].sum(),
                    "value_within": rest[(value_basis, "within")].sum(),
                    "value_outside": rest[(value_basis, "outside")].sum(),
                }
            ]
        )
        plot_df = pd.concat([other_row, plot_df], ignore_index=True)

    unit = "positions" if value_basis == "employment" else "establishments"
    total_within = float(mix.loc[mix["location"] == "within", value_basis].sum())
    total_outside = float(mix.loc[mix["location"] == "outside", value_basis].sum())

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["share_outside"],
            y=plot_df["label"],
            orientation="h",
            name=f"Other provinces ({total_outside:,.0f} {unit})",
            marker_color=COLOR_OTHER_BAR,
            customdata=plot_df[["value_outside"]],
            hovertemplate="%{y}<br>Other provinces: %{x:.1%} (%{customdata[0]:,.0f} " + unit + ")<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=plot_df["share_within"],
            y=plot_df["label"],
            orientation="h",
            name=f"Within HQ province ({total_within:,.0f} {unit})",
            marker_color=COLOR_FOCUS,
            customdata=plot_df[["value_within"]],
            hovertemplate="%{y}<br>Within HQ province: %{x:.1%} (%{customdata[0]:,.0f} " + unit + ")<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="group",
        height=max(420, 34 * len(plot_df) + 110),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(title=f"Share of group total ({unit})", tickformat=".0%"),
        yaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    return fig


def make_province_scatter(
    multisource: pd.DataFrame,
    x_column: str,
    y_column: str,
    selected_id_code: str,
    x_label: str,
    y_label: str,
    x_is_share: bool = False,
    y_is_share: bool = False,
) -> go.Figure:
    """All-province scatter with the selected province highlighted."""
    df = multisource.copy()
    df[x_column] = pd.to_numeric(df[x_column], errors="coerce")
    df[y_column] = pd.to_numeric(df[y_column], errors="coerce")
    df = df.dropna(subset=[x_column, y_column])

    is_selected = df["id_code"] == selected_id_code
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.loc[~is_selected, x_column],
            y=df.loc[~is_selected, y_column],
            mode="markers+text",
            text=df.loc[~is_selected, "id_abbr"],
            textposition="top center",
            textfont=dict(size=9, color=COLOR_OTHER_BAR),
            marker=dict(color=COLOR_OTHER_BAR, size=9, opacity=0.7),
            hovertemplate="%{customdata[0]}<extra></extra>",
            customdata=df.loc[~is_selected, ["id_name"]],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[is_selected, x_column],
            y=df.loc[is_selected, y_column],
            mode="markers+text",
            text=df.loc[is_selected, "id_abbr"],
            textposition="top center",
            textfont=dict(size=11, color=COLOR_FOCUS),
            marker=dict(color=COLOR_FOCUS, size=14),
            hovertemplate="%{customdata[0]}<extra></extra>",
            customdata=df.loc[is_selected, ["id_name"]],
            showlegend=False,
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    if x_is_share:
        fig.update_xaxes(tickformat=".0%")
    if y_is_share:
        fig.update_yaxes(tickformat=".0%")
    return fig


def make_ranked_indicator_bar(
    multisource: pd.DataFrame,
    column: str,
    selected_id_code: str,
    label: str,
    is_share: bool = False,
    ascending: bool = False,
) -> go.Figure:
    """All 24 provinces ranked on one indicator, selected province highlighted."""
    df = multisource.copy()
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column]).sort_values(column, ascending=ascending)

    colors = [
        COLOR_FOCUS if code == selected_id_code else COLOR_OTHER_BAR for code in df["id_code"]
    ]
    value_format = "%{y:.1%}" if is_share else "%{y:,.2f}"
    fig = go.Figure(
        go.Bar(
            x=df["id_abbr"],
            y=df[column],
            marker_color=colors,
            customdata=df[["id_name"]],
            hovertemplate=f"%{{customdata[0]}}: {value_format}<extra></extra>",
        )
    )
    national_mean = df[column].mean()
    fig.add_hline(y=national_mean, line_dash="dash", line_color=COLOR_NATIONAL, line_width=1.5)
    fig.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title=label,
        xaxis_title=None,
    )
    if is_share:
        fig.update_yaxes(tickformat=".0%")
    return fig


# --------------------------------------------------------------------------- #
# Page 4 — relatedness / opportunity figures.                                  #
# --------------------------------------------------------------------------- #


def make_feasibility_ranking_bar(
    agg_df: pd.DataFrame,
    state_colors: dict[str, str],
    state_labels: dict[str, str] | None = None,
) -> go.Figure:
    """Section 0: provinces ranked by mean density (column ``value``) across the
    selected industries, each bar colored by the province's best relatedness
    state. The caller decides whether ``value`` is with-self or no-self density."""
    df = agg_df.sort_values("value", ascending=True).copy()
    state_labels = state_labels or {}
    colors = [state_colors.get(str(s), COLOR_OTHER_BAR) for s in df["best_state"]]
    label_names = [state_labels.get(str(s), str(s)) for s in df["best_state"]]
    fig = go.Figure(
        go.Bar(
            x=df["value"],
            y=df["id_name"],
            orientation="h",
            marker_color=colors,
            customdata=np.column_stack(
                [label_names, df["n_specialized"], df["n_present"], df["n_absent"], df["n_classes"]]
            ),
            hovertemplate=(
                "%{y}<br>Mean relatedness density: %{x:.3f}<br>"
                "Best margin: %{customdata[0]}<br>"
                "Selected industries — specialized: %{customdata[1]}, "
                "present: %{customdata[2]}, absent: %{customdata[3]} "
                "(of %{customdata[4]})<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=max(420, 20 * len(df) + 80),
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Mean relatedness density (across selected industries)",
        yaxis_title=None,
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


def make_proximity_heatmap(
    matrix: pd.DataFrame,
    focal: str,
    neighbors: list[str],
    labels: dict[str, str],
    colorscale: str = "Blues",
    section_lookup: dict[str, str] | None = None,
) -> go.Figure:
    """Section A.1: symmetric-φ heatmap for the focal industry and its top
    neighbors. The diagonal (self-proximity = 1) is masked so it doesn't read
    as the color-scale maximum. ``section_lookup`` (4-digit class -> section
    letter) prefixes the code on the tick labels, e.g. 'A0311'."""
    section_lookup = section_lookup or {}
    codes = [focal] + [c for c in neighbors if c != focal]
    sub = matrix.loc[codes, codes].astype(float).copy()
    z = sub.values.copy()
    np.fill_diagonal(z, np.nan)

    def tick(code: str) -> str:
        display_code = f"{section_lookup.get(code, '')}{code}"
        name = str(labels.get(code, "")).strip()
        return f"{display_code} · {name[:26]}" if name else display_code

    ticks = [tick(c) for c in codes]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=ticks,
            y=ticks,
            colorscale=colorscale,
            zmin=0,
            zmax=float(np.nanmax(z)) if np.isfinite(np.nanmax(z)) else 1.0,
            colorbar=dict(title="φ (proximity)", thickness=14),
            hovertemplate="%{y}<br>%{x}<br>φ = %{z:.3f}<extra></extra>",
            hoverongaps=False,
        )
    )
    fig.update_layout(
        height=max(480, 30 * len(codes) + 140),
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(tickangle=45, tickfont=dict(size=9), side="bottom"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
    )
    return fig


def make_density_presence_scatter(
    prov_df: pd.DataFrame,
    state_colors: dict[str, str],
    state_labels: dict[str, str] | None = None,
    show_quadrants: bool = False,
) -> go.Figure:
    """Section A: one dot per province for the focal class — x = presence
    ``rca/(1+rca)`` (0.5 = RCA 1), y = density (column ``value``), colored by
    state, labeled with ``id_abbr``. Separates 'already there' from 'fertile
    ground'. ``show_quadrants`` adds region labels (used in the no-self view,
    where the 'fertile ground' corner is genuinely populated)."""
    df = prov_df.copy()
    df["rca"] = pd.to_numeric(df["rca"], errors="coerce").fillna(0.0)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["presence"] = df["rca"] / (1.0 + df["rca"])
    state_labels = state_labels or {}

    fig = go.Figure()
    for state in ["specialized", "present_not_spec", "absent"]:
        part = df[df["state"] == state]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["presence"],
                y=part["value"],
                mode="markers+text",
                text=part["id_abbr"],
                textposition="top center",
                textfont=dict(size=9, color="#555555"),
                marker=dict(color=state_colors.get(state, COLOR_OTHER_BAR), size=11, opacity=0.85),
                name=state_labels.get(state, state),
                customdata=part[["id_name", "rca", "intensity"]],
                hovertemplate=(
                    "%{customdata[0]}<br>RCA: %{customdata[1]:.2f}<br>"
                    "Density: %{y:.3f}<br>Establishments: %{customdata[2]:,.0f}<extra></extra>"
                ),
            )
        )
    fig.add_vline(
        x=0.5, line_dash="dash", line_color="#333333", line_width=1,
        annotation_text="RCA = 1", annotation_position="top",
    )
    if show_quadrants:
        fig.add_annotation(
            x=0.02, y=1.0, xref="x", yref="paper", xanchor="left", yanchor="top",
            text="◀ new-entry candidates<br>(related capabilities, not specialized)",
            showarrow=False, align="left", font=dict(size=10, color="#777777"),
        )
        fig.add_annotation(
            x=0.98, y=1.0, xref="x", yref="paper", xanchor="right", yanchor="top",
            text="already specialized ▶", showarrow=False, align="right",
            font=dict(size=10, color="#777777"),
        )
    fig.update_layout(
        height=440,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Presence  ·  RCA / (1 + RCA)",
        yaxis_title="Relatedness density",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    fig.update_xaxes(range=[-0.03, 1.03])
    return fig


def make_opportunity_bars(
    opp_df: pd.DataFrame,
    state_colors: dict[str, str],
    highlight_col: str = "is_selected_product",
    highlight_color: str = COLOR_FOCUS,
    show_establishments: bool = True,
) -> go.Figure:
    """Section B: horizontal opportunity bars for one RCA state, ordered by
    density (column ``value``, highest at top; caller sets it to the with-self or
    no-self variant). Y labels are the (section-prefixed) class codes only — the
    industry ``name`` and any decoration live in the hover. ``opp_df`` needs:
    label, value, name, rca, intensity, hover_extra, state, is_selected_product.
    Bars whose industry is one of the product's own are outlined in
    ``highlight_color``. ``show_establishments`` drops the establishment line for
    the absent margin (always 0 there)."""
    df = opp_df.sort_values("value", ascending=True).copy()
    if "hover_extra" not in df.columns:
        df["hover_extra"] = ""
    df["hover_extra"] = df["hover_extra"].fillna("")
    colors = [state_colors.get(str(s), COLOR_OTHER_BAR) for s in df["state"]]
    line_widths = [2.4 if bool(h) else 0.0 for h in df.get(highlight_col, False)]
    line_colors = [highlight_color if bool(h) else "rgba(0,0,0,0)" for h in df.get(highlight_col, False)]

    estab_line = "Establishments: %{customdata[2]:,.0f}<br>" if show_establishments else ""
    fig = go.Figure(
        go.Bar(
            x=df["value"],
            y=df["label"],
            orientation="h",
            marker=dict(color=colors, line=dict(color=line_colors, width=line_widths)),
            customdata=df[["name", "rca", "intensity", "hover_extra"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Density: %{x:.3f}<br>RCA: %{customdata[1]:.2f}<br>"
                + estab_line
                + "%{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=max(300, 24 * len(df) + 90),
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Relatedness density",
        yaxis_title=None,
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig
