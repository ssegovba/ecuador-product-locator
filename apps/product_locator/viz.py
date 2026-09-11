from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import (
    ASSIGNMENT_COLORSCALE,
    COLOR_BADGE_BG,
    COLOR_BOUNDARY,
    COLOR_FOCUS,
    COLOR_MISSING,
    COLOR_NATIONAL,
    COLOR_OTHER_BAR,
    COLOR_OTHER_LINE,
    COLOR_SANKEY_REST,
    COLOR_SANKEY_WITHIN,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_ON_FILL,
    COLOR_TEXT_STRONG,
    MAP_CENTER,
    MAP_HEIGHT,
    MAP_STYLE,
    MAP_ZOOM,
    PRESENCE_COLORSCALE,
    SECTOR_COLORS,
    STATE_LABELS,
    STATE_LABELS_ES,
)
from lang import t


# The boundary/marker-outline grey, owned by config.py. Aliased here because every
# figure in this module reaches for it and the short name reads better inline.
BOUNDARY_COLOR = COLOR_BOUNDARY


def _boundary_trace(
    province_geojson: dict,
    ids: list[str],
    color: str,
    width: float,
    id_key: str = "id_code",
) -> go.Scattermap:
    """A line trace tracing the outlines of the named provinces.

    Used to highlight a subset on a choropleth. It cannot be done with a second
    ``Choroplethmap`` carrying a transparent fill: on map traces
    ``marker.opacity`` gates the **whole** rendered location, outline included —
    measured, at opacity 0.0/0.001/0.05 not a single border pixel is drawn, and
    the fill is already visible by the time the line appears. A ``Scattermap``
    line has no such coupling, so the outline renders at full strength over an
    untouched fill.

    Rings are concatenated with ``None`` separators, so one trace covers every
    polygon of every named province (Galápagos is a MultiPolygon of islands).
    """
    wanted = {str(code) for code in ids}
    longitudes: list[float | None] = []
    latitudes: list[float | None] = []
    for feature in province_geojson.get("features", []):
        if str(feature.get("properties", {}).get(id_key)) not in wanted:
            continue
        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        polygons = (
            [geometry.get("coordinates", [])]
            if kind == "Polygon"
            else geometry.get("coordinates", []) if kind == "MultiPolygon" else []
        )
        for polygon in polygons:
            for ring in polygon:
                longitudes.extend([point[0] for point in ring] + [None])
                latitudes.extend([point[1] for point in ring] + [None])
    return go.Scattermap(
        lon=longitudes,
        lat=latitudes,
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip",
        showlegend=False,
        name="",
    )


def fmt_number(value: float | int | None, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "No data"
    return f"{value:,.{decimals}f}"


def fmt_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "No data"
    return f"{value:.1%}"


def fmt_usd(value: float | None) -> str:
    """Compact USD, for card-sized places where the exact dollar is noise."""
    if value is None or pd.isna(value):
        return "no data"
    value = float(value)
    if value >= 1e9:
        return f"${value / 1e9:.1f}B"
    if value >= 1e6:
        return f"${value / 1e6:.1f}M"
    if value >= 1e3:
        return f"${value / 1e3:.0f}k"
    return f"${value:,.0f}"


def fmt_share_precise(value: float | None) -> str:
    """A share as a percentage that never rounds a nonzero value to '0.00%'.

    Two decimals down to 0.1%, then two *significant* digits — a 4-digit
    industry can hold 0.0034% of a country and still be three times
    over-represented in a province, and a hover that says 0.00% next to ×3
    reads as a bug rather than as a small industry.
    """
    if value is None or pd.isna(value):
        return "no data"
    if value == 0:
        return "0%"
    if abs(value) >= 0.001:
        return f"{value:.2%}"
    return f"{value * 100:.2g}%"


def make_province_tile_choropleth(
    province_geojson: dict,
    plot_df: pd.DataFrame,
    value_column: str,
    colorbar_title: str,
    color_scale: str,
    hover_columns: list[str],
    hover_labels: dict[str, str],
    locations_column: str = "codigo_provincia",
    missing_color: str = COLOR_MISSING,
    outline_ids: list[str] | None = None,
    outline_color: str = COLOR_FOCUS,
    outline_width: float = 3.5,
    hover_formats: dict[str, str] | None = None,
) -> go.Figure:
    """Choropleth on a MapLibre tile basemap (plotly >= 5.24 'map' traces).

    ``outline_ids`` draws an overlay of those ``id_code`` provinces outlined in
    ``outline_color`` (e.g. the provinces a product is assigned to); its hover is
    skipped so hover falls through to the value trace beneath.

    ``hover_formats`` maps a hover column to a d3 format spec (``".3f"``,
    ``".1%"``) — without it plotly prints a float at full machine precision,
    which reads as noise on a ratio like a location quotient.
    """
    plot_df = plot_df.copy()
    plot_df[value_column] = pd.to_numeric(plot_df[value_column], errors="coerce")
    hover_formats = hover_formats or {}

    hover_lines = []
    for index, column in enumerate(hover_columns):
        label = hover_labels.get(column, column.replace("_", " ").title())
        spec = hover_formats.get(column)
        hover_lines.append(f"{label}: %{{customdata[{index}]{':' + spec if spec else ''}}}")
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
        # A line trace, not a transparent Choroplethmap — see _boundary_trace:
        # marker.opacity gates the outline as well as the fill, so the overlay
        # that used to live here drew nothing at all.
        fig.add_trace(
            _boundary_trace(
                province_geojson, [str(code) for code in outline_ids],
                outline_color, outline_width,
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
    fig.add_vline(x=0, line_width=1, line_color=COLOR_TEXT_STRONG)
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
            marker=dict(symbol="diamond", size=9, color=COLOR_TEXT_STRONG),
            name="Ecuador",
            hovertemplate="%{y}<br>Ecuador: %{x:.2f}%<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color=COLOR_TEXT_STRONG)
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
        node_colors.append(COLOR_SANKEY_WITHIN if is_inside else COLOR_OTHER_BAR)
        values.append(getattr(row, value_column))
        link_colors.append("rgba(139,0,0,0.35)" if is_inside else "rgba(127,127,127,0.30)")
    if not rest.empty:
        labels.append(f"Other provinces ({len(rest)})")
        node_colors.append(COLOR_SANKEY_REST)
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
        # automargin explicitly on, on both axes: the tick labels are up to
        # 32 chars of 'A0311 · Marine fishing' and the margins here are zero.
        # The browser's bundled plotly.js defaults automargin on and hid this;
        # a static render drew the grid with no labels at all.
        xaxis=dict(tickangle=45, tickfont=dict(size=9), side="bottom", automargin=True),
        yaxis=dict(autorange="reversed", tickfont=dict(size=9), automargin=True),
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
                textfont=dict(size=9, color=BOUNDARY_COLOR),
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
        x=0.5, line_dash="dash", line_color=COLOR_TEXT_STRONG, line_width=1,
        annotation_text="RCA = 1", annotation_position="top",
    )
    if show_quadrants:
        fig.add_annotation(
            x=0.02, y=1.0, xref="x", yref="paper", xanchor="left", yanchor="top",
            text="◀ new-entry candidates<br>(related capabilities, not specialized)",
            showarrow=False, align="left", font=dict(size=10, color=COLOR_TEXT_MUTED),
        )
        fig.add_annotation(
            x=0.98, y=1.0, xref="x", yref="paper", xanchor="right", yanchor="top",
            text="already specialized ▶", showarrow=False, align="right",
            font=dict(size=10, color=COLOR_TEXT_MUTED),
        )
    fig.update_layout(
        height=440,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Presence  ·  RCA / (1 + RCA)",
        yaxis_title="Relatedness density",
        # automargin explicitly on: with zero left/bottom margins a static render
        # drops the y tick labels entirely and clips the x-axis title. The
        # browser's plotly.js defaults it on, which is what hid this.
        xaxis=dict(automargin=True, range=[-0.03, 1.03]),
        yaxis=dict(automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
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


# Variable registry for the Page-4 attractiveness figures. Keys are columns of
# class_labor_profile.csv; "log" says the axis (never the values) is
# log-spaced — jobs per firm is severely right-skewed (median ~5, max ~1,245),
# while hover always reports the raw figure.
ATTRACTIVENESS_VARS: dict[str, dict[str, object]] = {
    "wage_per_fte": {
        "label": "Average wage paid (USD per FTE, national)",
        "short": "Average wage paid",
        "log": False,
        "hover_fmt": "$%{customdata[2]:,.0f}",
        "pctile_gsector": "wage_per_fte_pctile_gsector",
        "pctile_nat": "wage_per_fte_pctile",
    },
    "fte_per_firm": {
        "label": "Average jobs per firm (national)",
        "short": "Average jobs per firm",
        "log": True,
        "hover_fmt": "%{customdata[2]:,.1f}",
        "pctile_gsector": "fte_per_firm_pctile_gsector",
        "pctile_nat": "fte_per_firm_pctile",
    },
}


_ATTRACTIVENESS_TEXT = {
    "wage_per_fte": {
        "label": ("Average wage paid (USD per FTE, national)",
                  "Salario promedio pagado (USD por plaza equivalente, nacional)"),
        "short": ("Average wage paid", "Salario promedio pagado"),
    },
    "fte_per_firm": {
        "label": ("Average jobs per firm (national)",
                  "Empleo promedio por empresa (nacional)"),
        "short": ("Average jobs per firm", "Empleo promedio por empresa"),
    },
}


def attractiveness_var(var: str) -> dict[str, object]:
    """``ATTRACTIVENESS_VARS[var]`` with its text fields in the current language.

    ``log``, ``hover_fmt`` and the two percentile column names are mechanics and
    never translate.
    """
    spec = dict(ATTRACTIVENESS_VARS[str(var)])
    spec.update({f: t(*pair) for f, pair in _ATTRACTIVENESS_TEXT[str(var)].items()})
    return spec


def make_sector_distribution_box(
    profile_df: pd.DataFrame,
    var: str,
    gsector_colors: dict[str, str],
    matched_df: pd.DataFrame,
    national_ref: float,
    focus_value: float | None = None,
    matched_label: str = "This product's industries",
) -> go.Figure:
    """Distribution of one attractiveness variable within each broad sector —
    horizontal box plots with every class drawn as a point (boxes, not violins:
    the smallest sectors carry ~11 classes). Sectors sorted by class count,
    largest on top. ``matched_df`` (rows of ``profile_df``) is overlaid as open
    rings on each class's sector row, legended as ``matched_label`` — the
    appendix page rings one picked industry rather than a product's set, and the
    legend has to say which; ``focus_value`` adds a dotted vertical line for one
    focused class. The dashed line is the national FTE-weighted average
    (``national_ref``).

    ``var`` must be a key of ``ATTRACTIVENESS_VARS``. Jobs per firm uses a log
    *axis* on raw values — hover always shows the actual figure (an industry
    averaging 270 jobs per firm must read 270, never its log).
    """
    spec = ATTRACTIVENESS_VARS[var]
    df = profile_df.dropna(subset=[var]).copy()
    order = df["gsector_label"].value_counts().index.tolist()  # largest first

    def _customdata(part: pd.DataFrame) -> np.ndarray:
        # No sector field: the row and color already say it.
        return np.column_stack([
            part["display_code"],
            part["class_name"].astype(str).str.slice(0, 60),
            part[var],
            part[spec["pctile_gsector"]],
            part[spec["pctile_nat"]],
        ])

    hover = (
        "<b>%{customdata[0]} · %{customdata[1]}</b><br>"
        f"{spec['short']}: {spec['hover_fmt']}<br>"
        "Percentile in its own sector: %{customdata[3]:.0f}<br>"
        "Percentile nationally: %{customdata[4]:.0f}<extra></extra>"
    )

    fig = go.Figure()
    for gsector in order:
        part = df[df["gsector_label"] == gsector]
        color = gsector_colors.get(gsector, COLOR_OTHER_BAR)
        fig.add_trace(
            go.Box(
                x=part[var],
                y=part["gsector_label"],
                orientation="h",
                boxpoints="all",
                jitter=0.5,
                pointpos=-1.5,
                hoveron="points",
                fillcolor=color,
                line=dict(color=BOUNDARY_COLOR, width=1),
                marker=dict(color=color, size=4.5, opacity=0.75),
                customdata=_customdata(part),
                hovertemplate=hover,
                showlegend=False,
            )
        )
    if not matched_df.empty:
        ringed = matched_df.dropna(subset=[var])
        fig.add_trace(
            go.Scatter(
                x=ringed[var],
                y=ringed["gsector_label"],
                mode="markers",
                marker=dict(
                    symbol="circle-open", size=14, color=COLOR_FOCUS,
                    line=dict(width=3, color=COLOR_FOCUS),
                ),
                name=matched_label,
                customdata=_customdata(ringed),
                hovertemplate=hover,
            )
        )
    fig.add_vline(x=national_ref, line_dash="dash", line_color=COLOR_NATIONAL, line_width=1.5)
    # The vline shape takes the data value even on a log axis, but an annotation
    # anchored to that axis wants log10 — two conventions in one layout.
    fig.add_annotation(
        x=np.log10(national_ref) if spec["log"] else national_ref,
        xref="x", y=1.0, yref="paper", yanchor="bottom",
        text="national average", showarrow=False,
        font=dict(size=10, color=COLOR_NATIONAL),
    )
    if focus_value is not None and pd.notna(focus_value):
        fig.add_vline(x=float(focus_value), line_dash="dot", line_color=COLOR_FOCUS, line_width=1.5)
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis_title=spec["label"],
        yaxis_title=None,
        # automargin explicitly on: the bundled browser plotly.js defaults it,
        # static renderers may not, and the sector names are up to 44 chars.
        xaxis=dict(automargin=True),
        yaxis=dict(
            categoryorder="array",
            categoryarray=order[::-1],  # largest sector on top
            tickfont=dict(size=11),
            automargin=True,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        boxgap=0.35,
    )
    if spec["log"]:
        fig.update_xaxes(type="log")
    return fig


def make_attractiveness_feasibility_scatter(
    df: pd.DataFrame,
    var: str,
    state_colors: dict[str, str],
    state_labels: dict[str, str],
    province_name: str,
    label_points: bool = True,
    focus_code: str | None = None,
    x_title: str | None = None,
) -> go.Figure:
    """The feasibility × attractiveness plane for one province. One dot per class
    — x = whatever feasibility measure the caller put in column ``x``,
    y = one national attractiveness variable (``var``). All classes render as
    small muted context dots (the province's whole opportunity space); the
    highlighted set (``is_matched``) is colored by RCA ``state`` and sized by
    national employment (the materiality screen). Median crosshairs over the
    plotted classes cut the plane into quadrants — upper-right = attractive *and*
    feasible here.

    **The x column is the caller's to define, and so is its axis title** (
    ``x_title``): the bottom-up page plots its β feasibility mix in 0-1 rank
    space, and the axis has to say so or the plot and the score would appear to
    disagree. The hover always carries the two **raw** ingredients — relative
    presence (RCA) and related-only relatedness density — because a percentile is
    not a quantity a reader can act on.

    ``df`` needs: x, ``var``, state, is_matched, display_code, class_name,
    gsector_label, rca, density, wage_per_fte, wage_per_fte_pctile_gsector,
    fte_per_firm, fte_per_firm_pctile_gsector, plazas_equiv_estab_nat,
    plazas_equiv_prov / n_estab_prov (establishment-booked employment and
    establishment count in the selected province — same attribution as the
    national figure), n_firms.

    ``is_matched`` is simply *the highlighted set* — the bottom-up page passes the
    top-k per presence state by its opportunity score. ``label_points=False``
    drops the code labels next to the markers, which the page does once the
    highlighted set grows past the point where the text stays readable.
    ``focus_code`` (a 4-digit class) rings one industry wherever it sits,
    highlighted or not: the "where does *this* one land?" question, which the
    shortlist alone cannot answer.
    """
    spec = ATTRACTIVENESS_VARS[var]
    df = df.dropna(subset=["x", var]).copy()
    context = df[~df["is_matched"]]
    matched = df[df["is_matched"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=context["x"],
            y=context[var],
            mode="markers",
            marker=dict(color=COLOR_OTHER_BAR, size=5, opacity=0.35),
            name=t("All industries (context)", "Todas las industrias (contexto)"),
            customdata=np.column_stack([
                context["display_code"],
                context["class_name"].astype(str).str.slice(0, 60),
                context["gsector_label"],
                context["rca"],
                context["density"],
            ]),
            hovertemplate=(
                "%{customdata[0]} · %{customdata[1]}<br>"
                + t("Feasibility: ", "Factibilidad: ")
                + "%{x:.2f}<br>"
                + t("Relative presence: ", "Presencia relativa: ")
                + "×%{customdata[3]:.2f} · "
                + t("related-only density: ", "densidad de relacionadas: ")
                + "%{customdata[4]:.3f}<br>%{customdata[2]}<extra></extra>"
            ),
        )
    )

    # Size the highlighted markers by national employment (area encoding). The
    # spread is huge (p95/p05 ~1,400x over all classes), so an explicit sizeref
    # plus a floor keeps near-zero-FTE classes visible.
    max_plazas = float(matched["plazas_equiv_estab_nat"].max()) if not matched.empty else 1.0
    sizeref = 2.0 * max(max_plazas, 1.0) / (24.0 ** 2)
    # The two raw ingredients of the x value get their own line: the axis carries
    # the mixed percentile, but "relative presence ×3.4, related-only density
    # 0.081" is the pair a reader can actually check.
    matched_hover = (
        "<b>%{customdata[0]} · %{customdata[1]}</b><br>"
        + t("Feasibility (percentile mix): ",
            "Factibilidad (mezcla de percentiles): ")
        + "%{x:.2f}<br>"
        + t("Relative presence: ", "Presencia relativa: ")
        + "×%{customdata[2]:.2f} · "
        + t("related-only density: ", "densidad de relacionadas: ")
        + "%{customdata[3]:.3f}<br>"
        + t("Avg wage paid (national): ", "Salario prom. pagado (nacional): ")
        + "$%{customdata[4]:,.0f} (P%{customdata[5]:.0f}"
        + t(" in its sector)<br>", " en su sector)<br>")
        + t("Avg jobs per firm (national): ",
            "Empleo prom. por empresa (nacional): ")
        + "%{customdata[6]:,.1f} (P%{customdata[7]:.0f}"
        + t(" in its sector)<br>", " en su sector)<br>")
        + t("Employment held nationally: ", "Empleo sostenido a nivel nacional: ")
        + "%{customdata[8]:,.0f} "
        + t("FTE · ", "plazas · ")
        + "%{customdata[9]:,.0f} "
        + t("firms<br>", "empresas<br>")
        + t("Employment in this province: ", "Empleo en esta provincia: ")
        + "%{customdata[10]:,.0f} "
        + t("FTE · ", "plazas · ")
        + "%{customdata[11]:,.0f} "
        + t("establishments<br>", "establecimientos<br>")
        + "%{customdata[12]}<extra></extra>"
    )
    def _card(part: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            part["display_code"],
            part["class_name"].astype(str).str.slice(0, 60),
            part["rca"],
            part["density"],
            part["wage_per_fte"],
            part["wage_per_fte_pctile_gsector"],
            part["fte_per_firm"],
            part["fte_per_firm_pctile_gsector"],
            part["plazas_equiv_estab_nat"],
            part["n_firms"],
            part["plazas_equiv_prov"],
            part["n_estab_prov"],
            part["gsector_label"],
        ])

    for state in ["specialized", "present_not_spec", "absent"]:
        part = matched[matched["state"] == state]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["x"],
                y=part[var],
                mode="markers+text" if label_points else "markers",
                text=part["display_code"] if label_points else None,
                textposition="top center",
                textfont=dict(size=9, color=BOUNDARY_COLOR),
                marker=dict(
                    color=state_colors.get(state, COLOR_OTHER_BAR),
                    size=part["plazas_equiv_estab_nat"].fillna(0.0),
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=7,
                    opacity=0.9,
                    line=dict(width=1, color=COLOR_TEXT_ON_FILL),
                ),
                name=state_labels.get(state, state),
                customdata=_card(part),
                hovertemplate=matched_hover,
            )
        )

    # The searched industry: an open ring on top of whatever it already is —
    # highlighted, or one of the grey context dots the shortlist passed over.
    # Drawn last so it is never hidden under another marker.
    if focus_code:
        focus = df[df["codigo_clase"].astype(str) == str(focus_code)]
        if not focus.empty:
            fig.add_trace(
                go.Scatter(
                    x=focus["x"],
                    y=focus[var],
                    mode="markers+text",
                    text=focus["display_code"],
                    textposition="bottom center",
                    textfont=dict(size=10, color=COLOR_FOCUS),
                    marker=dict(
                        symbol="circle-open", size=22, color=COLOR_FOCUS,
                        line=dict(width=3, color=COLOR_FOCUS),
                    ),
                    name=t("Industry you searched", "Industria que buscó"),
                    customdata=_card(focus),
                    hovertemplate=matched_hover,
                )
            )

    # Median crosshairs over the plotted classes: x is the province's density
    # distribution, y the national attractiveness distribution. add_vline /
    # add_hline take data values on log axes too (plotly >= 6).
    x_med = float(df["x"].median())
    y_med = float(df[var].median())
    fig.add_vline(x=x_med, line_dash="dash", line_color=COLOR_OTHER_BAR, line_width=1)
    fig.add_hline(y=y_med, line_dash="dash", line_color=COLOR_OTHER_BAR, line_width=1)
    fig.add_annotation(
        x=0.99, y=0.99, xref="paper", yref="paper", xanchor="right", yanchor="top",
        text=t("more feasible & more attractive ↗",
               "más factible y más atractiva ↗"),
        showarrow=False,
        font=dict(size=10, color=COLOR_TEXT_MUTED),
    )

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=x_title
        or t(f"Relatedness density in {province_name}",
             f"Densidad de industrias relacionadas en {province_name}"),
        yaxis_title=spec["label"],
        xaxis=dict(automargin=True),
        yaxis=dict(automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        hovermode="closest",
    )
    if spec["log"]:
        # Explicit ticks, the same fix make_relative_presence_bars needed: plotly's
        # default log minor ticks label 0.5 as "5" and 2,000 as "2", so a reader can
        # be a full order of magnitude out. Every decade anchor in range gets a
        # labelled tick and the ambiguous minors go away.
        low, high = float(df[var].min()), float(df[var].max())
        candidates = [
            0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000,
        ]
        ticks = [t for t in candidates if low / 1.6 <= t <= high * 1.6]
        fig.update_yaxes(
            type="log",
            tickvals=ticks or None,
            ticktext=[f"{t:,g}" for t in ticks] or None,
            automargin=True,
        )
    return fig


# --------------------------------------------------------------------------- #
# Bottom-up page — composition treemap and relative-presence bars.             #
# --------------------------------------------------------------------------- #

# Registry for the two composition / relative-presence variables. Keys are
# columns of industry_labor_composition.csv; both are establishment-booked and
# genuinely provincial (that export's wage and jobs-per-firm columns are
# national in every row and stay off these figures entirely).
COMPOSITION_VARS: dict[str, dict[str, str]] = {
    "plazas_equiv": {
        "label": "Employment (FTE positions)",
        "short": "Employment",
        "unit": "FTE",
        "value_fmt": ",.0f",
        "share_col": "plazas_equiv_share",
    },
    "n_estab": {
        "label": "Establishments",
        "short": "Establishments",
        "unit": "establishments",
        "value_fmt": ",.0f",
        "share_col": "n_estab_share",
    },
}


_COMPOSITION_TEXT = {
    "plazas_equiv": {
        "label": ("Employment (FTE positions)", "Empleo (plazas equivalentes)"),
        "short": ("Employment", "Empleo"),
        "unit": ("FTE", "plazas"),
    },
    "n_estab": {
        "label": ("Establishments", "Establecimientos"),
        "short": ("Establishments", "Establecimientos"),
        "unit": ("establishments", "establecimientos"),
    },
}


def composition_var(var: str) -> dict[str, str]:
    """``COMPOSITION_VARS[var]`` with its text fields in the current language.

    ``value_fmt`` and ``share_col`` are mechanics and never translate.
    """
    spec = dict(COMPOSITION_VARS[str(var)])
    spec.update({f: t(*pair) for f, pair in _COMPOSITION_TEXT[str(var)].items()})
    return spec


def make_composition_treemap(
    industries: pd.DataFrame,
    var: str,
    gsector_colors: dict[str, str],
    geography_label: str,
) -> go.Figure:
    """Section 1 (bottom-up): what one geography's registered economy is made of.

    ``industries`` is one row per industry at a single CIIU level, with columns
    ``code``, ``display_code``, ``name``, ``gsector_label``, ``var`` and ``var``'s
    within-geography share. The node table is built here — root → broad sector →
    industry tile — and **every parent value is the sum of its children**, which
    is the only way ``branchvalues="total"`` is safe: plotly renders
    silently-wrong tile sizes on a mismatch instead of raising, so the
    reconciliation is asserted below.

    Hover carries the section-prefixed code, the industry name, the raw value and
    the within-geography share — nothing else. In particular no wage or
    jobs-per-firm: those are national in every row of the export, and mixing a
    national ratio into a provincial composition figure is exactly the confusion
    this page avoids.
    """
    spec = composition_var(var)
    share_column = spec["share_col"]
    df = industries.copy()
    df[var] = pd.to_numeric(df[var], errors="coerce").fillna(0.0)
    df[share_column] = pd.to_numeric(df[share_column], errors="coerce").fillna(0.0)
    df = df[df[var] > 0].sort_values(var, ascending=False)

    sectors = (
        df.groupby("gsector_label", as_index=False)[[var, share_column]]
        .sum()
        .sort_values(var, ascending=False)
    )
    total = float(df[var].sum())

    nodes = pd.DataFrame(
        {
            "id": (
                ["root"]
                + ["g::" + s for s in sectors["gsector_label"].astype(str)]
                + ["i::" + c for c in df["code"].astype(str)]
            ),
            "label": (
                [geography_label]
                + sectors["gsector_label"].astype(str).tolist()
                + df["display_code"].astype(str).tolist()
            ),
            "parent": (
                [""]
                + ["root"] * len(sectors)
                + ["g::" + s for s in df["gsector_label"].astype(str)]
            ),
            "value": (
                [total]
                + sectors[var].astype(float).tolist()
                + df[var].astype(float).tolist()
            ),
            "name": (
                [geography_label]
                + sectors["gsector_label"].astype(str).tolist()
                + df["name"].astype(str).tolist()
            ),
            "share": (
                # Summed from the children like every other parent value, not
                # hardcoded to 1: under the menu filter the displayed slice is a
                # *part* of the geography, and the root hover has to say so.
                [float(sectors[share_column].sum())]
                + sectors[share_column].astype(float).tolist()
                + df[share_column].astype(float).tolist()
            ),
            "gsector": (
                [""]
                + sectors["gsector_label"].astype(str).tolist()
                + df["gsector_label"].astype(str).tolist()
            ),
        }
    )

    # branchvalues="total" tripwire: every parent must equal the sum of its children.
    own = nodes.set_index("id")["value"]
    child_sums = nodes[nodes["parent"] != ""].groupby("parent")["value"].sum()
    for node_id, child_sum in child_sums.items():
        if not np.isclose(float(own[node_id]), float(child_sum), rtol=1e-9, atol=1e-6):
            raise ValueError(
                f"treemap node '{node_id}' is {float(own[node_id]):,.2f} but its children "
                f"sum to {float(child_sum):,.2f} — branchvalues='total' would render "
                "silently-wrong tiles"
            )

    colors = [
        COLOR_OTHER_LINE if not g else gsector_colors.get(g, COLOR_OTHER_BAR)
        for g in nodes["gsector"]
    ]
    fig = go.Figure(
        go.Treemap(
            ids=nodes["id"],
            labels=nodes["label"],
            parents=nodes["parent"],
            values=nodes["value"],
            branchvalues="total",
            marker=dict(colors=colors, line=dict(color="white", width=1)),
            # Tile text is truncated so it stays inside the tile; the hover
            # carries the full name.
            text=nodes["name"].astype(str).str.slice(0, 44),
            texttemplate="<b>%{label}</b><br>%{text}",
            textfont=dict(size=12),
            customdata=np.column_stack([nodes["name"], nodes["share"]]),
            hovertemplate=(
                "<b>%{label}</b> · %{customdata[0]}<br>"
                f"{spec['short']}: %{{value:{spec['value_fmt']}}} {spec['unit']}<br>"
                + "%{customdata[1]:.1%}"
                + t(f" of {geography_label}", f" de {geography_label}")
                + "<extra></extra>"
            ),
            tiling=dict(pad=2),
            pathbar=dict(visible=True),
        )
    )
    fig.update_layout(height=560, margin=dict(l=6, r=6, t=30, b=6))
    return fig


POTENTIAL_VIEWS: dict[str, dict[str, str]] = {
    "potential": {
        "label": "Potential (FTE at the national frontier)",
        "short": "Potential",
        "column": "potential",
    },
    "gap": {
        "label": "Gap to the frontier (FTE)",
        "short": "Gap",
        "column": "gap",
    },
}


_POTENTIAL_VIEW_TEXT = {
    "potential": {
        "label": ("Potential (FTE at the national frontier)",
                  "Potencial (plazas en la frontera nacional)"),
        "short": ("Potential", "Potencial"),
    },
    "gap": {
        "label": ("Gap to the frontier (FTE)", "Brecha hasta la frontera (plazas)"),
        "short": ("Gap", "Brecha"),
    },
}


def potential_view(view: str) -> dict[str, str]:
    """``POTENTIAL_VIEWS[view]`` with its text fields in the current language.

    ``column`` is a dataframe column name and never translates.
    """
    spec = dict(POTENTIAL_VIEWS[str(view)])
    spec.update({f: t(*pair) for f, pair in _POTENTIAL_VIEW_TEXT[str(view)].items()})
    return spec


def make_potential_treemap(
    tiles: pd.DataFrame,
    view: str,
    color_by: str,
    state_colors: dict[str, str],
    state_labels: dict[str, str],
    gsector_colors: dict[str, str],
    province_label: str,
) -> go.Figure:
    """The shortlist priced against Ecuador's own frontier, as a treemap.

    ``tiles`` is one row per shortlisted industry that has a benchmark, with
    ``codigo_clase``, ``display_code``, ``class_name``, ``state``,
    ``gsector_label``, ``leader_id_name``, ``leader_plazas``, ``leader_estab``,
    ``current``, ``potential`` and ``gap``. ``view`` picks which of the last two
    sizes the tiles; ``color_by`` recolors the leaves — ``"sector"`` (the broad
    sector, the same ``GSECTOR_COLORS`` the composition treemap uses, so the two
    figures can be read against each other) or ``"state"`` (what the province has
    of the industry today, the same ``STATE_COLORS`` as the plane above).

    **The hierarchy is always broad sector, whichever dimension is coloring.**
    Grouping by the colour dimension was the first build (2026-08-24) and was
    replaced the same day: a treemap's layout is a function of its hierarchy and
    its values, so regrouping on every toggle re-tiled the whole figure and the
    reader lost the industry they were looking at. With the hierarchy fixed and
    only ``marker.colors`` changing, **every tile keeps its exact position** and
    the toggle does what it says — a recolour, not a new figure. The trade-off,
    accepted: under ``"state"`` a sector block holds mixed colours and the states
    have no block of their own. That is the point of an overlay, and the section
    caption says which is which.

    Switching ``view`` *does* move tiles, and must: potential and gap are
    different quantities, so the areas genuinely differ.

    Parent tiles keep their **sector** colour in both modes — they label the fixed
    structure, and a group has no single presence state. Parent values are summed
    from their children and the reconciliation is asserted, exactly as in
    :func:`make_composition_treemap`: ``branchvalues="total"`` renders
    silently-wrong tile sizes on a mismatch instead of raising.

    Zero-area tiles are dropped by plotly of their own accord (the gap view's
    already-at-the-frontier rows); the caller counts and names them rather than
    letting them vanish silently. The hover is deliberately **quantitative
    only** — the leader province with the employment and units behind it, then
    this province's current stock, its potential and the gap. No commentary about
    which base the presence states rest on: that belongs in the section caption,
    where a reader meets it once.
    """
    spec = potential_view(view)
    value_column = spec["column"]
    df = tiles.copy()
    for column in ["current", "potential", "gap", "leader_plazas", "leader_estab"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    df = df[df[value_column] > 0].sort_values(value_column, ascending=False)
    if df.empty:
        # A legitimate state, not an error: on the gap view every shortlisted
        # industry can already sit at the frontier. Say so on the canvas rather
        # than handing plotly an empty node table.
        fig = go.Figure()
        fig.add_annotation(
            text=t(
                f"No shortlisted industry has a measurable "
                f"{spec['short'].lower()} in {province_label}.",
                f"Ninguna industria de la lista corta tiene "
                f"{spec['short'].lower()} medible en {province_label}.",
            ),
            showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
            font=dict(size=13, color=COLOR_TEXT_MUTED),
        )
        fig.update_layout(
            height=220, margin=dict(l=6, r=6, t=30, b=6),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return fig

    # The hierarchy never depends on ``color_by`` — that is what keeps tiles still
    # when the toggle flips. Sorting is by value alone, for the same reason.
    by_state = str(color_by) == "state"
    df["group"] = df["gsector_label"].astype(str)
    leaf_colors = (
        df["state"].astype(str).map(state_colors)
        if by_state
        else df["group"].map(gsector_colors)
    ).fillna(COLOR_OTHER_BAR)

    groups = (
        df.groupby("group", as_index=False)
        .agg(value=(value_column, "sum"))
        .sort_values("value", ascending=False)
    )
    # Parents always carry the sector colour: they label the structure, and a sector
    # block has no single presence state to take a colour from.
    group_colors = groups["group"].map(gsector_colors).fillna(COLOR_OTHER_BAR)
    total = float(df[value_column].sum())

    nodes = pd.DataFrame(
        {
            "id": (
                ["root"]
                + ["g::" + g for g in groups["group"].astype(str)]
                + ["i::" + c for c in df["codigo_clase"].astype(str)]
            ),
            "label": (
                [province_label]
                + groups["group"].astype(str).tolist()
                + df["display_code"].astype(str).tolist()
            ),
            "parent": [""] + ["root"] * len(groups) + ["g::" + g for g in df["group"].astype(str)],
            "value": (
                [total]
                + groups["value"].astype(float).tolist()
                + df[value_column].astype(float).tolist()
            ),
            "name": (
                [province_label]
                + groups["group"].astype(str).tolist()
                + df["class_name"].astype(str).tolist()
            ),
            "color": [COLOR_OTHER_LINE] + group_colors.tolist() + leaf_colors.tolist(),
            "sector": [""] * (1 + len(groups)) + df["group"].tolist(),
            "state_label": [""] * (1 + len(groups)) + [
                state_labels.get(str(s), str(s)) for s in df["state"]
            ],
            "leader": [""] * (1 + len(groups)) + df["leader_id_name"].astype(str).tolist(),
            "leader_plazas": [0.0] * (1 + len(groups)) + df["leader_plazas"].tolist(),
            "leader_estab": [0.0] * (1 + len(groups)) + df["leader_estab"].tolist(),
            "current": [0.0] * (1 + len(groups)) + df["current"].tolist(),
            "potential": [0.0] * (1 + len(groups)) + df["potential"].tolist(),
            "gap": [0.0] * (1 + len(groups)) + df["gap"].tolist(),
            "is_leaf": [False] * (1 + len(groups)) + [True] * len(df),
        }
    )

    # branchvalues="total" tripwire: every parent must equal the sum of its children.
    own = nodes.set_index("id")["value"]
    child_sums = nodes[nodes["parent"] != ""].groupby("parent")["value"].sum()
    for node_id, child_sum in child_sums.items():
        if not np.isclose(float(own[node_id]), float(child_sum), rtol=1e-9, atol=1e-6):
            raise ValueError(
                f"potential treemap node '{node_id}' is {float(own[node_id]):,.2f} but its "
                f"children sum to {float(child_sum):,.2f} — branchvalues='total' would render "
                "silently-wrong tiles"
            )

    # Leaves carry the full reading; the group and root rows would otherwise print
    # zeros for every per-industry number, which reads as missing data.
    fte = t("FTE", "plazas")
    frontier_word = t("Frontier", "Frontera")
    across = t("across", "en")
    estab_word = t("establishments", "establecimientos")
    today = t("today", "hoy")
    potential_word = t("potential", "potencial")
    gap_word = t("gap", "brecha")
    hover = [
        (
            f"<b>{label}</b> · {name}<br>"
            f"{sector} · {state}<br>"
            f"{spec['short']}: {value:,.0f} {fte}<br>"
            f"{frontier_word}: {leader} — {lplazas:,.0f} {fte} {across} "
            f"{lestab:,.0f} {estab_word}<br>"
            f"{province_label} {today}: {current:,.0f} {fte} · "
            f"{potential_word} {potential:,.0f} · {gap_word} {gap:,.0f}<extra></extra>"
        )
        if leaf
        else f"<b>{label}</b><br>{spec['short']}: {value:,.0f} {fte}<extra></extra>"
        for label, name, sector, state, value, leader, lplazas, lestab, current, potential, gap, leaf in zip(
            nodes["label"], nodes["name"], nodes["sector"], nodes["state_label"],
            nodes["value"], nodes["leader"], nodes["leader_plazas"], nodes["leader_estab"],
            nodes["current"], nodes["potential"], nodes["gap"], nodes["is_leaf"],
        )
    ]

    fig = go.Figure(
        go.Treemap(
            ids=nodes["id"],
            labels=nodes["label"],
            parents=nodes["parent"],
            values=nodes["value"],
            branchvalues="total",
            marker=dict(colors=nodes["color"].tolist(), line=dict(color="white", width=1)),
            text=nodes["name"].astype(str).str.slice(0, 44),
            texttemplate="<b>%{label}</b><br>%{text}",
            textfont=dict(size=12),
            hovertemplate=hover,
            tiling=dict(pad=2),
            # Fixed order, so the layout cannot depend on anything but the values.
            sort=True,
            pathbar=dict(visible=True),
        )
    )
    fig.update_layout(height=520, margin=dict(l=6, r=6, t=30, b=6))
    return fig


def make_density_contribution_bar(
    contrib: pd.DataFrame,
    state_colors: dict[str, str],
    state_labels: dict[str, str],
    focal_code: str,
    province_label: str,
) -> go.Figure:
    """Section 3 expander: where one industry's feasibility score comes from.

    Horizontal bars of each contributor's share of the density numerator
    (``φ_ij × M̃_j`` over the total), colored by **what the province has of that
    contributor** — the same green / amber / blue as the plane, so "this score
    rests on things you already do" is readable without arithmetic. The focal
    industry's own term is drawn first and labelled *(itself)*: on the with-self
    view it routinely carries 60–99% of the score, and seeing that is the point.

    ``contrib`` comes from ``metrics.density_contributions`` joined to names and
    each contributor's ``state`` in the province.
    """
    df = contrib.copy().sort_values("contribution_share", ascending=True)
    df["is_self"] = df["codigo_clase"].astype(str) == str(focal_code)
    self_word = t("itself", "ella misma")
    labels = [
        f"{code} · {self_word}" if is_self else str(code)
        for code, is_self in zip(df["display_code"], df["is_self"])
    ]
    colors = [state_colors.get(str(s), COLOR_OTHER_BAR) for s in df["state"]]

    fig = go.Figure(
        go.Bar(
            x=df["contribution_share"],
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(color="white", width=0.5)),
            customdata=np.column_stack([
                df["name"].astype(str).str.slice(0, 56),
                [state_labels.get(str(s), str(s)) for s in df["state"]],
                df["phi"],
                df["rca"],
                df["intensity"].fillna(0.0),
                df["contribution_share"],
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + "%{customdata[5]:.1%}"
                + t(" of the score<br>", " del puntaje<br>")
                + t("Proximity to this industry: ", "Proximidad a esta industria: ")
                + "φ = %{customdata[2]:.3f}<br>"
                + t(f"In {province_label}: ", f"En {province_label}: ")
                + "%{customdata[1]} · RCA %{customdata[3]:.2f} · "
                + "%{customdata[4]:,.0f} "
                + t("establishments", "establecimientos")
                + "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=max(240, 28 * len(df) + 90),
        margin=dict(l=6, r=6, t=10, b=10),
        xaxis=dict(
            title=t("Share of the feasibility score",
                    "Parte del puntaje de factibilidad"),
            tickformat=".0%",
            automargin=True,
        ),
        # type="category" is load-bearing: CIIU codes look numeric, and plotly
        # otherwise builds a *numeric* y-axis on which the bars vanish.
        yaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=labels,
            title=None,
            tickfont=dict(size=11),
            automargin=True,
        ),
        showlegend=False,
    )
    return fig


def make_relative_presence_bars(
    top_df: pd.DataFrame,
    bottom_df: pd.DataFrame,
    var: str,
    gsector_colors: dict[str, str],
    peer_label: str,
    province_label: str,
) -> go.Figure:
    """Section 2 (bottom-up): the industries a province is most — and least —
    concentrated in, relative to a peer economy.

    Two blocks in **one** figure on a shared **log** x-axis: over- and
    under-representation are ratios, so ×4 and ×0.25 must sit the same distance
    from the ×1 reference line. ``top_df`` / ``bottom_df`` arrive ranked and
    trimmed by the caller (zeros and guard-hidden industries are removed
    upstream — a ratio of 0 has no place on a log axis). Both need:
    display_code, name, gsector_label, rp, share_prov, share_peer, value,
    n_estab, value_peer, n_estab_peer.

    **The mark is a stem from ×1 to the value, not a filled bar.** A bar on a
    log axis grows from whatever floor the axis happens to take, so its length
    encodes the axis range rather than the data — every industry reads as
    roughly the same size (caught in the render, invisible in the numbers). A
    stem anchored at ×1 has length |log(ratio)|, which is exactly the
    multiplicative distance from parity in both directions.

    Y labels are section-prefixed codes only; the industry name, its broad
    sector, and **both sides of the ratio in full** — each side's share plus the
    raw value and establishment count behind it — live in the hover. That is the
    disclosure that catches the distorters the ranking guard misses, and the
    reason a peer share of 0.0074% next to a ×3.09 ratio reads as a small
    industry rather than as a broken number.
    """
    spec = composition_var(var)
    block = pd.concat(
        [bottom_df.sort_values("rp", ascending=True), top_df.sort_values("rp", ascending=True)],
        ignore_index=True,
    )
    n_bottom = len(bottom_df)
    colors = [gsector_colors.get(str(g), COLOR_OTHER_BAR) for g in block["gsector_label"]]

    # The raw value and the establishment count are one and the same on the
    # establishments view — say it once there, twice on the employment view.
    def _side(share_index: int, value_index: int, estab_index: int) -> str:
        estab_word = t("establishments", "establecimientos")
        if var == "n_estab":
            return (
                f"%{{customdata[{share_index}]}} "
                f"(%{{customdata[{estab_index}]:,.0f}} {estab_word})"
            )
        across = t("across", "en")
        return (
            f"%{{customdata[{share_index}]}} "
            f"(%{{customdata[{value_index}]:{spec['value_fmt']}}} {spec['unit']} "
            f"{across} %{{customdata[{estab_index}]:,.0f}} {estab_word})"
        )

    fig = go.Figure()
    # Stems first (below the dots). Shapes take data values on a log axis; on a
    # categorical axis they take the 0-based category index.
    for position, (value, color) in enumerate(zip(block["rp"], colors)):
        fig.add_shape(
            type="line",
            x0=1.0, x1=float(value), xref="x",
            y0=position, y1=position, yref="y",
            line=dict(color=color, width=5),
            layer="below",
        )
    fig.add_trace(
        go.Scatter(
            x=block["rp"],
            y=block["display_code"],
            mode="markers",
            marker=dict(color=colors, size=12, line=dict(color="white", width=1)),
            customdata=np.column_stack([
                block["display_code"],
                block["name"].astype(str).str.slice(0, 60),
                block["gsector_label"],
                block["rp"],
                [fmt_share_precise(s) for s in block["share_prov"]],
                block["value"],
                block["n_estab"],
                [fmt_share_precise(s) for s in block["share_peer"]],
                block["value_peer"],
                block["n_estab_peer"],
            ]),
            hovertemplate=(
                "<b>%{customdata[0]} · %{customdata[1]}</b><br>"
                "%{customdata[2]}<br>"
                + t("Relative presence: ", "Presencia relativa: ")
                + "×%{customdata[3]:,.2f}<br>"
                + t(f"Share in {province_label}: ",
                    f"Participación en {province_label}: ")
                + f"{_side(4, 5, 6)}<br>"
                + t(f"Share in {peer_label}: ",
                    f"Participación en {peer_label}: ")
                + f"{_side(7, 8, 9)}"
                + "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color=COLOR_NATIONAL, line_width=1.5)
    # On a log axis the vline *shape* takes the data value (1.0) while an
    # annotation anchored to the same axis wants log10 — two conventions in one
    # layout object, and log10(1) = 0 is the one place they happen to agree.
    fig.add_annotation(
        x=0.0, xref="x", y=1.0, yref="paper", yanchor="bottom",
        text=t(f"same share as {peer_label}",
               f"misma participación que {peer_label}"),
        showarrow=False,
        font=dict(size=10, color=COLOR_NATIONAL),
    )
    if n_bottom and len(block) > n_bottom:
        # Separator between the under- and over-represented blocks. Category
        # positions on a categorical axis are 0-based indices.
        fig.add_hline(
            y=n_bottom - 0.5, line_dash="dot", line_color=COLOR_OTHER_BAR, line_width=1
        )
        fig.add_annotation(
            x=1.0, xref="paper", xanchor="right", y=len(block) - 0.4, yref="y", yanchor="bottom",
            text=t(f"more concentrated than {peer_label} ▲",
                   f"más concentrada que {peer_label} ▲"),
            showarrow=False,
            font=dict(size=10, color=COLOR_OTHER_BAR),
        )
        fig.add_annotation(
            x=1.0, xref="paper", xanchor="right", y=n_bottom - 0.45, yref="y", yanchor="top",
            text=t(f"▼ less concentrated than {peer_label}",
                   f"▼ menos concentrada que {peer_label}"),
            showarrow=False,
            font=dict(size=10, color=COLOR_OTHER_BAR),
        )

    # Explicit ratio ticks: plotly's log minor ticks read "5 · 0.01 · 2 · 5 · 0.1"
    # across a wide range, which is unreadable as a ratio scale.
    candidates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 10, 25, 100, 1000]
    low, high = float(block["rp"].min()), float(block["rp"].max())
    ticks = [t for t in candidates if low / 1.6 <= t <= high * 1.6]
    fig.update_layout(
        height=max(420, 26 * len(block) + 120),
        margin=dict(l=6, r=6, t=45, b=10),
        xaxis_title=t(
            f"Relative presence — share in {province_label} ÷ share in {peer_label}",
            f"Presencia relativa: participación en {province_label} ÷ participación en "
            f"{peer_label}",
        ),
        yaxis_title=None,
        showlegend=False,
        # automargin explicitly on: the browser's bundled plotly.js defaults it,
        # static renderers (kaleido) may not, and the tick labels vanish then.
        xaxis=dict(
            type="log",
            automargin=True,
            tickvals=ticks or None,
            ticktext=[f"×{t:g}" for t in ticks] or None,
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=block["display_code"].tolist(),
            tickfont=dict(size=11),
            automargin=True,
        ),
    )
    return fig


# --------------------------------------------------------------------------- #
# Top-down lens — opportunity presence choropleth, ranking bars, assignment    #
# heatmap.                                                                     #
# --------------------------------------------------------------------------- #

# The two presence views. "lq" is a ratio against the national average (×1 is
# the reference point); "scale" is a share of the industry's national total.
PRESENCE_VIEWS: dict[str, dict[str, object]] = {
    "lq": {
        "label": "Specialization (location quotient)",
        "short": "Specialization",
        "axis": "Weighted presence  ·  ×1 = the national average",
        "colorbar": "Presence (×national)",
        "hover_fmt": "×%{customdata[1]:,.3f}",
        "hover_number": ".3f",
        "tickformat": None,
        "reference": 1.0,
    },
    "scale": {
        "label": "Scale (share of the industry nationally)",
        "short": "Scale",
        "axis": "Weighted share of the industry's national total",
        "colorbar": "Share of national",
        "hover_fmt": "%{customdata[1]:.2%}",
        "hover_number": ".2%",
        "tickformat": ".0%",
        "reference": None,
    },
}

PRESENCE_BASES = {"plazas": "Employment (FTE positions)", "estab": "Establishments"}

# The text half of the two dicts above, paired for translation. Kept separate so
# PRESENCE_VIEWS itself stays a plain dict of *mechanics* (hover formats, tick
# formats, the reference line) that no language can change, and so the pages
# still importing the English dicts keep working while the sweep lands page by
# page. Resolve through presence_view() / presence_bases(), never by hand.
_PRESENCE_VIEW_TEXT = {
    "lq": {
        "label": ("Specialization (location quotient)",
                  "Especialización (cociente de localización)"),
        "short": ("Specialization", "Especialización"),
        "axis": ("Weighted presence  ·  ×1 = the national average",
                 "Presencia ponderada  ·  ×1 = el promedio nacional"),
        "colorbar": ("Presence (×national)", "Presencia (×nacional)"),
    },
    "scale": {
        "label": ("Scale (share of the industry nationally)",
                  "Escala (participación nacional de la industria)"),
        "short": ("Scale", "Escala"),
        "axis": ("Weighted share of the industry's national total",
                 "Participación ponderada en el total nacional de la industria"),
        "colorbar": ("Share of national", "Participación nacional"),
    },
}

_PRESENCE_BASE_TEXT = {
    "plazas": ("Employment (FTE positions)", "Empleo (plazas equivalentes)"),
    "estab": ("Establishments", "Establecimientos"),
}


def presence_view(view: str) -> dict[str, object]:
    """``PRESENCE_VIEWS[view]`` with its text fields in the current language."""
    spec = dict(PRESENCE_VIEWS[str(view)])
    spec.update({field: t(*pair) for field, pair in _PRESENCE_VIEW_TEXT[str(view)].items()})
    return spec


def presence_bases() -> dict[str, str]:
    """``PRESENCE_BASES`` in the current language."""
    return {code: t(*pair) for code, pair in _PRESENCE_BASE_TEXT.items()}


def state_labels() -> dict[str, str]:
    """The three presence-state labels in the current language.

    Every figure factory already takes the label map as an argument, so pages
    pass this instead of ``config.STATE_LABELS`` and the figures follow the
    language without any of them learning about ``t()``. The parenthetical
    qualifier is preserved in both, because callers derive the short form by
    stripping " (...)".
    """
    return {
        key: t(STATE_LABELS[key], STATE_LABELS_ES[key]) for key in STATE_LABELS
    }

# Safety bound on a hover line. plotly's `hoverlabel` has no width or wrap
# property, so any wrapping has to be done before it gets here — the caller is
# expected to trim the product *name* (34 chars keeps the widest line near 54,
# ~335px, against 69-char names that crowd the figure). This is only the backstop
# that keeps a pathological label from stretching the box.
_HOVER_LABEL_CHARS = 70


def make_presence_choropleth(
    province_geojson: dict,
    plot_df: pd.DataFrame,
    view: str,
    top_codes: list[str] | None = None,
) -> go.Figure:
    """Blended presence of one opportunity's industries, across the 24 provinces.

    Same tile-map conventions as everything else in the app (this delegates to
    :func:`make_province_tile_choropleth`). ``plot_df`` needs ``id_code``,
    ``id_name``, ``score``, ``direct``, ``anchor`` and ``n_industries``;
    ``top_codes`` outlines the provinces the ranking assigns this product to, so
    the map and the bars below it can never tell different stories.

    A province scoring exactly zero keeps its value rather than becoming "no
    data": zero here is a finding — none of the product's industries is
    observable there — not a gap in the source.
    """
    spec = presence_view(view)
    hover_columns = ["id_name", "score", "direct", "anchor", "n_industries"]
    labels = {
        "id_name": t("Province", "Provincia"),
        "score": t(
            f"Weighted presence ({str(spec['short']).lower()})",
            f"Presencia ponderada ({str(spec['short']).lower()})",
        ),
        "direct": t("· from the product's own industries",
                    "· de las industrias del producto"),
        "anchor": t("· from the anchors' industries",
                    "· de las industrias de las anclas"),
        "n_industries": t("Industries observable here", "Industrias observables aquí"),
    }
    # Three decimals on a location quotient (two on a share): the underlying
    # float carries machine precision and reads as noise unformatted.
    number_format = str(spec["hover_number"])
    formats = {"score": number_format, "direct": number_format, "anchor": number_format}
    return make_province_tile_choropleth(
        province_geojson=province_geojson,
        plot_df=plot_df.copy(),
        value_column="score",
        colorbar_title=str(spec["colorbar"]),
        color_scale=PRESENCE_COLORSCALE,
        hover_columns=hover_columns,
        hover_labels=labels,
        locations_column="id_code",
        outline_ids=top_codes,
        outline_color=COLOR_FOCUS,
        outline_width=3.0,
        hover_formats=formats,
    )


def make_presence_ranking_bar(
    rank_df: pd.DataFrame,
    view: str,
    top_codes: list[str] | None = None,
    show_anchor: bool = True,
) -> go.Figure:
    """The page's headline answer: provinces ranked by blended presence.

    ``rank_df`` needs ``id_code``, ``id_name``, ``score``, ``direct``,
    ``anchor``, ``n_industries``. Bars are colored by whether the province is one
    of this product's **top-k assignments** — the same set Page B counts — so the
    ranking cut is visible rather than implied. In the location-quotient view a
    dashed ×1 line marks the national average; in the scale view the axis is
    formatted as a share and no reference line applies (a share of a national
    total has no natural parity point).
    """
    spec = presence_view(view)
    df = rank_df.sort_values("score", ascending=True).copy()
    picks = set(top_codes or [])
    colors = [COLOR_FOCUS if code in picks else COLOR_OTHER_BAR for code in df["id_code"]]

    anchor_line = (
        t("· from the anchors' industries: ", "· de las industrias de las anclas: ")
        + "%{customdata[3]:,.3f}<br>"
        if show_anchor
        else ""
    )
    fig = go.Figure(
        go.Bar(
            x=df["score"],
            y=df["id_name"],
            orientation="h",
            marker_color=colors,
            customdata=np.column_stack([
                df["id_name"], df["score"], df["direct"], df["anchor"], df["n_industries"],
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + t("Weighted presence: ", "Presencia ponderada: ")
                + f"{spec['hover_fmt']}<br>"
                + t("· from the product's own industries: ",
                    "· de las industrias del producto: ")
                + "%{customdata[2]:,.3f}<br>"
                + anchor_line
                + t("Industries observable here: ", "Industrias observables aquí: ")
                + "%{customdata[4]}<extra></extra>"
            ),
        )
    )
    if spec["reference"] is not None:
        fig.add_vline(
            x=float(spec["reference"]), line_dash="dash",
            line_color=COLOR_NATIONAL, line_width=1.5,
        )
        fig.add_annotation(
            x=float(spec["reference"]), xref="x", y=1.0, yref="paper", yanchor="bottom",
            text=t("national average", "promedio nacional"), showarrow=False,
            font=dict(size=10, color=COLOR_NATIONAL),
        )
    fig.update_layout(
        height=max(420, 22 * len(df) + 110),
        margin=dict(l=6, r=6, t=45, b=10),
        yaxis_title=None,
        showlegend=False,
        # automargin explicitly on: the browser's bundled plotly.js defaults it,
        # static renderers (kaleido) may not, and province names run to 30 chars.
        xaxis=dict(title=str(spec["axis"]), automargin=True, tickformat=spec["tickformat"]),
        yaxis=dict(
            categoryorder="array",
            categoryarray=df["id_name"].tolist(),
            tickfont=dict(size=11),
            automargin=True,
        ),
    )
    return fig


def _add_highlight_bands(
    fig: go.Figure,
    matrix: pd.DataFrame,
    highlight_rows: list[str] | None,
    highlight_columns: list[str] | None,
) -> None:
    """Band the named rows and columns and outline their intersection.

    On a categorical axis shape coordinates are 0-based category indices.
    ``layer="above"`` is load-bearing: heatmap cells are opaque, so a band drawn
    underneath shows only through the 1px `xgap`/`ygap` and reads as a hairline
    rather than a band.
    """
    row_index = {code: position for position, code in enumerate(matrix.index)}
    column_index = {code: position for position, code in enumerate(matrix.columns)}
    marked_rows = [row_index[c] for c in (highlight_rows or []) if c in row_index]
    marked_columns = [column_index[c] for c in (highlight_columns or []) if c in column_index]
    band = dict(fillcolor=COLOR_FOCUS, opacity=0.12, line_width=0, layer="above")
    for position in marked_rows:
        fig.add_shape(
            type="rect", xref="paper", x0=0, x1=1, yref="y",
            y0=position - 0.5, y1=position + 0.5, **band,
        )
    for position in marked_columns:
        fig.add_shape(
            type="rect", yref="paper", y0=0, y1=1, xref="x",
            x0=position - 0.5, x1=position + 0.5, **band,
        )
    for row_position in marked_rows:
        for column_position in marked_columns:
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=column_position - 0.5, x1=column_position + 0.5,
                y0=row_position - 0.5, y1=row_position + 0.5,
                fillcolor="rgba(0,0,0,0)", line=dict(color=COLOR_FOCUS, width=2),
                layer="above",
            )


def make_opportunity_heatmap(
    product_matrix: pd.DataFrame,
    row_labels: dict[str, str],
    column_labels: dict[str, str],
    column_names: dict[str, str],
    province_totals: dict[str, int] | None = None,
    highlight_rows: list[str] | None = None,
    highlight_columns: list[str] | None = None,
) -> go.Figure:
    """Where each of the 50 opportunities is allocated — the **primitive** grid.

    One row per opportunity, one column per province; the cell carries the
    province's **rank** within that product's top-k, blank where it is not
    assigned. Every row therefore holds exactly *k* marks (fewer only where the
    zero-score rule bites), and a **column total is a count of opportunities** —
    the number a policymaker actually wants, and a different quantity from the
    industry grid's column totals, which count industry marks and roughly double
    it.

    Unassigned cells are **NaN, not 0**: the scale is reversed so rank 1 reads
    darkest, and a 0 would then be the darkest value on the plot rather than a
    blank. ``hoverongaps=False`` keeps the blanks from hovering.
    """
    ranks = product_matrix.to_numpy(dtype=float)
    z = np.where(ranks > 0, ranks, np.nan)
    rows = [row_labels.get(code, code) for code in product_matrix.index]
    columns = [column_labels.get(code, code) for code in product_matrix.columns]
    text = np.where(ranks > 0, np.char.mod("%d", product_matrix.to_numpy(dtype=int)), "")
    totals = province_totals or {}
    province_layer = np.array([
        [
            t(
                f"{column_names.get(c, c)} — top pick for "
                f"{int(totals.get(c, 0))} of the {len(product_matrix.index)} "
                "opportunities",
                f"{column_names.get(c, c)}: elegida para "
                f"{int(totals.get(c, 0))} de las {len(product_matrix.index)} "
                "oportunidades",
            )
            for c in product_matrix.columns
        ]
    ] * len(product_matrix.index))
    top = float(np.nanmax(z)) if np.isfinite(z).any() else 1.0

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=columns,
            y=rows,
            colorscale=ASSIGNMENT_COLORSCALE,
            reversescale=True,  # rank 1 = darkest
            zmin=1,
            # zmax is stretched half a rank past the last one so the weakest rank
            # still lands on a visible mid-tone. At an exact zmax it renders
            # near-white and reads as an empty cell — which is the opposite of
            # what this grid is for, since every mark is an assignment.
            zmax=top + 0.5,
            colorbar=dict(title=t("Rank", "Puesto"), thickness=14, dtick=1),
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10),
            customdata=province_layer,
            hovertemplate=(
                "<b>%{y}</b><br>%{customdata}<br>"
                + t("Ranked ", "Puesto ")
                + "%{z:.0f}"
                + t(" among this opportunity's provinces",
                    " entre las provincias de esta oportunidad")
                + "<extra></extra>"
            ),
            hoverongaps=False,
            xgap=1,
            ygap=1,
        )
    )
    _add_highlight_bands(fig, product_matrix, highlight_rows, highlight_columns)
    fig.update_layout(
        height=max(520, 20 * len(product_matrix.index) + 170),
        margin=dict(l=6, r=6, t=30, b=10),
        xaxis=dict(
            # type="category" on both axes: province abbreviations and HS codes
            # both look numeric to plotly, and a numeric axis collapses the cells.
            type="category", side="top", tickangle=45,
            tickfont=dict(size=10), automargin=True, title=None,
        ),
        yaxis=dict(
            type="category", autorange="reversed",
            categoryorder="array", categoryarray=rows,
            tickfont=dict(size=10), automargin=True, title=None,
        ),
    )
    return fig


def make_candidate_heatmap(
    state_matrix: pd.DataFrame,
    state_colors: dict[str, str],
    state_labels: dict[str, str],
    row_labels: dict[str, str],
    column_labels: dict[str, str],
    column_names: dict[str, str],
    breadth: dict[str, int] | None = None,
    rank_lookup: dict[tuple[str, str], int] | None = None,
    column_groups: list[tuple[str, int]] | None = None,
) -> go.Figure:
    """Candidate industries × provinces, each cell colored by its **presence state**.

    ``state_matrix`` is industries (rows) × province ``id_code`` (columns) carrying the state rank
    — 2 specialized, 1 present-not-specialized, 0 absent — and **NaN where the industry is not on
    that province's shortlist**. Row order is the caller's (the page sorts by breadth), so this
    only renders.

    **Why state and not a count.** The top-down sibling's cells count opportunities and can exceed
    1; here an industry has exactly one state per province, so a count would be a binary mark and
    the color channel would be wasted. Coloring by state instead makes the cross-province finding
    legible at a glance: an industry whose row changes color is one a province could *enter* while
    its neighbour could *deepen*.

    Three discrete colors on a continuous scale: ``zmin/zmax`` are set half a step outside 0-2 and
    the colorscale is stepped at the thirds, so each state gets a flat band rather than a gradient
    a reader would try to interpolate. The colorbar is relabelled with the state names — a
    ``go.Heatmap`` cannot carry a legend, so the colorbar is the only place the encoding can be
    stated.

    Empty cells are left as gaps (``hoverongaps=False``): an industry that is not a candidate in a
    province is not a zero, and a hover offering "no data" on 80% of the grid would bury the marks.

    ``column_groups`` is ``[(label, n_columns), …]`` in column order — the province regions. It
    draws a separator between blocks and names each one under the grid. Without it the region
    ordering is real but invisible, and a caption claiming the columns are grouped by region while
    nothing marks the boundaries is worse than not grouping them: a reader cannot tell a Costa-wide
    row from a scattered one, which is the geographic reading the ordering exists to enable.
    """
    z = state_matrix.to_numpy(dtype=float)
    rows = [row_labels.get(code, code) for code in state_matrix.index]
    columns = [column_labels.get(code, code) for code in state_matrix.columns]

    breadth = breadth or {}
    rank_lookup = rank_lookup or {}
    order = ["absent", "present_not_spec", "specialized"]
    labels = np.array([
        [str(state_labels.get(order[int(value)], "")) if pd.notna(value) else "" for value in row]
        for row in z
    ])
    # The province column carries how many industries it contributes, and the cell its rank inside
    # its own state there — the two questions a reader asks of a mark they can see.
    ranks = np.array([
        [
            (
                "<br>#"
                + str(rank_lookup[(code, province)])
                + t(" of its state in this province",
                    " de su estado en esta provincia")
                if (code, province) in rank_lookup
                else ""
            )
            for province in state_matrix.columns
        ]
        for code in state_matrix.index
    ])
    breadths = np.array([
        [
            t(
                f"<br>A candidate in {int(breadth.get(code, 0))} of "
                f"{len(state_matrix.columns)} provinces",
                f"<br>Candidata en {int(breadth.get(code, 0))} de "
                f"{len(state_matrix.columns)} provincias",
            )
            for _ in state_matrix.columns
        ]
        for code in state_matrix.index
    ])
    names = np.array([[column_names.get(c, c) for c in state_matrix.columns]] * len(state_matrix.index))
    customdata = np.dstack([labels, ranks, breadths, names])

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=columns,
            y=rows,
            zmin=-0.5,
            zmax=2.5,
            colorscale=[
                [0.0, state_colors["absent"]], [1 / 3, state_colors["absent"]],
                [1 / 3, state_colors["present_not_spec"]], [2 / 3, state_colors["present_not_spec"]],
                [2 / 3, state_colors["specialized"]], [1.0, state_colors["specialized"]],
            ],
            colorbar=dict(
                thickness=14,
                tickmode="array",
                tickvals=[0, 1, 2],
                ticktext=[str(state_labels.get(state, state)).split(" (")[0] for state in order],
                tickfont=dict(size=10),
                title=None,
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>%{customdata[3]}<br>"
                "%{customdata[0]}%{customdata[1]}%{customdata[2]}<extra></extra>"
            ),
            hoverongaps=False,
            xgap=1,
            ygap=1,
        )
    )
    fig.update_layout(
        height=max(520, 18 * len(state_matrix.index) + 170),
        margin=dict(l=6, r=6, t=30, b=10),
        # type="category" on both axes: province abbreviations and prefixed industry codes both
        # look numeric to plotly, and a numeric axis collapses the cells.
        xaxis=dict(
            type="category", side="top", tickangle=45,
            tickfont=dict(size=10), automargin=True, title=None,
            showline=False, zeroline=False, mirror=False,
        ),
        yaxis=dict(
            type="category", autorange="reversed",
            categoryorder="array", categoryarray=rows,
            tickfont=dict(size=10), automargin=True, title=None,
        ),
        plot_bgcolor=COLOR_BADGE_BG,  # so an empty cell reads as empty, not as white data
    )
    # Region blocks: a rule on each internal boundary, and the region named beneath its block.
    # The x-axis sits on top, so the space under the grid is free for the labels.
    if column_groups:
        edge = 0
        for label, width in column_groups:
            if edge:
                fig.add_vline(
                    x=edge - 0.5, line_width=1.5, line_color=COLOR_TEXT_STRONG, opacity=0.55,
                )
            # `yshift` in **pixels**, not a paper-fraction offset: the figure's height
            # varies with the row count (520 to 2,000+), so a fractional gap that clears
            # the plot background's bottom edge on a tall grid straddles it on a short
            # one — which is what put the region names on top of that edge and made them
            # unreadable. A pixel shift is height-independent, and the bottom margin
            # below reserves room for it.
            fig.add_annotation(
                x=edge + width / 2 - 0.5, xref="x", y=0, yref="paper", yanchor="top",
                yshift=-9,
                text=label, showarrow=False, font=dict(size=10, color=COLOR_TEXT_MUTED),
            )
            edge += width
        fig.update_layout(margin=dict(l=6, r=6, t=30, b=46))
    return fig


def make_assignment_heatmap(
    matrix: pd.DataFrame,
    row_labels: dict[str, str],
    column_labels: dict[str, str],
    column_names: dict[str, str],
    highlight_rows: list[str] | None = None,
    highlight_columns: list[str] | None = None,
    cell_products: dict[tuple[str, str], list[str]] | None = None,
    product_labels: dict[str, str] | None = None,
    province_totals: dict[str, int] | None = None,
    n_products: int = 50,
) -> go.Figure:
    """Industry × province **counts** of top-k assignments over all 50 products.

    ``matrix`` is industries (rows) × province ``id_code`` (columns) of integer
    counts. A cell above 1 means the industry is a top pick there for several
    different products; a blank column means no product's ranking ever reaches
    that province — the point of the figure, not a rendering gap.

    ``highlight_rows`` / ``highlight_columns`` band one product's industries and
    its top-k provinces and outline their intersection: the cells that product
    actually contributes.

    ``cell_products`` names the opportunities behind each count **in the hover**.
    That is deliberate rather than click-driven: ``go.Heatmap`` is not a
    selectable trace — of plotly.js's trace modules only a minority implement
    ``selectPoints`` and heatmap is not among them — so ``on_select`` on this
    figure returns an empty selection no matter what. The hover carries no widget
    state, so it also survives the figure being rebuilt on every parameter
    change, which a click-based reveal would not (the plotly_chart widget id
    hashes the whole figure spec). Measured on the real matrix, the busiest cell
    serves 6 opportunities and 63% of non-empty cells serve exactly one, so the
    worst hover is six short lines. The caller supplies already-trimmed labels;
    ``_HOVER_LABEL_CHARS`` is only a backstop.
    """
    z = matrix.to_numpy(dtype=float)
    rows = [row_labels.get(code, code) for code in matrix.index]
    columns = [column_labels.get(code, code) for code in matrix.columns]
    text = np.where(z > 0, np.char.mod("%d", matrix.to_numpy(dtype=int)), "")
    # The province line carries its opportunity total, so hovering anywhere in a
    # column answers "how many of the 50 does this province win?" — which the
    # column total of *this* grid does not (it counts industry marks).
    totals = province_totals or {}
    full_names = np.array([
        [
            (
                t(
                    f"{column_names.get(c, c)} — top pick for {int(totals[c])} of "
                    f"the {n_products} opportunities",
                    f"{column_names.get(c, c)}: elegida para {int(totals[c])} de "
                    f"las {n_products} oportunidades",
                )
                if c in totals
                else column_names.get(c, c)
            )
            for c in matrix.columns
        ]
    ] * len(matrix.index))

    cell_products = cell_products or {}
    product_labels = product_labels or {}
    listings = np.array([
        [
            (
                ":<br>" + "<br>".join(
                    "  • " + str(product_labels.get(hs4, hs4))[:_HOVER_LABEL_CHARS]
                    for hs4 in sorted(cell_products.get((code, province), []))
                )
                if cell_products.get((code, province))
                else ""
            )
            for province in matrix.columns
        ]
        for code in matrix.index
    ])

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=columns,
            y=rows,
            colorscale=ASSIGNMENT_COLORSCALE,
            zmin=0,
            zmax=float(z.max()) if z.size and z.max() > 0 else 1.0,
            colorbar=dict(title=t("Products", "Productos"), thickness=14, dtick=1),
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10),
            customdata=np.dstack([full_names, listings]),
            hovertemplate=(
                "<b>%{y}</b><br>%{customdata[0]}<br>"
                + t("Top pick for ", "Elegida para ")
                + "%{z:.0f}"
                + t(" of the 50 opportunities", " de las 50 oportunidades")
                + "%{customdata[1]}<extra></extra>"
            ),
            xgap=1,
            ygap=1,
        )
    )

    _add_highlight_bands(fig, matrix, highlight_rows, highlight_columns)

    fig.update_layout(
        height=max(520, 20 * len(matrix.index) + 170),
        margin=dict(l=6, r=6, t=30, b=10),
        xaxis=dict(
            # type="category" is load-bearing on BOTH axes: CIIU codes look
            # numeric to plotly, and a numeric axis silently collapses the cells
            # (the bottom-up build lost a whole figure to exactly this).
            type="category", side="top", tickangle=45,
            tickfont=dict(size=10), automargin=True, title=None,
        ),
        yaxis=dict(
            type="category", autorange="reversed",
            categoryorder="array", categoryarray=rows,
            tickfont=dict(size=10), automargin=True, title=None,
        ),
    )
    return fig


def make_candidates_bar(
    counts: pd.DataFrame,
    lens_colors: dict[str, str],
    lens_labels: dict[str, str],
    floor: int,
) -> go.Figure:
    """How many candidate industries each province has, split by which lens found them.

    ``counts`` is one row per province with ``id_code``, ``id_name``, ``region``,
    ``top_down``, ``bottom_up`` and ``total``. Horizontal stacked bars, one per
    province, **sorted by total descending, then by the top-down count descending** —
    size is the first reading a reader takes from this chart, and grouping the bars by
    region instead would make the size pattern hard to see in order to make a regional
    pattern easy. The region is in the hover instead, which is enough for a pattern that
    is real but secondary (Sierra averages 17.7 industries per province; Amazonía and
    Galápagos sit at exactly the floor).

    **The secondary key is what stops the colours interleaving.** Seventeen of the 24
    provinces are tied at the floor, so without it the two segments alternate down the
    block for no reason a reader can see. Ordering the tie by top-down count turns it
    into one monotone wedge — the provinces the product analysis reached most at the top,
    the two it never reached at all (all-red bars) at the very bottom, which is the
    chart's whole argument. ``id_code`` breaks any remaining tie so the order is stable
    between runs.

    The two segments are not two halves of one measure — they are two **provenances**.
    The top-down segment is the assignment from the 50 HS4 opportunities, the stronger
    evidentiary base; the bottom-up segment is the top-up that fills a province to
    ``floor`` when the assignment could not. A province at the floor with an all-red bar
    is one the product analysis never reached, and that is the finding the chart exists
    to make visible in one glance.

    The dashed reference line marks ``floor``: every bar reaches it by construction, so
    it reads as the level the list guarantees rather than as a target some provinces
    miss.
    """
    # Plotly draws a horizontal category axis bottom-up, so the sort is inverted here:
    # ascending on total and top_down puts the largest, most top-down province at the top.
    frame = counts.sort_values(
        ["total", "top_down", "id_code"], ascending=[True, True, False], kind="mergesort"
    )
    labels = [str(name).strip() for name in frame["id_name"]]

    fig = go.Figure()
    for lens in ("top_down", "bottom_up"):
        fig.add_trace(
            go.Bar(
                x=frame[lens],
                y=labels,
                orientation="h",
                name=lens_labels.get(lens, lens),
                marker=dict(color=lens_colors.get(lens, COLOR_OTHER_BAR)),
                customdata=np.column_stack([
                    frame["region"].astype(str),
                    frame["top_down"].astype(int),
                    frame["bottom_up"].astype(int),
                    frame["total"].astype(int),
                ]),
                hovertemplate=(
                    "<b>%{y}</b> · %{customdata[0]}<br>"
                    + t("From an opportunity: ", "Desde una oportunidad: ")
                    + "%{customdata[1]}<br>"
                    + t("Topped up from the province ranking: ",
                        "Complemento del ordenamiento provincial: ")
                    + "%{customdata[2]}<br>"
                    + t("Total: ", "Total: ")
                    + "%{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=float(floor), line=dict(color=COLOR_TEXT_STRONG, width=1, dash="dash"),
    )
    fig.add_annotation(
        x=float(floor), y=1.0, yref="paper", yanchor="bottom",
        text=t(f"floor of {int(floor)}", f"piso de {int(floor)}"), showarrow=False,
        font=dict(size=11, color=COLOR_TEXT_MUTED),
    )
    fig.update_layout(
        barmode="stack",
        height=max(520, 22 * len(frame) + 150),
        margin=dict(l=6, r=6, t=52, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            title=None, font=dict(size=11),
        ),
        xaxis=dict(
            title=t("Candidate industries", "Industrias candidatas"), zeroline=False,
            gridcolor=COLOR_OTHER_LINE, tickfont=dict(size=11),
        ),
        yaxis=dict(type="category", title=None, tickfont=dict(size=11), automargin=True),
        bargap=0.25,
    )
    return fig
