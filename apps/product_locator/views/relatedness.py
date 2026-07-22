"""Page 4: related industries & opportunities (economic-complexity view).

Consumes the precomputed exports of ``notebooks/relatedness_density.ipynb`` from
``data/processed/summary_tables/relatedness_density/`` — proximity (industry
space) and per-province relatedness density — and reads off concrete
opportunities at the intensive and extensive margin. It never recomputes
proximity/density from microdata; refreshing the analysis means re-running the
notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import state
from config import (
    COLOR_PRESENT_OUTLINE,
    DENSITY_COLORSCALE,
    STATE_COLORS,
    STATE_LABELS,
    THIN_NEIGHBORHOOD_THRESHOLD,
)
from data_loader import (
    load_ciiu_reference,
    load_density_support,
    load_isic_descriptions,
    load_local_relevance,
    load_province_density,
    load_province_geodata,
    load_province_geojson,
    load_proximity_matrix,
    load_tradability,
    load_tradability_pctile,
)
from metrics import (
    aggregate_product_density,
    build_opportunity_table,
    density_contributions,
    top_neighbors,
)
from viz import (
    fmt_number,
    make_density_presence_scatter,
    make_feasibility_ranking_bar,
    make_opportunity_bars,
    make_province_tile_choropleth,
    make_proximity_heatmap,
)

TRADABILITY_CHOICES = {"Atlas": "atlas", "Regional": "regional", "None": None}

# ISIC Rev.4 / CIIU 4.1 division (first 2 digits) -> section letter, so class
# codes can be shown section-prefixed (e.g. "A0311") as elsewhere in the app.
_SECTION_RANGES = [
    ("A", 1, 3), ("B", 5, 9), ("C", 10, 33), ("D", 35, 35), ("E", 36, 39),
    ("F", 41, 43), ("G", 45, 47), ("H", 49, 53), ("I", 55, 56), ("J", 58, 63),
    ("K", 64, 66), ("L", 68, 68), ("M", 69, 75), ("N", 77, 82), ("O", 84, 84),
    ("P", 85, 85), ("Q", 86, 88), ("R", 90, 93), ("S", 94, 96), ("T", 97, 98),
    ("U", 99, 99),
]


def _section_letter(code: str) -> str:
    try:
        division = int(str(code)[:2])
    except (ValueError, TypeError):
        return ""
    for letter, low, high in _SECTION_RANGES:
        if low <= division <= high:
            return letter
    return ""


def _label_lookup() -> dict[str, str]:
    """4-digit class -> Spanish-preferred display name (matches the rest of the app)."""
    isic_desc = load_isic_descriptions()
    ciiu_ref = load_ciiu_reference()
    labels = dict(zip(isic_desc["isic_code"], isic_desc["isic_name"]))
    labels.update(ciiu_ref["class_name_lookup"])  # prefer Spanish CIIU names
    return labels


# Self / no-self density toggle. Default = "Include own presence" (with-self).
_SELF_ON, _SELF_OFF = "Include own presence", "Related only"
_HELP_SECTION_AB = (
    "Include own presence — each industry's score includes its own presence: "
    "where can this product be made today? · Related only — own presence excluded: "
    "where do the related capabilities exist, even if the industry itself is absent?"
)
_HELP_SECTION_B = (
    "Include own presence — an industry's score includes its own presence in this province. "
    "Related only — the score counts only the other related industries present here "
    "(own presence excluded)."
)


def _selfterm_toggle(section_key: str, help_text: str) -> bool:
    """Render the self/no-self control; return True when the with-self view is on."""
    choice = st.segmented_control(
        "Density view",
        [_SELF_ON, _SELF_OFF],
        default=_SELF_ON,
        key=f"rel-self-{section_key}",
        help=help_text,
    )
    return (choice or _SELF_ON) == _SELF_ON


def render() -> None:
    state.init_state()
    state.require_selection()

    st.title("Related industries & opportunities")
    st.markdown(
        "Firms tend to co-produce **related** industries. Measuring how often industries "
        "are produced together yields a **proximity** between them, and — per province — a "
        "**relatedness density**: how much of an industry's related capabilities are already "
        "present locally. High density flags where an industry is plausibly *feasible* to "
        "promote. This generalizes Page 3's headquarters-reach intuition into a measured, "
        "economy-wide view, and reads off opportunities at the **intensive** margin (deepen "
        "what is present) and the **extensive** margin (new-entry candidates)."
    )

    labels = _label_lookup()
    matrix = load_proximity_matrix()
    density_df = load_province_density()
    tradability_df = load_tradability()
    relevance_df = load_local_relevance()
    pctile_df = load_tradability_pctile()
    support_df = load_density_support()
    support_ns = support_df.set_index("codigo_clase")["phi_mass_noself"]
    geo = load_province_geodata()
    prov_lookup = (
        geo[["id_code", "id_abbr", "id_name", "region"]]
        .dropna(subset=["id_code"])
        .drop_duplicates("id_code")
        .set_index("id_code")
    )

    # Curated classes that actually exist in the relatedness exports.
    density_classes = set(density_df["codigo_clase"].unique())
    matrix_classes = set(matrix.index)
    section_lookup = {code: _section_letter(code) for code in density_classes}
    valid_codes = [
        code
        for code in st.session_state["selected_isic"]
        if code in density_classes and code in matrix_classes
    ]
    dropped = [c for c in st.session_state["selected_isic"] if c not in valid_codes]

    if not valid_codes:
        st.warning(
            "None of the selected industries appear in the relatedness analysis "
            "(they may be unclassified or absent from the business registry). "
            "Go back and select industries with registered activity."
        )
        st.stop()
    if dropped:
        st.caption(
            f"{len(dropped)} selected industr{'y' if len(dropped) == 1 else 'ies'} "
            f"({', '.join(dropped)}) not covered by the relatedness analysis and omitted here."
        )

    province_options = sorted(density_df["id_code"].unique())

    def _prov_label(code: str) -> str:
        if code in prov_lookup.index:
            row = prov_lookup.loc[code]
            return f"{row['id_name']} ({row['region']})"
        return code

    def _view_word(include_self: bool) -> str:
        return "own presence included" if include_self else "related capabilities only (own presence excluded)"

    # ------------------------------------------------------------------ #
    # Section 0 — product feasibility ranking (headline).                 #
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader(f"Where is this product (HS {st.session_state['hs_code']}) most feasible?")
    head_cols = st.columns([3, 2])
    with head_cols[1]:
        self0 = _selfterm_toggle("s0", _HELP_SECTION_AB)
    value_col0 = "density_cont" if self0 else "density_cont_noself"
    st.caption(
        "Provinces ranked by mean relatedness density across the selected industries "
        f"(**{_view_word(self0)}**). Bar color marks each province's **best** margin across "
        "those industries — green = already specialized, amber = present but not specialized, "
        "blue = absent (a genuinely new opportunity)."
        + ("" if self0 else " In this view the aggregate mixes industries with different "
           "neighborhood sizes, so read magnitudes with care.")
    )
    agg = aggregate_product_density(density_df, valid_codes, value_col=value_col0)
    # A stable default province for Section B (with-self #1, independent of the toggle).
    agg_default = aggregate_product_density(density_df, valid_codes, value_col="density_cont")
    default_prov = agg_default.iloc[0]["id_code"] if not agg_default.empty else province_options[0]
    if agg.empty:
        st.info("No relatedness density available for the selected industries.")
    else:
        st.plotly_chart(
            make_feasibility_ranking_bar(agg, STATE_COLORS, STATE_LABELS),
            width="stretch",
            key="rel-feasibility",
        )

    # ------------------------------------------------------------------ #
    # Section A — industry view (industry-dependent).                     #
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader("Industry view")
    ind_cols = st.columns([3, 2])
    with ind_cols[0]:
        focal = st.selectbox(
            "Focal industry",
            options=valid_codes,
            format_func=lambda code: f"{code} — {labels.get(code, code)}",
            key="rel-focal-industry",
        )
    with ind_cols[1]:
        selfA = _selfterm_toggle("sa", _HELP_SECTION_AB)
    value_colA = "density_cont" if selfA else "density_cont_noself"
    focal_name = labels.get(focal, focal)
    focal_ns = float(support_ns.get(focal, float("nan")))
    focal_thin = pd.notna(focal_ns) and focal_ns < THIN_NEIGHBORHOOD_THRESHOLD

    heat_col, map_col = st.columns([1, 1])
    with heat_col:
        st.markdown("**Proximity to the closest industries**")
        neighbors = top_neighbors(matrix, focal, k=12)
        if not neighbors:
            st.info("No proximity neighbors available for this industry.")
        else:
            st.plotly_chart(
                make_proximity_heatmap(
                    matrix, focal, neighbors, labels,
                    colorscale="Blues", section_lookup=section_lookup,
                ),
                width="stretch",
                key=f"rel-heatmap-{focal}",
            )
            st.caption(
                "Symmetric proximity φ ∈ [0, 1]: the share of *diversifying* firms (those active "
                "in more than one industry) producing both industries, relative to the more "
                "common of the two. The diagonal (self = 1) is masked."
            )

    focal_density = density_df[density_df["codigo_clase"] == focal].copy()
    with map_col:
        st.markdown("**Relatedness density by province**")
        plot_df = focal_density.merge(
            prov_lookup[["id_abbr"]], left_on="id_code", right_index=True, how="left"
        )
        plot_df["density_display"] = plot_df[value_colA].map(lambda v: fmt_number(v, 3))
        plot_df["rca_display"] = plot_df["rca"].map(lambda v: fmt_number(v, 2))
        plot_df["intensity_display"] = plot_df["intensity"].map(lambda v: fmt_number(v, 0))
        # Presence reflects the RCA-defined state, not just M: present-but-not-
        # specialized (0 < RCA < 1) must not be labeled "Absent".
        _presence = {
            "specialized": "Present (RCA ≥ 1)",
            "present_not_spec": "Present (RCA < 1)",
            "absent": "Absent",
        }
        plot_df["presence_txt"] = plot_df["state"].map(_presence).fillna("—")
        hover_labels = {
            "id_name": "Province",
            "density_display": "Relatedness density",
            "rca_display": "RCA",
            "presence_txt": "Presence",
            "intensity_display": "Establishments",
        }
        outline_ids = focal_density.loc[focal_density["M"] == 1, "id_code"].tolist()
        st.plotly_chart(
            make_province_tile_choropleth(
                province_geojson=load_province_geojson(),
                plot_df=plot_df,
                value_column=value_colA,
                colorbar_title="Density",
                color_scale=DENSITY_COLORSCALE,
                hover_columns=list(hover_labels.keys()),
                hover_labels=hover_labels,
                locations_column="id_code",
                outline_ids=outline_ids,
                outline_color=COLOR_PRESENT_OUTLINE,
            ),
            width="stretch",
            key=f"rel-choropleth-{focal}-{value_colA}",
        )
        if selfA:
            st.caption(
                "Darker = more of this industry's related capabilities are already present "
                "(**own presence included**). Provinces outlined in red already have the industry "
                "(specialized, RCA ≥ 1), so their high density is not a *new* opportunity."
            )
        else:
            thin_note = (
                f" ⚠ This industry has a **thin neighborhood** (off-diagonal φ mass "
                f"{fmt_number(focal_ns, 2)} < {THIN_NEIGHBORHOOD_THRESHOLD}), so the no-self "
                "scores are unstable — read with caution." if focal_thin else ""
            )
            st.caption(
                "**Related capabilities only** (own presence excluded): darker = a deeper local "
                "ecosystem of related industries, whether or not the industry itself is present. "
                "The red outline still flags provinces that already have it (RCA ≥ 1). This is a "
                "*different metric* from the with-self view — not comparable in magnitude." + thin_note
            )

    st.markdown("**Already there vs. fertile ground**")
    scatter_df = focal_density.merge(
        prov_lookup[["id_abbr"]], left_on="id_code", right_index=True, how="left"
    )
    scatter_df["value"] = scatter_df[value_colA]
    st.plotly_chart(
        make_density_presence_scatter(
            scatter_df, STATE_COLORS, STATE_LABELS, show_quadrants=not selfA
        ),
        width="stretch",
        key=f"rel-scatter-{focal}-{value_colA}",
    )
    if selfA:
        st.caption(
            "Each dot is a province (y = density with **own presence included**). Provinces high "
            "on the vertical axis but **left** of the RCA = 1 line are the strongest new-entry "
            "candidates. Because the self-term lifts present provinces, the fertile-ground corner "
            "is usually sparse — switch the density view to *Related only* to populate it."
        )
    else:
        st.caption(
            "y = density with **own presence excluded**. Now the vertical axis measures the local "
            "*related* ecosystem alone, so provinces upper-**left** of the RCA = 1 line — high "
            "related capabilities, not yet specialized — are genuine new-entry candidates."
        )

    # Decomposition — moved to the bottom of Section A (industry-level), follows Section A's toggle.
    dprov = st.session_state.get("rel-province")
    if dprov not in province_options:
        dprov = default_prov
    dprov_name = prov_lookup.loc[dprov, "id_name"] if dprov in prov_lookup.index else dprov
    focal_here = focal_density[focal_density["id_code"] == dprov]
    density_here = float(focal_here[value_colA].iloc[0]) if not focal_here.empty else float("nan")
    with st.expander(
        f"What's behind the density score? ({focal_name} in {dprov_name}: "
        f"{fmt_number(density_here, 3)}, {_view_word(selfA)})"
    ):
        prov_density_all = density_df[density_df["id_code"] == dprov]
        contrib = density_contributions(
            matrix.loc[focal], prov_density_all, focal=focal, top_n=10, include_self=selfA
        )
        contrib = contrib[contrib["contribution"] > 0].copy()
        if contrib.empty:
            st.info("No contributing related industries with local presence.")
        else:
            def _contrib_label(c: str) -> str:
                base = f"{section_lookup.get(c, '')}{c} — {labels.get(c, c)}"
                return base + " *(this industry)*" if c == focal else base

            contrib["Industry"] = contrib["codigo_clase"].map(_contrib_label)
            display = contrib.rename(
                columns={
                    "phi": "φ to focal",
                    "rca": "Province RCA",
                    "contribution_share": "Share of density",
                }
            )[["Industry", "φ to focal", "Province RCA", "Share of density"]]
            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
                column_config={
                    "φ to focal": st.column_config.NumberColumn(format="%.3f"),
                    "Province RCA": st.column_config.NumberColumn(format="%.2f"),
                    "Share of density": st.column_config.NumberColumn(format="percent"),
                },
            )
            if selfA:
                st.caption(
                    f"The top contributors to {dprov_name}'s density for {focal_name}: each is "
                    "φ (proximity to the focal industry) × its local presence, as a share of the "
                    "numerator (they sum to ≤ 100%). The focal industry's own presence (the "
                    "self-term) is usually the largest — switch to *Related only* to see the "
                    "related-industry contributions renormalized on their own."
                )
            else:
                st.caption(
                    f"Related-industry contributors to {dprov_name}'s no-self density for "
                    f"{focal_name} — the focal industry's own presence is excluded and the shares "
                    "are renormalized over the related industries (sum ≤ 100%). Province follows "
                    "the selector in Section B below."
                )

    # ------------------------------------------------------------------ #
    # Section B — province opportunities (province-dependent).            #
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader("Province opportunities")
    st.caption(
        "This section is **product-agnostic** — the rankings describe the province's own economy "
        "(identical whichever product you arrived with; your product's industries are only "
        "highlighted). It answers the complementary question: *given the capabilities already in "
        "this province, what could it deepen or diversify into?*"
    )

    sel_cols = st.columns([2, 2, 1, 1])
    with sel_cols[0]:
        sel_province = st.selectbox(
            "Province",
            options=province_options,
            index=province_options.index(default_prov) if default_prov in province_options else 0,
            format_func=_prov_label,
            key="rel-province",
        )
    st.session_state["selected_province"] = sel_province
    with sel_cols[1]:
        trad_choice = st.segmented_control(
            "Tradability filter",
            list(TRADABILITY_CHOICES.keys()),
            default="Atlas",
            key="rel-tradability",
        )
    with sel_cols[2]:
        top_k = st.number_input(
            "Bars per state",
            min_value=3,
            max_value=40,
            value=15,
            step=1,
            key="rel-topk",
            help="Maximum industries shown per state; fewer appear when not that many exist.",
        )
    with sel_cols[3]:
        selfB = _selfterm_toggle("sb", _HELP_SECTION_B)
    value_colB = "density_cont" if selfB else "density_cont_noself"
    defn = TRADABILITY_CHOICES.get(trad_choice or "Atlas", "atlas")
    prov_name = prov_lookup.loc[sel_province, "id_name"] if sel_province in prov_lookup.index else sel_province

    opp = build_opportunity_table(
        density_df, tradability_df, sel_province, valid_codes,
        defn=defn, top_n=int(top_k), value_col=value_colB,
    )
    if opp.empty:
        st.info("No industries to rank for this province under this filter.")
        return

    # Local-relevance decoration + industry-space support for the thin-neighborhood flag.
    prov_rel = relevance_df[relevance_df["id_code"] == sel_province][
        ["codigo_clase", "mean_relevance", "rank_in_province"]
    ]
    opp = opp.merge(prov_rel, on="codigo_clase", how="left")
    opp = opp.merge(pctile_df, on="codigo_clase", how="left")
    opp = opp.merge(support_df[["codigo_clase", "phi_mass_noself"]], on="codigo_clase", how="left")
    opp["name"] = opp["codigo_clase"].map(lambda c: labels.get(c, c))
    opp["value"] = opp[value_colB]

    def _is_thin(r: pd.Series) -> bool:
        return (not selfB) and pd.notna(r["phi_mass_noself"]) and r["phi_mass_noself"] < THIN_NEIGHBORHOOD_THRESHOLD

    def _row_label(r: pd.Series) -> str:
        code = f"{section_lookup.get(r['codigo_clase'], '')}{r['codigo_clase']}"
        star = "★ " if pd.notna(r["rank_in_province"]) and r["rank_in_province"] <= 10 else ""
        warn = "⚠ " if _is_thin(r) else ""
        return f"{warn}{star}{code}"

    def _hover_extra(r: pd.Series) -> str:
        parts = []
        if pd.notna(r["mean_relevance"]):
            rank = r["rank_in_province"]
            rank_txt = f" · local rank {int(rank)}" if pd.notna(rank) else ""
            parts.append(f"Local relevance: {r['mean_relevance']:.2f}{rank_txt}")
        if r["state"] == "absent" and pd.notna(r["tradability_pctile"]):
            parts.append(f"National tradability: {r['tradability_pctile']:.0%} pctile")
        if pd.notna(r["phi_mass_noself"]):
            parts.append(f"Neighborhood mass (φ, no-self): {r['phi_mass_noself']:.2f}")
        if _is_thin(r):
            parts.append("⚠ thin neighborhood — no-self score unstable")
        return ("<br>" + "<br>".join(parts)) if parts else ""

    opp["label"] = opp.apply(_row_label, axis=1)
    opp["hover_extra"] = opp.apply(_hover_extra, axis=1)

    filter_txt = "no tradability filter" if defn is None else f"{trad_choice} tradability"
    st.caption(
        f"Industries in {prov_name} grouped by RCA state and ranked by relatedness density "
        f"(**{_view_word(selfB)}**; {filter_txt}; up to {int(top_k)} per state). Labels are class "
        "codes — hover for the name and details; ★ marks the province's 10 most locally relevant "
        "industries; ⚠ marks thin-neighborhood classes whose *no-self* score is unstable. Bars "
        "outlined in red are the product's own industries."
    )

    STATES = [
        ("specialized", "Specialized · core (RCA ≥ 1)"),
        ("present_not_spec", "Present, not specialized (RCA < 1)"),
        ("absent", "Absent · new entry (RCA = 0)"),
    ]
    bar_cols = st.columns(3)
    for (state_key, header), col in zip(STATES, bar_cols):
        with col:
            st.markdown(f"**{header}**")
            sub = opp[opp["state"] == state_key]
            if sub.empty:
                st.info("None under this filter.")
                continue
            st.plotly_chart(
                make_opportunity_bars(
                    sub, STATE_COLORS, show_establishments=(state_key != "absent")
                ),
                width="stretch",
                key=f"rel-opp-{state_key}-{sel_province}-{trad_choice}-{value_colB}",
            )

    if selfB:
        st.caption(
            "Hover shows the industry name plus — for **present** industries — its local relevance "
            "(geographic concentration) and, for **absent** industries, its national tradability "
            "percentile. With **own presence included**, a *specialized* industry can score a high "
            "density simply because it is already strongly present (the self-term), even with few "
            "related neighbors — switch the density view to *Related only* to rank by the related "
            "ecosystem alone. This view and the no-self view are different metrics, not comparable "
            "in magnitude."
        )
    else:
        st.caption(
            "**Own presence excluded**: bars now rank industries by their local *related* "
            "ecosystem alone, so specialized-but-isolated industries no longer float to the top. "
            "The trade-off is instability for thin-neighborhood classes (⚠) — their denominator is "
            "near zero, so treat those scores as noise. Hover shows local relevance / national "
            "tradability and each class's off-diagonal φ mass."
        )
