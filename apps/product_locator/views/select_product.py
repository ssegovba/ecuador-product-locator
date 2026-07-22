"""Page 1: select an HS product and curate its candidate ISIC industries."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import state
from config import HS_EDITIONS
from data_loader import (
    load_ciiu_reference,
    load_ec_trade,
    load_hs_isic_crosswalk,
    load_hs_reference,
    load_isic_descriptions,
    load_reem_data,
)
from resolver import resolve_isic_matches
from viz import fmt_percent


def _on_product_change() -> None:
    state.reset_selection()


def _fmt_usd(value: float) -> str:
    value = float(value)
    if value >= 1e9:
        return f"${value / 1e9:.1f}B"
    if value >= 1e6:
        return f"${value / 1e6:.1f}M"
    if value >= 1e3:
        return f"${value / 1e3:.0f}k"
    return f"${value:,.0f}"


def _render_trade_context(
    isic_code: str,
    hs_code: str,
    n_hs: int,
    matching_codes,
    export_by_code: pd.Series,
    national_exports: float,
    sel_export: float,
    sel_share_national: float,
    description_lookup: dict,
    year: int | None,
) -> None:
    """Compact export-scale line + a 'Trade context' expander per industry."""
    ind = export_by_code[export_by_code.index.isin(matching_codes)]
    footprint = float(ind.sum())
    footprint_share = footprint / national_exports if national_exports else 0.0
    year_txt = f" ({year})" if year else ""

    if footprint > 0:
        line = (
            f"↳ {n_hs} HS products map here · industry exports {_fmt_usd(footprint)} "
            f"({fmt_percent(footprint_share)} of national{year_txt})"
        )
    else:
        line = f"↳ {n_hs} HS products map here · no recorded exports{year_txt}"
    st.caption(line)

    with st.expander("Trade context"):
        if sel_export > 0:
            sel_share_ind = sel_export / footprint if footprint else 0.0
            st.markdown(
                f"**Your product** HS {hs_code}: {_fmt_usd(sel_export)} exported"
                f"{year_txt} — {fmt_percent(sel_share_ind)} of this industry's exports, "
                f"{fmt_percent(sel_share_national)} of national exports."
            )
        else:
            st.markdown(
                f"**Your product** HS {hs_code} recorded no exports{year_txt} — "
                "likely produced for the domestic market rather than export."
            )

        if footprint > 0:
            top = ind.groupby(level=0).sum().sort_values(ascending=False).head(3)
            st.markdown("**Top exported products in this industry:**")
            for code, value in top.items():
                name = description_lookup.get(code, "")
                st.markdown(
                    f"- {code} — {name}: {_fmt_usd(value)} "
                    f"({fmt_percent(value / footprint)} of industry)"
                )
        else:
            st.markdown("_None of this industry's HS products were exported in the latest year._")

        st.caption(
            "Note: a product's exports can map to more than one industry, so these "
            "industry footprints may overlap and double-count across the matched industries."
        )


def render() -> None:
    st.title("Select a product")
    st.caption(
        "Choose an HS classification edition and a 6-digit product code. "
        "The app maps the product to the industries (ISIC Rev. 4 classes) that produce it; "
        "review and deselect any industry that is not relevant before continuing."
    )

    editions = list(HS_EDITIONS.keys())
    edition = st.selectbox(
        "HS classification edition",
        editions,
        index=editions.index(st.session_state["hs_edition"]),
        format_func=lambda e: f"HS {e}",
        key="_hs_edition_widget",
        on_change=_on_product_change,
    )
    st.session_state["hs_edition"] = edition

    hs_ref = load_hs_reference(edition)
    description_lookup = hs_ref.set_index("hs_code")["description"].to_dict()
    options = hs_ref["hs_code"].tolist()

    previous_code = st.session_state.get("hs_code")
    # Clear (✕) button beside the box so the user can reset the product without
    # deleting the typed text. Handled BEFORE the selectbox is instantiated, so
    # resetting the widget's session key is allowed (no "modified after created").
    code_col, clear_col = st.columns([16, 1], vertical_alignment="bottom")
    with clear_col:
        clear_clicked = st.button(
            "✕", key="_clear_hs", help="Clear the selected product",
            disabled=previous_code is None,
        )
    if clear_clicked:
        st.session_state["_hs_code_widget"] = None
        st.session_state["hs_code"] = None
        st.session_state["hs_description"] = None
        state.reset_selection()
        st.rerun()
    with code_col:
        hs_code = st.selectbox(
            "HS6 product",
            options,
            index=options.index(previous_code) if previous_code in options else None,
            format_func=lambda code: f"{code} — {description_lookup.get(code, '')}",
            placeholder="Type a code or keyword to search",
            key="_hs_code_widget",
            on_change=_on_product_change,
        )

    if hs_code is None:
        st.info("Pick a product to see its matching industries.")
        return

    st.session_state["hs_code"] = hs_code
    st.session_state["hs_description"] = description_lookup.get(hs_code, "")

    crosswalk = load_hs_isic_crosswalk(edition)
    isic_descriptions = load_isic_descriptions()
    ciiu_ref = load_ciiu_reference()
    reem_classes = set(load_reem_data()["codigo_clase"].unique())

    matches = resolve_isic_matches(
        hs_code=hs_code,
        crosswalk=crosswalk,
        isic_descriptions=isic_descriptions,
        class_name_lookup=ciiu_ref["class_name_lookup"],
        reem_classes=reem_classes,
    )

    if matches.empty:
        st.warning(
            f"HS {hs_code} (edition {edition}) has no match in the HS→ISIC concordance. "
            "Try a related product code."
        )
        return

    st.subheader(f"Matching industries ({len(matches)})")
    st.caption(
        "All industries are selected by default. Read the descriptions and deselect "
        "the ones you consider less relevant for this product."
    )

    # Trade context (Ecuador exports, latest available year, same HS edition).
    trade = load_ec_trade(edition)
    trade_year = int(trade["year"].iloc[0]) if not trade.empty else None
    national_exports = float(trade["export_value"].sum())
    export_by_code = trade.set_index("hs_code")["export_value"]
    sel_export = float(export_by_code.get(hs_code, 0.0))
    sel_share_national = sel_export / national_exports if national_exports else 0.0
    hs_per_isic = crosswalk.groupby("isic4_code")["hs_code"].nunique().to_dict()
    codes_by_isic = {
        code: crosswalk.loc[crosswalk["isic4_code"] == code, "hs_code"].unique()
        for code in matches["codigo_clase"]
    }

    # Select all / clear all — set each checkbox's state, then rerun.
    cb_keys = [f"_isic_cb_{hs_code}_{c}" for c in matches["codigo_clase"]]
    btn_cols = st.columns([1, 1, 6])
    if btn_cols[0].button("Select all", key=f"_selall_{hs_code}"):
        for k in cb_keys:
            st.session_state[k] = True
        st.rerun()
    if btn_cols[1].button("Clear all", key=f"_clrall_{hs_code}"):
        for k in cb_keys:
            st.session_state[k] = False
        st.rerun()

    previously_selected = set(st.session_state.get("selected_isic") or [])
    restore = bool(previously_selected) and st.session_state.get("selection_complete")

    checked_codes: list[str] = []
    for row in matches.itertuples():
        key = f"_isic_cb_{hs_code}_{row.codigo_clase}"
        # Seed the default via session_state (not the widget's `value=`), so the
        # Select all / Clear all buttons can set the same key without Streamlit
        # warning that a default and a Session-State value were both supplied.
        default_checked = row.codigo_clase in previously_selected if restore else True
        st.session_state.setdefault(key, default_checked)
        label = f"**{row.codigo_clase} — {row.isic_name}**"
        if not row.in_reem:
            label += " :gray-badge[no registered enterprises]"
        checked = st.checkbox(label, key=key)
        st.caption(row.description)
        _render_trade_context(
            row.codigo_clase,
            hs_code,
            hs_per_isic.get(row.codigo_clase, 0),
            codes_by_isic.get(row.codigo_clase, []),
            export_by_code,
            national_exports,
            sel_export,
            sel_share_national,
            description_lookup,
            trade_year,
        )
        st.divider()
        if checked:
            checked_codes.append(row.codigo_clase)

    n_in_reem = int(matches["in_reem"].sum())
    if n_in_reem < len(matches):
        st.info(
            f"{len(matches) - n_in_reem} of {len(matches)} matched industries have no "
            "registered enterprises in the 2024 business registry (REEM); they will not "
            "contribute to the geographic analysis."
        )

    with st.expander("Diagnostics: HS → ISIC concordance detail"):
        audit = matches.copy()
        audit.insert(0, "hs_code", hs_code)
        audit.insert(1, "hs_edition", edition)
        st.dataframe(audit, width="stretch", hide_index=True)

    if not checked_codes:
        st.warning("Select at least one industry to continue.")

    if st.button(
        "Continue to geographic analysis",
        type="primary",
        disabled=not checked_codes,
        width="stretch",
    ):
        st.session_state["matched_isic"] = matches.to_dict("records")
        st.session_state["selected_isic"] = checked_codes
        st.session_state["selection_complete"] = True

        import nav  # deferred: nav imports this module

        st.switch_page(nav.PAGE_GEOGRAPHY)
