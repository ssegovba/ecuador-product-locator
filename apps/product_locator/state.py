"""Session-state schema and guards for the three-stage wizard.

Widget state is dropped by Streamlit when a page does not render the widget,
so every selection that must survive page changes lives in the canonical
(non-widget) keys defined here.
"""

from __future__ import annotations

import streamlit as st

from config import DEFAULT_EMPLOYMENT_BASE, DEFAULT_HS_EDITION

_DEFAULTS: dict[str, object] = {
    "hs_edition": DEFAULT_HS_EDITION,
    "hs_code": None,
    "hs_description": None,
    # list of dicts: codigo_clase, isic_name, description, match_weight, in_reem
    "matched_isic": [],
    "selected_isic": [],
    "employment_base": DEFAULT_EMPLOYMENT_BASE,
    "selection_complete": False,
    "selected_province": None,
}


def init_state() -> None:
    for key, default in _DEFAULTS.items():
        st.session_state.setdefault(key, default)


def reset_selection() -> None:
    """Clear downstream results when the HS edition or code changes."""
    st.session_state["matched_isic"] = []
    st.session_state["selected_isic"] = []
    st.session_state["selection_complete"] = False
    st.session_state["selected_province"] = None


def require_selection() -> None:
    """Stop rendering Pages 2-4 until a product selection has been completed."""
    if not st.session_state.get("selection_complete"):
        import nav  # deferred: nav imports the view modules, which import this module

        st.info("Select a product and its industries first.")
        st.page_link(nav.PAGE_SELECT, label="Go to product selection", icon="🔎")
        st.stop()
