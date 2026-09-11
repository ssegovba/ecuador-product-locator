"""Session-state schema for the two lenses.

The top-down lens is **two pages that share one parameter set**: whatever the
user picks on Page A (view, base, support guard, route preset, assignment bar) is what Page B
replays for all 50 opportunities. Streamlit drops a widget's state when a run
does not render that widget — and Page A hides the route control on
intensive-margin opportunities — so every shared parameter is re-asserted here on
each run. Re-assigning a key to itself marks it as user-set and is what keeps it
alive across page changes.

The bottom-up page is independent: it calls :func:`init_state` and nothing else,
and never reads or writes any of these keys.
"""

from __future__ import annotations

import streamlit as st

from config import DEFAULT_PRESENCE_CUT, DEFAULT_ROUTE_PRESET

# Canonical (non-widget) selection.
_SELECTION: dict[str, object] = {
    "opportunity_hs4": None,
}

# Shared top-down parameters. These ARE the widget keys — both pages bind their
# controls to them, which is what keeps the two in sync without a copy step.
TD_PARAM_DEFAULTS: dict[str, object] = {
    "td-view": "lq",  # location quotient; "scale" = share of the national total
    "td-base": "plazas",  # employment; "estab" = establishments
    "td-support": True,  # support guard applied
    "td-route": DEFAULT_ROUTE_PRESET,
    # The assignment bar, as a multiple of the presence measure's parity point.
    "td-cut": DEFAULT_PRESENCE_CUT,
}


def init_state() -> None:
    for key, default in {**_SELECTION, **TD_PARAM_DEFAULTS}.items():
        st.session_state.setdefault(key, default)
    # Keep the shared widget keys alive across page changes: Streamlit garbage-
    # collects widget state that a run did not render, and Page A renders the
    # route control only for extensive-margin opportunities.
    for key in TD_PARAM_DEFAULTS:
        st.session_state[key] = st.session_state[key]


def clear_opportunity() -> None:
    """Drop the selected opportunity; the parameters deliberately survive."""
    st.session_state["opportunity_hs4"] = None
