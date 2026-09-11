from __future__ import annotations

import streamlit as st

import nav
import state
from config import (
    APP_CAPTION_EN,
    APP_CAPTION_ES,
    APP_TITLE_EN,
    APP_TITLE_ES,
    DEFAULT_LANG,
    LANG_LABELS,
    LANGS,
)
from lang import t

# Language is resolved at the very top of the run, before anything renders:
# st.set_page_config needs the browser-tab title in the right language, and
# nav.build() below needs it for the page titles.
#
# The toggle's own widget state comes **first** in that order, and that is the
# whole point. Clicking the toggle reruns the script, and on that rerun the new
# code lives in "lang-widget" while "lang" still holds the previous run's value
# (it is only written further down, at the widget). Reading "lang" first would
# leave the nav titles and the page body one click behind the toggle.
#
# A `?lang=` query parameter is the last resort before the default, so a shared
# link carries its language to whoever opens it — the app is sent around as a
# link, and the two audiences want different defaults (Spanish for Ecuadorian
# readers, English for Growth Lab). It seeds only; once a session has a
# language, the toggle owns it.
_lang = st.session_state.get("lang-widget")
if _lang not in LANGS:
    _lang = st.session_state.get("lang")
if _lang not in LANGS:
    _lang = st.query_params.get("lang")
if _lang not in LANGS:
    _lang = DEFAULT_LANG
st.session_state["lang"] = _lang

st.set_page_config(page_title=t(APP_TITLE_EN, APP_TITLE_ES), layout="wide")

state.init_state()

# Pages are rebuilt on every run so their titles follow the current language;
# nav.build() also rebinds the module-level PAGE_* names the views link through.
# It must run before st.navigation and before any view renders.
page_groups = nav.build()

# The sidebar carries the app's identity, the language toggle, and nothing else.
# It deliberately does **not** echo the selected opportunity: that was a
# page-wizard affordance from the first version, when every page hung off the
# product pick. Only Page A needs a product now, and it shows the selection in
# its own header (with a ✕ to clear it), so repeating it in the sidebar just
# followed the reader onto pages it means nothing on.
with st.sidebar:
    st.title(t(APP_TITLE_EN, APP_TITLE_ES))
    st.caption(t(APP_CAPTION_EN, APP_CAPTION_ES))
    # The canonical language lives in "lang"; the widget owns "lang-widget".
    # Keeping them apart is what makes a deselect harmless: st.segmented_control
    # lets a reader click the active pill to clear it, which would otherwise
    # blank the language mid-run and snap the whole app back to Spanish. On a
    # deselect the widget reads None and "lang" simply keeps the value resolved
    # at the top of this run.
    chosen = st.segmented_control(
        "Language",
        list(LANGS),
        format_func=lambda code: LANG_LABELS[code],
        default=_lang,
        key="lang-widget",
        label_visibility="collapsed",
    )
    st.session_state["lang"] = chosen or _lang

# Mirror the choice into the URL so the current view is shareable in the
# language it is being read in. Guarded on inequality: assigning to
# st.query_params reruns the script, and an unguarded write would never settle.
if st.query_params.get("lang") != st.session_state["lang"]:
    st.query_params["lang"] = st.session_state["lang"]

pg = st.navigation(page_groups)
pg.run()
