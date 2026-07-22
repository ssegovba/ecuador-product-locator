from __future__ import annotations

import streamlit as st

import nav
import state
from config import APP_CAPTION, APP_TITLE

st.set_page_config(page_title=APP_TITLE, layout="wide")

state.init_state()

with st.sidebar:
    st.title(APP_TITLE)
    st.caption(APP_CAPTION)
    if st.session_state.get("selection_complete"):
        st.success(
            f"Product: HS {st.session_state['hs_code']} "
            f"({st.session_state['hs_edition']}) · "
            f"{len(st.session_state['selected_isic'])} industries selected"
        )

pg = st.navigation(nav.PAGES)
pg.run()
