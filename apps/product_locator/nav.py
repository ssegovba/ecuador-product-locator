"""Single place where the wizard's pages are declared.

Imported by main.py (to build st.navigation) and lazily by state.py /
views (for st.page_link / st.switch_page), so every module references the
same st.Page instances.
"""

from __future__ import annotations

import streamlit as st

from views import geography, province_profile, relatedness, select_product

PAGE_SELECT = st.Page(
    select_product.render, title="1 · Product & industries", url_path="select", default=True
)
PAGE_GEOGRAPHY = st.Page(geography.render, title="2 · Geographic analysis", url_path="geography")
PAGE_PROVINCE = st.Page(province_profile.render, title="3 · Province profile", url_path="province")
PAGE_RELATEDNESS = st.Page(
    relatedness.render, title="4 · Related industries & opportunities", url_path="opportunities"
)

PAGES = [PAGE_SELECT, PAGE_GEOGRAPHY, PAGE_PROVINCE, PAGE_RELATEDNESS]
