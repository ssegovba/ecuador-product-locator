"""Single place where the app's pages are declared.

Imported by main.py (to build st.navigation) and lazily by views (for
st.page_link / st.switch_page), so every module references the same st.Page
instances.

**Pages are rebuilt on every run** by :func:`build`, because their titles are
translated and module-level constants would freeze at import in whichever
language happened to be current then. ``build()`` rebinds the module-level
``PAGE_*`` / ``PAGE_GROUPS`` / ``PAGES`` names in place, so every view keeps
reading them exactly as before — nothing outside this module changed. It also
runs once at import, so a view rendered directly (the AppTest page runner does
exactly that, without main.py) still finds the names defined.

``url_path`` is fixed and never translated: links, bookmarks and the ``?lang=``
parameter all have to survive a language change.
"""

from __future__ import annotations

import streamlit as st

from lang import current_lang, t
from views import (
    assignment,
    bottom_up,
    candidate_list,
    candidates,
    home,
    opportunity,
)


# The language the current PAGE_* objects were built for. Rebuilding is skipped
# when it has not changed, which makes build() cheap enough for any caller to
# invoke defensively -- views/home.py does, so that rendering it directly (the
# AppTest page runner, which never executes main.py) still gets page links in
# the right language instead of whichever one happened to be current at import.
_BUILT_LANG: str | None = None


def build() -> dict[str, list[st.Page]]:
    """(Re)build the pages with current-language titles; return the nav groups.

    Called by main.py on every run before ``st.navigation``, by any view that
    reads the ``PAGE_*`` names, and once at import so those names always exist.
    A no-op when the language has not moved since the last build.
    """
    global _BUILT_LANG
    global PAGE_HOME, PAGE_OPPORTUNITY, PAGE_ASSIGNMENT, PAGE_BOTTOM_UP
    global PAGE_CANDIDATES, PAGE_CANDIDATE_LIST
    global HOME_PAGES, TOP_DOWN_PAGES, BOTTOM_UP_PAGES, SYNTHESIS_PAGES
    global PAGE_GROUPS, PAGES

    lang = current_lang()
    if _BUILT_LANG == lang:
        return PAGE_GROUPS
    _BUILT_LANG = lang

    # The default page. Every analysis page opens in the middle of an argument, so
    # the app lands on an explanation of what it contains first -- it is shared
    # publicly and most readers arrive without knowing there are two lenses at all.
    # url_path is set explicitly rather than left to Streamlit's default-page empty
    # path, so the six existing page URLs keep working unchanged.
    # Titled "Home", not "Start here": the group header above it already says
    # "Start here", and Streamlit draws both, so matching them made the sidebar
    # repeat itself. The group keeps the inviting phrasing; the entry is a plain
    # destination.
    PAGE_HOME = st.Page(
        home.render,
        title=t("Home", "Inicio"),
        url_path="home",
        default=True,
    )

    PAGE_OPPORTUNITY = st.Page(
        opportunity.render,
        title=t("Opportunity & provinces", "Oportunidad y provincias"),
        url_path="opportunity",
    )
    PAGE_ASSIGNMENT = st.Page(
        assignment.render,
        title=t("All opportunities × provinces", "Oportunidades × provincias"),
        url_path="assignment",
    )
    PAGE_BOTTOM_UP = st.Page(
        bottom_up.render,
        title=t("Province diversification explorer", "Explorador por provincia"),
        url_path="bottom-up",
    )
    # The explorer generalised to all 24 provinces at once — the bottom-up mirror of
    # PAGE_ASSIGNMENT, and placed at the same position in its group (the specific view
    # first, then the same ranking replayed for everything).
    PAGE_CANDIDATES = st.Page(
        candidates.render,
        title=t("Candidate industries × provinces", "Industrias candidatas × provincias"),
        url_path="candidates",
    )
    # The app's conclusion: the top-down assignment topped up from the bottom-up
    # ranking, with every row labelled for which lens found it. It gets a **lens of its
    # own** rather than a slot inside either contributing one — filing it under
    # "Bottom-up" would claim the list belongs to that half, and it also breaks the
    # symmetry the two existing lenses are built on (the specific view, then the same
    # ranking generalised), which this page does not share.
    PAGE_CANDIDATE_LIST = st.Page(
        candidate_list.render,
        title=t("The candidate industries list", "La lista de industrias candidatas"),
        url_path="candidate-list",
    )

    HOME_PAGES = [PAGE_HOME]
    TOP_DOWN_PAGES = [PAGE_OPPORTUNITY, PAGE_ASSIGNMENT]
    BOTTOM_UP_PAGES = [PAGE_BOTTOM_UP, PAGE_CANDIDATES]
    SYNTHESIS_PAGES = [PAGE_CANDIDATE_LIST]

    # Grouped navigation. The first two groups have the same shape: the specific view,
    # then the same ranking generalised to everything as a heatmap. The third is the
    # synthesis, listed **last** because it reads as a conclusion only once the reader
    # knows what each lens contributes — its header deliberately breaks the
    # "<Direction> — from a X" shape of the other two, since it is not a third
    # direction. No group gates any other: every page renders on its own, and no page
    # outside the top-down lens ever needs a product. The dict form is what draws the
    # section headers in the sidebar.
    # The landing page gets its own one-page group at the top rather than joining one
    # of the lenses: it belongs to neither, and a group-less entry would read as though
    # it were part of whichever group followed it.
    PAGE_GROUPS = {
        t("Start here", "Empieza aquí"): HOME_PAGES,
        t("Top-down — from an opportunity", "Enfoque descendente: desde una oportunidad"):
            TOP_DOWN_PAGES,
        t("Bottom-up — from a province", "Enfoque ascendente: desde una provincia"):
            BOTTOM_UP_PAGES,
        t("Both lenses in action", "Los dos enfoques en acción"): SYNTHESIS_PAGES,
    }

    PAGES = HOME_PAGES + TOP_DOWN_PAGES + BOTTOM_UP_PAGES + SYNTHESIS_PAGES
    return PAGE_GROUPS


build()
