"""The landing page — what this app contains, and how to move through it.

The app's default page. It exists because the five analysis pages all open in the
middle of an argument: each one assumes you already know which of the two lenses
you are in and what that lens takes as given. A first-time reader — this app is
shared publicly with Ecuadorian policymakers — needs the shape of the whole thing
before any single page makes sense.

**No analysis happens here.** The page reads no exports and computes nothing; it is
navigation and orientation only, which is also why it is the one page that cannot
go stale against a data refresh. Every figure it quotes is a structural count (how
many opportunities, how many provinces), not a result.

Bilingual through ``lang.t``, Spanish by default. The Spanish copy is a recast
rather than a transliteration: it carries no em dashes (see ``lang.py``), and its
vocabulary follows the Spanish report — *enfoque descendente / ascendente*,
*factibilidad*, *atractivo*, *industrias candidatas*, *plazas equivalentes*.
"""

from __future__ import annotations

import streamlit as st

import state
from config import APP_TITLE_EN, APP_TITLE_ES, CL_MIN_INDUSTRIES
from lang import t


def _lens(title: str, blurb: str, pages: list, captions: list[str]) -> None:
    """One lens block: heading, what it answers, then a link per page.

    ``pages`` are ``st.Page`` objects; the link labels come from the pages
    themselves, so a retitled page cannot end up with two names in the app —
    and so the links follow the language for free, since ``nav`` rebuilds the
    pages with translated titles on every run.

    An empty ``title`` renders no heading. The third section has only one lens, so
    its ``##`` section heading already names it and a ``####`` under it would be the
    same words twice.
    """
    if title:
        st.markdown(f"#### {title}")
    st.markdown(blurb)
    for page, caption in zip(pages, captions):
        st.page_link(page, icon=":material/arrow_forward:")
        st.caption(caption)


def render() -> None:
    state.init_state()

    # Imported inside render(): nav.py imports this module (via `views`), so a
    # module-level `import nav` would be a circular import. nav's own docstring
    # names this as the expected pattern for views that link to other pages.
    import nav

    # Cheap no-op when the language has not changed. It matters on the paths that
    # never run main.py (the AppTest page runner renders a view directly), where
    # nav's module-level pages would otherwise keep the language they were built
    # with at import time.
    nav.build()

    st.title(t(APP_TITLE_EN, APP_TITLE_ES))
    # No caption under the title: the sidebar shows APP_CAPTION verbatim on every
    # page, so repeating it here was the same sentence twice on one screen.

    st.markdown(
        t(
            "Every province of Ecuador is already good at something, and what it could "
            "plausibly do next is constrained by that — capabilities move to nearby "
            "activities far more readily than they appear from nothing. This app works that "
            "idea in two directions and reports where the two agree.\n\n"
            "The answer it is built to give is an **industry**: a line of activity a province "
            "could realistically grow into, named in the same classification the business "
            "registry uses, so it can be looked up, argued with, and checked against what is "
            "already there. Everything the app shows is measured from **what Ecuador's economy "
            "records today** — the 2024 business registry and the country's own export record. "
            "Nothing here is a forecast, and no number is a job this app expects to be created.",
            "Cada provincia del Ecuador ya es buena en algo, y eso condiciona lo que podría "
            "hacer a continuación: las capacidades se trasladan a actividades cercanas con "
            "mucha más facilidad de la que surgen desde cero. Esta aplicación presenta esa idea "
            "en dos direcciones y muestra dónde coinciden los dos resultados.\n\n"
            "La respuesta que busca dar es una **industria**: una línea de actividad a la que "
            "una provincia podría expandirse de forma realista, nombrada en la misma "
            "clasificación que usa el registro de empresas, de modo que se pueda consultar, "
            "discutir y contrastar con lo que ya existe. Todo lo que se muestra aquí se mide "
            "sobre **lo que hoy registra la economía ecuatoriana**, es decir el registro de "
            "empresas de 2024 y el historial exportador del propio país. Nada de esto es un "
            "pronóstico, y ninguna cifra es un empleo que esta aplicación espere que se cree.",
        )
    )

    st.divider()
    st.markdown(f"## {t('The two lenses', 'Las dos aproximaciones')}")
    st.markdown(
        t(
            "They start from opposite ends. One begins with a **product** Ecuador could "
            "export and asks which provinces could make it; the other begins with a "
            "**province** and asks what it could add. Neither is sufficient alone, which is "
            "why the third section exists.",
            "Parten de extremos opuestos. Una arranca de un **producto** que el Ecuador "
            "podría exportar y pregunta qué provincias podrían elaborarlo; la otra arranca de "
            "una **provincia** y pregunta qué podría añadir. Ninguna basta por sí sola, y por "
            "eso existe la tercera sección.",
        )
    )

    left, right = st.columns(2, gap="large")

    with left:
        _lens(
            t("Top-down — from an opportunity", "Enfoque descendente: desde una oportunidad"),
            t(
                "Starts from **50 export opportunities** already identified for Ecuador "
                "(20 on the intensive margin, 30 on the extensive). It maps each product to the industries that make it, then "
                "ranks provinces on how much of that industry mix they **already "
                "have** — assigning the product wherever presence reaches the national "
                "average. How many provinces qualify is a finding, not a setting.",
                "Parte de **50 oportunidades de exportación** ya identificadas para el "
                "Ecuador (20 en el margen intensivo y 30 en el extensivo). Vincula cada "
                "producto con las industrias que lo elaboran y luego ordena las provincias "
                "según cuánto de esa combinación de industrias **ya tienen de forma "
                "competitiva**, asignando el producto donde la presencia alcanza el "
                "promedio nacional. El número de provincias al que se asigna el producto es un resultado, no un "
                "parámetro que se fije.",
            ),
            [nav.PAGE_OPPORTUNITY, nav.PAGE_ASSIGNMENT],
            [
                t(
                    "One opportunity at a time: its industries, a map and a ranking of the "
                    "provinces, and the explanation behind any score worked through.",
                    "Una oportunidad a la vez: sus industrias, un mapa y un ordenamiento de "
                    "las provincias, y la explicación detrás de cualquier puntaje, paso a paso.",
                ),
                t(
                    "All 50 opportunities at once, as a grid — which products land where, and "
                    "which industries that puts to work.",
                    "Las 50 oportunidades a la vez, en una matriz: qué productos llegan a "
                    "dónde y qué industrias pone eso en juego.",
                ),
            ],
        )

    with right:
        _lens(
            t("Bottom-up — from a province", "Enfoque ascendente: desde una provincia"),
            t(
                "Starts from **one province's economy** as the registry records it, and ranks "
                "the industries it could add on two axes: how **feasible** an industry is "
                "there (what the province already does, plus how close the industry sits to "
                "it) and how **attractive** it is (wages, or jobs per firm). You set the "
                "balance between them.",
                "Parte de **la economía de una provincia** tal como la recoge el registro de "
                "empresas y ordena las industrias que podría añadir en dos ejes: la "
                "**factibilidad** de cada industria allí (lo que la provincia ya hace, más "
                "qué tan cerca se ubica la industria de eso) y su **atractivo** (salarios, o "
                "empleo por empresa). Usted fija el balance entre los dos.",
            ),
            [nav.PAGE_BOTTOM_UP, nav.PAGE_CANDIDATES],
            [
                t(
                    "One province in depth: what it is made of, how it compares with Ecuador "
                    "and its regional peers, and its top industries on the feasibility × "
                    "attractiveness plane.",
                    "Una provincia en detalle: de qué está hecha, cómo se compara con el "
                    "Ecuador y con sus pares regionales, y sus principales industrias en el "
                    "plano factibilidad × atractivo.",
                ),
                t(
                    "The same ranking replayed for all 24 provinces at once, as an "
                    "industry × province grid.",
                    "El mismo ordenamiento repetido para las 24 provincias a la vez, en una "
                    "matriz industria × provincia.",
                ),
            ],
        )

    st.divider()
    st.markdown(f"## {t('Both lenses in action', 'Los dos enfoques en acción')}")
    _lens(
        # No heading of its own: the section heading above already names it.
        "",
        t(
            f"The app's conclusion. Each province's **candidate industries list**: everything "
            f"the opportunity assignment put there, topped up from that province's own "
            f"bottom-up ranking until it holds at least {CL_MIN_INDUSTRIES} industries. Every "
            "row is labelled with which lens found it, and each one is priced against "
            "**Ecuador's own frontier** — what the province holding the largest share of its "
            "economy in that industry actually achieves. ",
            f"La conclusión de la aplicación. La **lista de industrias candidatas** de cada "
            f"provincia: todo lo que le atribuyó la asignación de oportunidades, completado "
            f"con el ordenamiento ascendente de esa misma provincia hasta reunir al menos "
            f"{CL_MIN_INDUSTRIES} industrias. Cada fila indica qué enfoque la encontró, y cada "
            "una se valora contra **la propia frontera del Ecuador**: lo que efectivamente "
            "alcanza la provincia que dedica la mayor parte de su economía a esa industria. ",
        ),
        [nav.PAGE_CANDIDATE_LIST],
        [
            t(
                "The lists themselves, province by province, with the frontier comparison and "
                "downloads for one province or all 24.",
                "Las listas mismas, provincia por provincia, con la comparación contra la "
                "frontera y descargas para una provincia o para las 24.",
            ),
        ],
    )

    st.divider()
    st.markdown(f"## {t('How to read the app', 'Cómo leer la aplicación')}")
    st.markdown(
        t(
            "- **If you have a province in mind**, go straight to the "
            "*candidate industries list* for its shortlist, then to the "
            "*province diversification explorer* to see how that shortlist was reached.\n"
            "- **If you have a product in mind**, start at *Opportunity & provinces*.\n"
            "- **Every page carries its settings in a badge**, so a figure can always be "
            "traced back to the choices that produced it.\n",
            "- **Si tiene una provincia en mente**, vaya directo a *La lista de industrias "
            "candidatas* para ver su selección, y luego al *Explorador por provincia* para "
            "entender cómo se llegó a ella.\n"
            "- **Si tiene un producto en mente**, empiece por *Oportunidad y provincias*.\n"
            "- **Cada página lleva sus parámetros en una etiqueta**, de modo que siempre se "
            "puede rastrear un gráfico hasta las decisiones que la produjeron.\n",
        )
    )

    st.divider()
    st.caption(
        t(
            "**Sources.** Presence, employment and establishments: INEC's business registry "
            "(REEM) 2024, counted where establishments operate. Opportunities: the 50 HS4 "
            "products identified for Ecuador, HS 2022, mapped to industries through an "
            "HS→ISIC concordance. Industry classification: CIIU Rev. 4. Employment figures "
            "are registered formal full-time-equivalent positions held **today** — a stock, "
            "never a projection.",
            "**Fuentes.** Presencia, empleo y establecimientos: Registro Estadístico de "
            "Empresas y Establecimientos (REEM) del INEC, 2024, contabilizados donde operan "
            "los establecimientos. Oportunidades: los 50 productos identificados para el "
            "Ecuador a cuatro dígitos del Sistema Armonizado, clasificación de 2022, "
            "vinculados a industrias mediante la concordancia entre el Sistema Armonizado y "
            "la CIIU. Clasificación de industrias: CIIU Rev. 4. Las cifras de empleo son "
            "plazas equivalentes formales registradas **hoy**, es decir un acervo y nunca "
            "una proyección.",
        )
    )
