"""The candidate industries list — where the app's two lenses meet.

The other five pages each argue one half of a case. *Top-down* starts from one of the
50 HS4 diversification opportunities and asks which provinces already have the
industries that make it; *bottom-up* starts from a province and ranks what it could
plausibly move into. Neither produces the thing a policymaker actually needs, which is
**a province's list**.

This page is that list, and it is neither lens's output — hence its own lens in the
sidebar rather than a sixth page filed under one of the two. The top-down assignment is
the base; any province it leaves under :data:`config.CL_MIN_INDUSTRIES` is topped up
from that province's bottom-up ranking, and **every row carries a ``lens`` column
saying which put it there**. Section 1 stacks the two for all 24 provinces at once.

**The list is derived in-app and this page reads no CSVs.** The same list is a
committed analysis artifact — ``data/processed/summary_tables/province_candidates/``,
built by ``utils/build_province_candidates.py`` — but the app assembles it live from the
same upstream exports every other page reads (:func:`metrics.candidate_list`). That
makes the app deployable to the general public with no builder run, keeps this page
consistent with the explorer and the grids *by construction*, and removes the
stale-file failure mode outright: there is no file to go stale. The app and the builder
are tied together instead by an **equality tripwire** in the test suite — the derived
list must equal the shipped finals on the full column set — so a divergence is a red
test at pin time rather than a wrong page at runtime.

**One control: the attractiveness metric**, and it is the rare control in this app that
changes *membership* rather than ordering — only 31 of the 79 top-ups (39%) are shared
between the two settings. The page says so at the control. Everything else the
derivation needs is pinned in ``config`` at the same values the builder uses
(``CL_*``), including ``k = 10``, which is deliberately not a slider: ten is enough
industries that a reader can look for structure in the list while staying small enough
to investigate seriously, and it was chosen for how a person uses a list rather than for
a statistical property.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import state
from config import (
    CL_BU_BETA,
    CL_BU_TOP_K,
    CL_BU_WEIGHT,
    CL_MIN_INDUSTRIES,
    CL_TD_BASE,
    CL_TD_CUT,
    CL_TD_ROUTE_WEIGHT,
    CL_TD_SUPPORT,
    CL_TD_VIEW,
    GSECTOR_COLORS,
    LENS_COLORS,
    LENS_LABELS,
    LENS_LABELS_ES,
    MIN_CROSSWALK_WEIGHT,
    MIN_ESTAB,
    MIN_PLAZAS,
    STATE_COLORS,
    TD_EXCLUDED_PROVINCES,
    THIN_NEIGHBORHOOD_THRESHOLD,
)
from data_loader import (
    load_class_labels,
    load_density_support,
    load_labor_class_employment,
    load_labor_composition,
    load_labor_profile,
    load_opportunities,
    load_province_density,
    load_province_rca,
    load_tradability,
    load_tradable_menu,
)
from metrics import candidate_list, leader_benchmark
from lang import coerce_choice, t
from viz import (
    POTENTIAL_VIEWS,
    make_candidates_bar,
    make_potential_treemap,
    potential_view,
    state_labels,
)

# Same registry and wording as the explorer's and the grid's own controls: a reader
# arriving from either must not meet a third name for the same knob.
_ATTR_CHOICES = {
    "wage_per_fte": "Average wage paid",
    "fte_per_firm": "Average jobs per firm",
}
_ATTR_CHOICES_ES = {
    "wage_per_fte": "Salario promedio pagado",
    "fte_per_firm": "Empleo promedio por empresa",
}
_ATTR_DEFAULT = "wage_per_fte"
_HELP_ATTR_EN = (
    "Which attractiveness figure ranks the bottom-up top-ups. **This one changes who is "
    "on the list, not just the order** — only 39% of the top-up picks are shared between "
    "the two settings. Average wage paid: what a full-time-equivalent worker earns per "
    "year in the industry (2024 business registry). · Average jobs per firm: how many "
    "jobs a typical employing firm carries. Both are **national** figures. The top-down "
    "rows are unaffected either way — that lens does not rank on attractiveness at all, "
    "which is why both settings share an identical top-down block."
)
_HELP_ATTR_ES = (
    "Qué cifra de atractivo ordena los complementos del enfoque ascendente. **Esta "
    "opción cambia quién está en la lista, no solo el orden**: solo el 39% de los "
    "complementos elegidos coincide entre los dos ajustes. Salario promedio pagado: lo "
    "que gana al año una plaza equivalente en la industria (registro de empresas 2024). "
    "· Empleo promedio por empresa: cuántos puestos sostiene una empresa empleadora "
    "típica. Ambas son cifras **nacionales**. Las filas del enfoque descendente no se "
    "ven afectadas en ningún caso, porque ese enfoque no ordena por atractivo, y por eso "
    "los dos ajustes comparten un bloque descendente idéntico."
)

# Keyed by code, never by display text: both are persisted in session state
# (cl-pot-view, cl-pot-color), so a translated label as the key would break the
# lookup or orphan the pick when the language changes.
_POT_VIEWS = {"potential": "Potential", "gap": "Gap to the frontier"}
_POT_VIEWS_ES = {"potential": "Potencial", "gap": "Brecha hasta la frontera"}
_POT_VIEW_DEFAULT = "potential"
_POT_COLORS = {"sector": "Broad sector", "state": "Presence state"}
_POT_COLORS_ES = {"sector": "Sector amplio", "state": "Estado de presencia"}
_POT_COLOR_DEFAULT = "sector"

_HELP_POT_VIEW_EN = (
    "**Potential** — what the industry would employ here if this province were as "
    "specialized in it as the most specialized province in Ecuador. · **Gap to the "
    "frontier** — that figure minus what the province already holds, so the tiles size "
    "the distance still to travel rather than the destination. Both are **stocks**, "
    "benchmarked on activity that already exists somewhere in Ecuador — never a forecast "
    "of jobs an entrant would create, and both denominated in the same registered FTE "
    "(`plazas_equiv`) as the rest of the analysis.\n\n"
    "Note the tiles are **employment**, while the presence states beside them in the "
    "table are **establishment-based** — a share of an economy is the only unit a "
    "benchmark like this can be expressed in. The table's **✓ Both bases** column says, "
    "row by row, whether a state survives the switch.\n\n"
    "⚠ **Both views leave some industries out**, and a treemap has no zero-area tile to "
    "show them with: one that no province in Ecuador holds 50 FTE in cannot be priced at "
    "all, and one already at the frontier has a gap of zero. The caption below counts and "
    "names whatever the view on screen drops — and **the table underneath always carries "
    "the province's full list**, tile or no tile."
)
_HELP_POT_VIEW_ES = (
    "**Potencial**: lo que la industria emplearía aquí si esta provincia estuviera tan "
    "especializada en ella como la provincia más especializada del Ecuador. · **Brecha "
    "hasta la frontera**: esa cifra menos lo que la provincia ya tiene, de modo que los "
    "mosaicos dimensionan la distancia por recorrer en lugar del destino. Ambos son "
    "**acervos**, referenciados en actividad que ya existe en alguna parte del Ecuador, "
    "nunca un pronóstico de empleos que crearía un entrante, y ambos denominados en las "
    "mismas plazas equivalentes registradas (`plazas_equiv`) que el resto del "
    "análisis.\n\n"
    "Note que los mosaicos son **empleo**, mientras que los estados de presencia que "
    "están junto a ellos en la tabla se basan en **establecimientos**: una participación "
    "en una economía es la única unidad en la que puede expresarse un referente como "
    "este. La columna **✓ Ambas bases** de la tabla indica, fila por fila, si el estado "
    "sobrevive al cambio.\n\n"
    "⚠ **Las dos vistas dejan algunas industrias fuera**, y un mapa de mosaicos no tiene "
    "un mosaico de área cero para mostrarlas: una industria en la que ninguna provincia "
    "del Ecuador tiene 50 plazas equivalentes no puede valorarse en absoluto, y una que "
    "ya está en la frontera tiene una brecha de cero. La nota de abajo cuenta y nombra lo "
    "que deje fuera la vista en pantalla, y **la tabla de más abajo siempre lleva la "
    "lista completa de la provincia**, con mosaico o sin él."
)
_HELP_POT_COLOR_EN = (
    "**Broad sector** — the same colors as the explorer's composition treemap, so the two "
    "can be read against each other. · **Presence state** — what the province has of the "
    "industry today, the colors of the explorer's plane.\n\nThe tiles are always laid out "
    "by broad sector and **nothing moves when you switch**: only the colors change, so an "
    "industry you have found stays exactly where it is."
)
_HELP_POT_COLOR_ES = (
    "**Sector amplio**: los mismos colores que el mapa de composición del explorador, "
    "para poder leer los dos uno contra otro. · **Estado de presencia**: lo que la "
    "provincia tiene hoy de la industria, con los colores del plano del explorador.\n\n"
    "Los mosaicos siempre se disponen por sector amplio y **nada se mueve al cambiar**: "
    "solo cambian los colores, así que una industria que usted haya encontrado se queda "
    "exactamente donde está."
)

_TABLE_COLUMNS_EN = {
    "rank_in_province": "#",
    "lens": "Found by",
    "display_code": "CIIU",
    "class_name": "Industry",
    "gsector_label": "Broad sector",
    "state": "State",
    "score": "Score",
    "n_opportunities": "Opportunities",
    "hs4_codes": "HS4 codes",
    "accessible_market_busd": "Accessible market (B USD)",
    "plazas_equiv": "Employment (FTE, here)",
    "n_estab": "Establishments (here)",
    "potential_plazas": "Potential (FTE)",
    "gap_plazas": "Gap (FTE)",
    "leader_province": "Frontier province",
    "both_bases": "✓ Both bases",
}
_TABLE_COLUMNS_ES = {
    "rank_in_province": "#",
    "lens": "Encontrada por",
    "display_code": "CIIU",
    "class_name": "Industria",
    "gsector_label": "Sector amplio",
    "state": "Estado",
    "score": "Puntaje",
    "n_opportunities": "Oportunidades",
    "hs4_codes": "Códigos HS4",
    "accessible_market_busd": "Mercado accesible (miles de M USD)",
    "plazas_equiv": "Empleo (plazas, aquí)",
    "n_estab": "Establecimientos (aquí)",
    "potential_plazas": "Potencial (plazas)",
    "gap_plazas": "Brecha (plazas)",
    "leader_province": "Provincia frontera",
    "both_bases": "✓ Ambas bases",
}


def _table_columns() -> dict[str, str]:
    """The table's {source column: header} map in the current language."""
    return {
        key: t(_TABLE_COLUMNS_EN[key], _TABLE_COLUMNS_ES[key])
        for key in _TABLE_COLUMNS_EN
    }


# show_spinner=False on purpose: a decorator argument is evaluated once at import,
# so a t() call here would freeze the spinner in whichever language was current
# then. The message is raised at the call site instead, where it follows the run.
@st.cache_data(show_spinner=False)
def candidate_list_frame(var: str) -> pd.DataFrame:
    """The 24-province list for one attractiveness metric, derived from the exports.

    Cached per metric, so toggling the control is instant after each has been built
    once. **The single wiring** of :func:`metrics.candidate_list` in the app — the
    equality tripwire in the test suite calls this same function, so the thing pinned
    against the committed CSVs is the thing the page renders, not a re-assembled
    lookalike.
    """
    labels = load_class_labels()
    profile = load_labor_profile().copy()
    # The app-wide Spanish-preferred class names, exactly as every other ranking page
    # overlays them before scoring.
    profile["class_name"] = profile["codigo_clase"].map(labels).fillna(profile["class_name"])
    comp = load_labor_composition()
    density = load_province_density()
    tradability = load_tradability()
    return candidate_list(
        var=var,
        provinces=sorted(code for code in density["id_code"].unique() if code != "99"),
        opportunities=load_opportunities(),
        rca=load_province_rca(),
        density=density,
        density_plazas=load_province_density(base="plazas"),
        profile=profile,
        emp=load_labor_class_employment(),
        support=load_density_support(),
        menu=load_tradable_menu() & set(comp.loc[comp["level"] == "class", "code"]),
        class_labels=labels,
        entry_regime=dict(zip(tradability["codigo_clase"], tradability["entry_regime"])),
        min_estab=MIN_ESTAB,
        min_plazas=MIN_PLAZAS,
        min_crosswalk_weight=MIN_CROSSWALK_WEIGHT,
        thin_threshold=THIN_NEIGHBORHOOD_THRESHOLD,
        td_view=CL_TD_VIEW,
        td_base=CL_TD_BASE,
        td_support=CL_TD_SUPPORT,
        td_route_weight=CL_TD_ROUTE_WEIGHT,
        td_cut=CL_TD_CUT,
        beta=CL_BU_BETA,
        weight=CL_BU_WEIGHT,
        top_k=CL_BU_TOP_K,
        min_industries=CL_MIN_INDUSTRIES,
        # `provinces` above stays all 24 on purpose: the exclusion belongs to the
        # top-down leg, so Galapagos arrives with no top-down rows and is filled
        # entirely from its own bottom-up ranking.
        td_excluded=TD_EXCLUDED_PROVINCES,
    )


def _lens_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per province: the two lenses' counts and the total."""
    counts = (
        frame.pivot_table(
            index=["id_code", "id_name", "region"], columns="lens",
            values="codigo_clase", aggfunc="count", fill_value=0,
        )
        .reset_index()
    )
    for lens in ("top_down", "bottom_up"):
        if lens not in counts.columns:
            counts[lens] = 0
    counts["total"] = counts["top_down"] + counts["bottom_up"]
    return counts


def _province_label(row: pd.Series) -> str:
    return f"{str(row['id_name']).strip()} ({row['region']})"


def _downloadable(rows: pd.DataFrame) -> pd.DataFrame:
    """The list as served by both download buttons, in rank order.

    ``current_plazas`` is dropped: it is a straight duplicate of ``plazas_equiv``
    (verified equal to 1e-6 on every benchmarked row) that is additionally blank where
    there is no frontier, so it carries nothing the neighbouring columns do not. The file
    is therefore one column narrower than the committed
    ``province_candidates_final_<metric>.csv``, which still ships it.
    """
    return (
        rows.sort_values(["id_code", "rank_in_province"], kind="mergesort")
        .drop(columns=["current_plazas"])
        .reset_index(drop=True)
    )


def _missing_tiles(rows: pd.DataFrame, view: str) -> pd.DataFrame:
    """The rows the treemap cannot draw in this view — zero-area or unpriceable.

    Derived from the frame on screen rather than from any stored count, so it cannot
    drift from what the figure did with the same rows.
    """
    values = pd.to_numeric(rows[POTENTIAL_VIEWS[view]["column"]], errors="coerce")
    return rows[values.isna() | (values <= 0)]


def render() -> None:
    state.init_state()  # ungated on purpose: no product and no province required

    st.title(t("The candidate industries list", "La lista de industrias candidatas"))
    st.markdown(
        t(
            "Both lenses, joined into the thing a province can actually work from: **at "
            f"least {CL_MIN_INDUSTRIES} industries each, for all 24 provinces**. The "
            "base is the top-down assignment — the industries the 50 HS4 opportunities "
            "put in a province because it already has the presence to make them. Where "
            f"that reaches fewer than {CL_MIN_INDUSTRIES}, the rest are **topped up** "
            "from the province's own bottom-up ranking. Every row says which lens found "
            "it, so the evidence behind it is never in doubt.",
            "Los dos enfoques, unidos en algo con lo que una provincia puede realmente "
            f"trabajar: **al menos {CL_MIN_INDUSTRIES} industrias cada una, para las 24 "
            "provincias**. La base es la asignación descendente, es decir las industrias "
            "que las 50 oportunidades HS4 colocan en una provincia porque ya tiene la "
            f"presencia para elaborarlas. Donde eso no llega a {CL_MIN_INDUSTRIES}, el "
            "resto se **completa** con el ordenamiento ascendente de la propia "
            "provincia. Cada fila indica qué enfoque la encontró, así que la evidencia "
            "detrás nunca queda en duda.",
        )
    )

    controls = st.columns([3, 5])
    with controls[0]:
        coerce_choice("cl-attr", list(_ATTR_CHOICES), _ATTR_CHOICES, _ATTR_CHOICES_ES)
        choice = st.segmented_control(
            t("Attractiveness metric", "Métrica de atractivo"), list(_ATTR_CHOICES),
            default=_ATTR_DEFAULT, key="cl-attr",
            help=t(_HELP_ATTR_EN, _HELP_ATTR_ES),
            format_func=lambda code: t(
                _ATTR_CHOICES.get(code, code), _ATTR_CHOICES_ES.get(code, code)
            ),
        )
        var = choice or _ATTR_DEFAULT
    with controls[1]:
        st.caption(
            t(
                "⚠ **This control changes who is on the list, not just the order.** It "
                "ranks the bottom-up top-ups, and only **39%** of them are shared "
                "between the two settings. The top-down rows are identical either way — "
                "that lens does not rank on attractiveness.",
                "⚠ **Este control cambia quién está en la lista, no solo el orden.** "
                "Ordena los complementos del enfoque ascendente, y solo el **39%** de "
                "ellos coincide entre los dos ajustes. Las filas del enfoque descendente "
                "son idénticas en ambos casos, porque ese enfoque no ordena por atractivo.",
            )
        )

    with st.spinner(
        t("Assembling the candidate list…", "Armando la lista de candidatas…")
    ):
        frame = candidate_list_frame(var)
    counts = _lens_counts(frame)

    # ------------------------------------------------------------------ #
    # Section 1 — the whole country.                                      #
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader(
        t("How many industries does each province get, and from which lens?",
          "¿Cuántas industrias recibe cada provincia, y de qué enfoque?")
    )
    st.caption(
        t(
            "One bar per province, sorted by total. The dark segment is what the 50 "
            "opportunities assigned; the red is what the province's own ranking had to "
            f"add to reach {CL_MIN_INDUSTRIES}. A short, all-red bar is a province the "
            "product analysis never reached — which is exactly why the bottom-up lens "
            "exists.",
            "Una barra por provincia, ordenadas por total. El segmento oscuro es lo que "
            "asignaron las 50 oportunidades; el rojo es lo que el propio ordenamiento de "
            f"la provincia tuvo que añadir para llegar a {CL_MIN_INDUSTRIES}. Una barra "
            "corta y toda roja es una provincia a la que el análisis de productos nunca "
            "llegó, que es justamente la razón por la que existe el enfoque ascendente.",
        )
    )
    st.plotly_chart(
        make_candidates_bar(
            counts, LENS_COLORS,
            {k: t(LENS_LABELS[k], LENS_LABELS_ES[k]) for k in LENS_LABELS},
            CL_MIN_INDUSTRIES,
        ),
        width="stretch",
        key=f"cl-bar-{var}",
    )

    n_top_down = int(counts["top_down"].sum())
    n_top_up = int(counts["bottom_up"].sum())
    at_floor = counts[counts["total"] == CL_MIN_INDUSTRIES]
    none_top_down = counts[counts["top_down"] == 0]
    # **Recommendations, not industries.** The row count is province x industry pairs;
    # the same industry is a candidate in several provinces (C2750 in 15 of them), so
    # calling these "337 industries" would overstate the breadth of the list four-fold.
    # The distinct count moves with the metric, so it is derived rather than quoted.
    n_distinct = frame["codigo_clase"].nunique()
    # Why a province gets nothing from the top-down lens is two different facts, and
    # they read the same in a count. Most such provinces clear parity on nothing — the
    # rule can only reach an industry a province already visibly has. Galápagos gets
    # nothing because the lens is **not allowed** to assign it (TD_EXCLUDED_PROVINCES),
    # which is a decision rather than a measurement, so the caption separates the two
    # instead of implying the assignment tried and failed there.
    unreached = none_top_down[~none_top_down["id_code"].isin(TD_EXCLUDED_PROVINCES)]
    withheld = none_top_down[none_top_down["id_code"].isin(TD_EXCLUDED_PROVINCES)]
    why = (
        t(
            "no opportunity's ranking reaches "
            f"{'them' if len(unreached) != 1 else 'it'}",
            "el ordenamiento de ninguna oportunidad "
            f"{'las alcanza' if len(unreached) != 1 else 'la alcanza'}",
        )
        if not unreached.empty
        else ""
    )
    if not withheld.empty:
        names = ", ".join(str(n).strip() for n in withheld["id_name"])
        withheld_note = (
            t(
                f"the top-down lens does not assign {names} at all",
                f"el enfoque descendente no asigna nada a {names}",
            )
            if not why
            else t(
                f", and the lens does not assign {names} at all",
                f", y el enfoque no asigna nada a {names}",
            )
        )
        why = f"{why}{withheld_note}"
    withheld_names = ", ".join(
        str(name).strip() for name in none_top_down["id_name"]
    )
    st.caption(
        t(
            f"**{len(frame):,} province–industry recommendations across {len(counts)} "
            f"provinces** — {n_top_down:,} from the opportunity assignment, "
            f"{n_top_up:,} topped up — drawn from **{n_distinct} distinct "
            "industries**, since the same industry can be a candidate in several "
            f"places. {len(at_floor)} provinces sit at exactly the floor of "
            f"{CL_MIN_INDUSTRIES}, and **{len(none_top_down)} get nothing at all from "
            f"the top-down lens** ({withheld_names}): {why}, so their entire list is "
            "the bottom-up top-up.",
            f"**{len(frame):,} recomendaciones provincia–industria en {len(counts)} "
            f"provincias**: {n_top_down:,} de la asignación de oportunidades y "
            f"{n_top_up:,} completadas, tomadas de **{n_distinct} industrias "
            "distintas**, ya que la misma industria puede ser candidata en varios "
            f"lugares. {len(at_floor)} provincias están exactamente en el piso de "
            f"{CL_MIN_INDUSTRIES}, y **{len(none_top_down)} no reciben nada del enfoque "
            f"descendente** ({withheld_names}): {why}, así que toda su lista es el "
            "complemento ascendente.",
        )
    )
    by_region = counts.groupby("region")["total"].agg(["mean", "count"]).sort_values(
        "mean", ascending=False
    )
    st.caption(
        t("**By region** — ", "**Por región**: ")
        + " · ".join(
            t(
                f"{region} {row['mean']:.1f} on average "
                f"({int(row['count'])} provinces)",
                f"{region} {row['mean']:.1f} en promedio "
                f"({int(row['count'])} provincias)",
            )
            for region, row in by_region.iterrows()
        )
        + t(
            ". The spread is a finding about the *evidence*, not about the provinces: "
            "the assignment rule can only reach an industry a province already visibly "
            "has, so a large diversified economy collects marks from many opportunities "
            "and a small one collects none. Every province still leaves this page with "
            "a list.",
            ". La dispersión es un hallazgo sobre la *evidencia*, no sobre las "
            "provincias: la regla de asignación solo puede alcanzar una industria que "
            "una provincia ya tenga visiblemente, así que una economía grande y "
            "diversificada acumula marcas de muchas oportunidades y una pequeña no "
            "acumula ninguna. Toda provincia sale igualmente de esta página con una "
            "lista.",
        )
    )

    # ------------------------------------------------------------------ #
    # Section 2 — one province.                                          #
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader(t("How much could each of these be worth?",
                   "¿Cuánto podría valer cada una de estas?"))

    options = counts.sort_values("id_code")["id_code"].tolist()
    names = {row["id_code"]: _province_label(row) for _, row in counts.iterrows()}
    # Clear (✕) beside the box, rendered BEFORE the selectbox so resetting the widget's
    # session key is allowed — the same ordering the explorer and Page A use.
    prov_col, prov_clear, _ = st.columns([10, 1, 7], vertical_alignment="bottom")
    with prov_clear:
        if st.button(
            "✕", key="cl-province-clear",
            help=t("Clear the selected province", "Quitar la provincia seleccionada"),
            disabled=st.session_state.get("cl-province") is None,
        ):
            st.session_state["cl-province"] = None
            st.rerun()
    with prov_col:
        coerce_choice("cl-province", options)
        province = st.selectbox(
            t("Province", "Provincia"),
            options=options,
            index=None,
            format_func=lambda code: names.get(code, code),
            placeholder=t("Type a province name to search",
                          "Escriba el nombre de una provincia"),
            key="cl-province",
            help=t(
                "Prices that province's list against Ecuador's own frontier.",
                "Valora la lista de esa provincia contra la propia frontera del "
                "Ecuador.",
            ),
        )
    if province is None:
        st.info(
            t(
                "Pick a province to see its list priced against Ecuador's own "
                "frontier. The chart above needs no selection.",
                "Elija una provincia para ver su lista valorada contra la propia "
                "frontera del Ecuador. El gráfico de arriba no necesita selección.",
            )
        )
        return

    place = names[province].split(" (")[0]
    rows = frame[frame["id_code"] == province].copy()

    # The treemap speaks the benchmark's own column names (it is shared with nothing
    # else now, but those are the names :func:`metrics.potential_gap` produces and the
    # figure's docstring documents). Kept as a **separate frame** so `rows` stays in the
    # list's schema for the table and the download below.
    tiles = rows.rename(columns={
        "potential_plazas": "potential", "gap_plazas": "gap",
        "leader_province": "leader_id_name",
    }).copy()
    # The hover's "today" figure. ``plazas_equiv`` rather than the list's own
    # ``current_plazas``, which is a straight duplicate of it (verified equal to 1e-6 on
    # every benchmarked row) that is additionally blank where there is no frontier —
    # ``plazas_equiv`` is the employment variable the whole analysis is denominated in,
    # including both quantities the tiles are sized by.
    tiles["current"] = tiles["plazas_equiv"]
    # ``leader_plazas`` / ``leader_estab`` — the FTE and units the frontier province
    # actually holds — are **not** in the list's schema, which carries only the leader's
    # name, so they are read from the benchmark itself. Supplying them as missing instead
    # would print "Frontier: TUNGURAHUA — 0 FTE across 0 establishments": the figure
    # coerces a blank to zero, turning an absence into a measurement of nobody. Some of
    # these are genuinely thin (C1621's frontier is Pastaza at 99 FTE in a single
    # establishment) and the hover is exactly where that should be visible.
    frontier = leader_benchmark(load_province_rca(), MIN_PLAZAS)
    for column in ("leader_plazas", "leader_estab"):
        tiles[column] = tiles["codigo_clase"].map(frontier[column])
    # Float residue, flattened before it reaches either the figure or the caption. A
    # province that **is** the frontier in an industry has a gap of zero by
    # construction, but `leader_share × T − current` computes it as up to ~1e-13 rather
    # than exactly 0 (Pichincha's C2023 today). The figure drops tiles on `> 0`, so the
    # residue survives as a tile of invisible area *and* escapes the dropped-rows
    # caption, which would then name one fewer industry than the reader cannot see.
    # Treated here rather than in `metrics.potential_gap`, whose output is pinned
    # against the committed CSVs and where the residue is the honest arithmetic.
    tiles["gap"] = tiles["gap"].mask(tiles["gap"].abs() < 1e-6, 0.0)

    st.caption(
        t(
            f"Every industry on {place}'s list as a tile, grouped into its broad "
            "sector. Sized by what it holds **today**, or — using Ecuador as its own "
            "benchmark — by what it would hold at the intensity of the province most "
            "specialized in it, or by the difference. The benchmark is an **existence "
            "proof, not a forecast**: it is activity some Ecuadorian province already "
            "has, and it is a **stock**, not jobs an entrant would create.",
            f"Cada industria de la lista de {place} como un mosaico, agrupada en su "
            "sector amplio. Dimensionada por lo que sostiene **hoy**, o, usando al "
            "Ecuador como su propio referente, por lo que sostendría con la intensidad "
            "de la provincia más especializada en ella, o por la diferencia. El "
            "referente es una **prueba de existencia, no un pronóstico**: es actividad "
            "que alguna provincia ecuatoriana ya tiene, y es un **acervo**, no empleos "
            "que crearía un entrante.",
        )
    )

    pot_cols = st.columns([3, 3, 2])
    with pot_cols[0]:
        coerce_choice("cl-pot-view", list(_POT_VIEWS), _POT_VIEWS, _POT_VIEWS_ES)
        pot_view = (
            st.segmented_control(
                t("Sized by", "Dimensionado por"), list(_POT_VIEWS),
                default=_POT_VIEW_DEFAULT, key="cl-pot-view",
                help=t(_HELP_POT_VIEW_EN, _HELP_POT_VIEW_ES),
                format_func=lambda code: t(
                    _POT_VIEWS.get(code, code), _POT_VIEWS_ES.get(code, code)
                ),
            )
            or _POT_VIEW_DEFAULT
        )
    with pot_cols[1]:
        coerce_choice("cl-pot-color", list(_POT_COLORS), _POT_COLORS, _POT_COLORS_ES)
        pot_color = (
            st.segmented_control(
                t("Colored by", "Coloreado por"), list(_POT_COLORS),
                default=_POT_COLOR_DEFAULT, key="cl-pot-color",
                help=t(_HELP_POT_COLOR_EN, _HELP_POT_COLOR_ES),
                format_func=lambda code: t(
                    _POT_COLORS.get(code, code), _POT_COLORS_ES.get(code, code)
                ),
            )
            or _POT_COLOR_DEFAULT
        )

    dropped = _missing_tiles(tiles, pot_view)
    drawn = tiles.drop(index=dropped.index)
    if drawn.empty:
        st.info(
            t(
                f"No industry on {place}'s list has a measurable "
                f"{potential_view(pot_view)['short'].lower()}.",
                f"Ninguna industria de la lista de {place} tiene "
                f"{potential_view(pot_view)['short'].lower()} medible.",
            )
        )
    else:
        st.plotly_chart(
            make_potential_treemap(
                tiles, pot_view, pot_color, STATE_COLORS, state_labels(),
                GSECTOR_COLORS, place
            ),
            width="stretch",
            key=f"cl-pot-{province}-{var}-{pot_view}-{pot_color}",
        )
        column = POTENTIAL_VIEWS[pot_view]["column"]
        drawn_total = float(pd.to_numeric(drawn[column], errors="coerce").sum())
        measure = potential_view(pot_view)["short"].lower()
        st.caption(
            t(
                f"**{len(drawn)} of {place}'s {len(tiles)} industries** are drawn, "
                f"totalling **{drawn_total:,.0f} FTE** on the *{measure}* measure.",
                f"Se dibujan **{len(drawn)} de las {len(tiles)} industrias** de "
                f"{place}, con un total de **{drawn_total:,.0f} plazas equivalentes** "
                f"en la medida de *{measure}*.",
            )
        )

    # The rows the view cannot draw, counted and — when few enough to name — named.
    # Derived from the frame on screen, never hardcoded.
    if not dropped.empty:
        several = len(dropped) > 1
        why = {
            "potential": t(
                f"no province in Ecuador holds {MIN_PLAZAS:,.0f} FTE in "
                f"{'them' if several else 'it'}, so "
                f"{'they have' if several else 'it has'} no frontier to be priced "
                "against",
                f"ninguna provincia del Ecuador tiene {MIN_PLAZAS:,.0f} plazas "
                f"equivalentes en {'ellas' if several else 'ella'}, así que no "
                f"{'tienen' if several else 'tiene'} frontera contra la que "
                "valorarse",
            ),
            "gap": t(
                f"{'they are' if several else 'it is'} either already at the national "
                "frontier — a gap of zero by construction — or "
                f"{'have' if several else 'has'} no frontier to price against at all",
                f"o bien {'están' if several else 'está'} ya en la frontera nacional, "
                "con una brecha de cero por construcción, o bien no "
                f"{'tienen' if several else 'tiene'} ninguna frontera contra la que "
                "valorarse",
            ),
        }[pot_view]
        named = (
            f" ({', '.join(dropped['display_code'].astype(str).tolist())})"
            if len(dropped) <= 12
            else ""
        )
        st.caption(
            t(
                f"**{len(dropped)} of the {len(tiles)} industries on this list "
                f"{'have' if several else 'has'} no tile in this view**{named}: "
                f"{why}. A treemap has no zero-area tile, so the row is absent rather "
                "than flat — every one of them is still on the list, and in the table "
                "below.",
                f"**{len(dropped)} de las {len(tiles)} industrias de esta lista no "
                f"{'tienen' if several else 'tiene'} mosaico en esta vista**{named}: "
                f"{why}. Un mapa de mosaicos no tiene mosaicos de área cero, así que la "
                "fila está ausente en lugar de plana; todas ellas siguen en la lista y "
                "en la tabla de abajo.",
            )
        )

    # ------------------------------------------------------------------ #
    # The list itself — the page's provenance element.                    #
    # ------------------------------------------------------------------ #
    col = _table_columns()
    table = rows.sort_values("rank_in_province")[list(col)].rename(columns=col)
    # Named for the lenses themselves, matching the sidebar's two group headers.
    table[col["lens"]] = table[col["lens"]].map({
        "top_down": t("Top-down", "Descendente"),
        "bottom_up": t("Bottom-up", "Ascendente"),
    })
    with st.expander(
        t(f"{place}'s {len(rows)} candidate industries, in rank order",
          f"Las {len(rows)} industrias candidatas de {place}, en orden"),
        expanded=False,
    ):
        st.dataframe(
            table, width="stretch", hide_index=True,
            column_config={
                # The top-down rows carry no score on purpose (the two lenses' scores
                # are not comparable). Rendered blank with the explanation in the column
                # tooltip — a NumberColumn cannot print an em-dash for NaN, and casting
                # the column to text to get one would break its numeric sorting.
                col["score"]: st.column_config.NumberColumn(
                    format="%.3f",
                    help=t(
                        "The bottom-up score. Blank on the opportunity rows on purpose "
                        "— the two lenses' scores are not comparable, so no single "
                        "ranking is ever taken across both.",
                        "El puntaje ascendente. En blanco en las filas de oportunidad a "
                        "propósito: los puntajes de los dos enfoques no son comparables, "
                        "así que nunca se hace un solo ordenamiento entre ambos.",
                    ),
                ),
                col["hs4_codes"]: st.column_config.TextColumn(
                    help=t(
                        "Which of the 50 opportunities put this industry here. Empty on "
                        "a top-up, which arrived from the province's own ranking rather "
                        "than from any product.",
                        "Cuál de las 50 oportunidades colocó esta industria aquí. Vacío "
                        "en un complemento, que llegó del propio ordenamiento de la "
                        "provincia y no de ningún producto.",
                    ),
                ),
                col["accessible_market_busd"]: st.column_config.NumberColumn(
                    format="%.2f",
                    help=t(
                        "**Billions of current USD, and a world market — not a forecast "
                        "for this province.** The size of the market the opportunities "
                        "listed beside it could sell into, from the upstream analysis, "
                        "summed where a row names more than one. It says nothing about "
                        "what this province would capture, and nothing about what "
                        "Ecuador ships today: HS 8418 exports $5.3M into a $20.5B "
                        "market. Where a row names two closely related products the sum "
                        "can overstate the distinct market, since their demand "
                        "overlaps. Blank on a top-up, which no opportunity reached.",
                        "**Miles de millones de dólares corrientes, y un mercado "
                        "mundial, no un pronóstico para esta provincia.** El tamaño del "
                        "mercado al que podrían venderle las oportunidades listadas a su "
                        "lado, tomado del análisis previo y sumado cuando una fila "
                        "nombra más de una. No dice nada sobre lo que esta provincia "
                        "capturaría, ni sobre lo que el Ecuador exporta hoy: la partida "
                        "HS 8418 exporta 5,3 millones a un mercado de 20.500 millones. "
                        "Cuando una fila nombra dos productos muy relacionados, la suma "
                        "puede sobrestimar el mercado distinto, porque su demanda se "
                        "traslapa. Vacío en un complemento, al que no llegó ninguna "
                        "oportunidad.",
                    ),
                ),
                col["both_bases"]: st.column_config.CheckboxColumn(
                    help=t(
                        "Does the presence state survive switching from establishments "
                        "to employment? States on this list are establishment-based.",
                        "¿Sobrevive el estado de presencia al cambio de establecimientos "
                        "a empleo? Los estados de esta lista se basan en "
                        "establecimientos.",
                    ),
                ),
            },
        )
        st.download_button(
            t(f"Download {place}'s list (CSV)",
              f"Descargar la lista de {place} (CSV)"),
            data=_downloadable(rows).to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"candidate_industries_{place.lower().replace(' ', '_')}_{var}.csv",
            mime="text/csv",
            key="cl-download",
            help=t(
                "The full derived frame for this province — more columns than the table "
                "above shows; the committed list's schema minus its redundant "
                "current_plazas column.",
                "El marco derivado completo de esta provincia, con más columnas que las "
                "que muestra la tabla de arriba: el esquema de la lista comprometida "
                "menos su columna redundante current_plazas.",
            ),
        )
        st.caption(
            t(
                "Rank runs the top-down rows first (by how many of the 50 opportunities "
                "put the industry here, then by observed presence), then the top-ups by "
                "their own score. **Employment and establishments are the province's "
                "own**; the attractiveness figures behind the score are national.",
                "El orden pone primero las filas descendentes (por cuántas de las 50 "
                "oportunidades colocan la industria aquí, y luego por presencia "
                "observada), y después los complementos por su propio puntaje. **El "
                "empleo y los establecimientos son de la propia provincia**; las cifras "
                "de atractivo detrás del puntaje son nacionales.",
            )
        )

    # All 24 at once, **outside** the expander: the table is province-scoped and
    # collapsed by default, so a reader who wants the whole list has no reason to open it
    # and would otherwise have to pick and download 24 times.
    st.download_button(
        t(f"Download all 24 provinces ({len(frame):,} rows, CSV)",
          f"Descargar las 24 provincias ({len(frame):,} filas, CSV)"),
        data=_downloadable(frame).to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"candidate_industries_all_provinces_{var}.csv",
        mime="text/csv",
        key="cl-download-all",
        help=t(
            "Every province's list in one file, at the attractiveness metric selected "
            "above — the same frame this page is built from.",
            "La lista de todas las provincias en un archivo, con la métrica de "
            "atractivo elegida arriba: el mismo marco con el que se construye esta "
            "página.",
        ),
    )
