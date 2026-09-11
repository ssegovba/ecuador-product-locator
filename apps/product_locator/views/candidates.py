"""Candidate industries × provinces — the bottom-up lens's cross-province summary.

The province explorer answers "what should *this* province do?" one province at a time. This page
replays that same ranking for all 24 at once and lays the union out as an industry × province
heatmap, so the questions the explorer structurally cannot answer become the easy ones:

- **Which provinces is this industry a candidate in?** A row, which no single-province page has.
- **Is it a broad opportunity or a place-specific one?** Row breadth separates the two.
- **Does its role change by province?** Cells are colored by **presence state**, so an industry
  that is a new entry in one province and a deepening play in another shows as a row that changes
  color — measured at 65% of all candidate industries (71 of 109) at the shipped defaults, and
  invisible from inside any one province.

**The page owns two controls and inherits the rest** (2026-08-27). The attractiveness metric and
*min. provinces* are its own; the tradable menu, both eligibility guards, *per state* k and the two
rank-space knobs are read from the **explorer's** widget keys, falling back to
``config.BU_DEFAULT_*`` when the reader has not opened the explorer yet. The grid is the explorer
replayed for all 24 provinces, so a reader who has tuned one province should see that same ranking
generalised rather than a second set of knobs to keep in sync — and the two pages can no longer
show different answers to the same question, which they did before v4 (menu off / k = 5 here
against the builder's menu on / k = 10).

Both eligibility guards therefore apply, from the same implementations the explorer uses
(``metrics.support_eligible`` per cell, ``metrics.connected_eligible`` per industry). The support
guard matters most on this page — a two-establishment "specialization" repeated across a dozen
provinces reads as a broad national opportunity, which is the misreading a grid invites.

It is the sibling of the top-down lens's *All opportunities × provinces*, with two deliberate
differences, both forced by there being no products here (see ``metrics.province_shortlist_matrix``):
cells carry a **state, not a count**, and there are **no blank columns** — every province has a
shortlist by construction, so the sibling's headline reading ("the provinces no ranking reaches")
has no counterpart and is not implied anywhere on this page.

Like the rest of the lens it **consumes precomputed exports and never recomputes** density, RCA or
the labor metrics; the arithmetic is percentile ranks, exactly as on the explorer.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import state
from config import (
    BU_CONTROL_KEYS,
    BU_DEFAULT_BETA,
    BU_DEFAULT_CONNECTED,
    BU_DEFAULT_SUPPORT,
    BU_DEFAULT_TOP_K,
    BU_DEFAULT_TRADABLE,
    BU_DEFAULT_WEIGHT,
    DEFAULT_BOTTOM_UP_PROVINCE,
    MIN_ESTAB,
    MIN_PLAZAS,
    STATE_COLORS,
    THIN_NEIGHBORHOOD_THRESHOLD,
)
from data_loader import (
    load_class_labels,
    load_density_support,
    load_labor_class_employment,
    load_labor_composition,
    load_labor_profile,
    load_province_density,
    load_province_geodata,
    load_tradable_menu,
)
from metrics import connected_eligible, province_shortlist_matrix
from lang import coerce_choice, t
from viz import make_candidate_heatmap, state_labels

# Same registry and wording as the explorer's own controls: a reader arriving from that page must
# not meet a third name for the same knob.
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
    "Average wage paid — what a full-time-equivalent worker earns per year in this industry "
    "(2024 business registry). · Average jobs per firm — how many jobs a typical employing firm "
    "carries. Both are **national** and identical for every province, so this choice moves every "
    "column the same way: the province-specific half of the story is the **feasibility** mix and "
    "the **colors**."
)
_HELP_ATTR_ES = (
    "Salario promedio pagado: lo que gana al año una plaza equivalente en esta industria "
    "(registro de empresas 2024). · Empleo promedio por empresa: cuántos puestos sostiene una "
    "empresa empleadora típica. Ambos son **nacionales** e idénticos para todas las provincias, "
    "así que esta elección mueve todas las columnas por igual: la mitad específica de cada "
    "provincia es la mezcla de **factibilidad** y los **colores**."
)
_HELP_MIN_PROVINCES_EN = (
    "Hide industries that are a candidate in fewer than this many provinces. At 1 (the default) "
    "nothing is hidden — raise it to strip the page down to the broad opportunities, which is worth "
    "doing at a high *per state* setting where the grid grows tall. The caption always reports how "
    "many industries the threshold removed."
)
_HELP_MIN_PROVINCES_ES = (
    "Oculta las industrias que son candidatas en menos provincias que este número. En 1 (el "
    "valor por defecto) no se oculta nada; súbalo para reducir la página a las oportunidades "
    "amplias, lo cual vale la pena con un valor alto de *por estado*, donde la matriz se "
    "alarga. La nota al pie siempre indica cuántas industrias quitó el umbral."
)

_REGION_ORDER = ["Costa", "Sierra", "Amazonia", "Isla"]


def _province_columns(
    geo: pd.DataFrame, present: list[str]
) -> tuple[list[str], dict, dict, list[tuple[str, int]]]:
    """Province columns ordered by region, then abbreviation.

    Grouping the columns by region is the one ordering that makes a geographic pattern legible
    without a map: a row lighting up across the Costa block reads differently from one scattered
    across all four regions.
    """
    lookup = (
        geo[["id_code", "id_name", "id_abbr", "region"]]
        .dropna(subset=["id_code"])
        .drop_duplicates("id_code")
    )
    lookup = lookup[lookup["id_code"].isin(present)].copy()
    lookup["region_rank"] = lookup["region"].apply(
        lambda region: _REGION_ORDER.index(region) if region in _REGION_ORDER else len(_REGION_ORDER)
    )
    lookup = lookup.sort_values(["region_rank", "id_abbr"])
    codes = lookup["id_code"].tolist()
    abbrs = dict(zip(lookup["id_code"], lookup["id_abbr"]))
    names = {
        row["id_code"]: f"{str(row['id_name']).strip()} ({row['region']})"
        for _, row in lookup.iterrows()
    }
    # Block widths in column order, so the figure can rule and label the boundaries.
    groups = [
        (str(region), int(count))
        for region, count in lookup["region"].value_counts().reindex(
            lookup["region"].drop_duplicates()
        ).items()
    ]
    return codes, abbrs, names, groups


# Where this page remembers what it last saw on the explorer. **Not** widget keys, which is
# the whole point — see ``_inherited``.
_SHADOW_PREFIX = "cand-inherited-"

_INHERITED_DEFAULTS = {
    "tradable": BU_DEFAULT_TRADABLE,
    "support": BU_DEFAULT_SUPPORT,
    "connected": BU_DEFAULT_CONNECTED,
    "top_k": BU_DEFAULT_TOP_K,
    "beta": BU_DEFAULT_BETA,
    "weight": BU_DEFAULT_WEIGHT,
}


def _inherited() -> dict[str, object]:
    """The explorer's settings, mirrored so they survive leaving the explorer.

    Three sources, in order: the explorer's **live widget value**, this page's **shadow copy**
    of the last one it saw, then the shipped default from ``config``.

    **Why the shadow is not optional.** Streamlit keeps a widget's ``session_state`` entry for
    exactly *one* script run after the widget stops being instantiated, then garbage-collects
    it. Navigating here from the explorer is that one run — so a naive read works on arrival
    and then silently reverts to the defaults on the reader's very next interaction with this
    page (changing the metric, nudging *min. provinces*). That is the same silent divergence
    between the two pages that v4 set out to remove, reintroduced one rerun later. Keys written
    directly rather than by a widget are never collected, so the shadow is stable for the whole
    session; it is refreshed whenever the reader passes back through the explorer.

    Note the explorer can also render a **partial** set: on the national geography ``'99'`` it
    returns early after the tradable toggle, so Section 3's widgets never appear. Hence one
    shadow per setting rather than one for the group.
    """
    settings: dict[str, object] = {}
    for name, default in _INHERITED_DEFAULTS.items():
        shadow = _SHADOW_PREFIX + name
        live = st.session_state.get(BU_CONTROL_KEYS[name])
        if live is None:
            settings[name] = st.session_state.get(shadow, default)
        else:
            st.session_state[shadow] = live  # seen while the explorer's widget still exists
            settings[name] = live
    return settings


def render() -> None:
    state.init_state()  # ungated on purpose: no product and no province required

    st.title(t("Candidate industries × provinces", "Industrias candidatas × provincias"))
    st.markdown(
        t(
            "The province explorer answers *what should this province do?* — one province at "
            "a time. This page runs the **same ranking for all 24 provinces**, using the "
            "criteria you selected there, and stacks the answers into one grid: every "
            "industry that reaches any province's shortlist gets a row, and each cell is "
            "colored by what that province already has of it. Read **across a row** to see "
            "where an industry is a candidate, and **down a column** for one province's "
            "shortlist.",
            "El explorador por provincia responde *¿qué debería hacer esta provincia?*, una "
            "provincia a la vez. Esta página corre el **mismo ordenamiento para las 24 "
            "provincias**, con los criterios que usted eligió allá, y apila las respuestas en "
            "una sola matriz: cada industria que llega a la lista corta de alguna provincia "
            "recibe una fila, y cada celda se colorea según lo que esa provincia ya tiene de "
            "ella. Lea **a lo largo de una fila** para ver dónde una industria es candidata, y "
            "**hacia abajo en una columna** para ver la lista corta de una provincia.",
        )
    )

    profile = load_labor_profile().copy()
    labels = load_class_labels()
    profile["class_name"] = profile["codigo_clase"].map(labels).fillna(profile["class_name"])
    density = load_province_density()
    comp = load_labor_composition()
    geo = load_province_geodata()

    universe_codes = set(comp.loc[comp["level"] == "class", "code"])
    menu = load_tradable_menu() & universe_codes

    # Two controls only. Everything that defines the *ranking* is inherited from the
    # explorer, which owns those widgets — see the module docstring. The third column is a
    # spacer: without it the metric control and the number input stretch across the page.
    #
    # The inherited values are deliberately **not** restated here. They were, briefly
    # (2026-08-27), as a caption listing all six — visually far too heavy for a line the
    # reader does not act on, and redundant: the intro says the criteria come from the
    # explorer, the grid caption below states the *k* it ran at, and each guard's own derived
    # caption fires when it is off. The closing caption names the explorer as their owner.
    controls = st.columns([3, 1, 4])
    with controls[0]:
        coerce_choice("cand-attr", list(_ATTR_CHOICES), _ATTR_CHOICES, _ATTR_CHOICES_ES)
        var = (
            st.segmented_control(
                t("Attractiveness metric", "Métrica de atractivo"), list(_ATTR_CHOICES),
                default=_ATTR_DEFAULT, key="cand-attr",
                help=t(_HELP_ATTR_EN, _HELP_ATTR_ES),
                format_func=lambda code: t(
                    _ATTR_CHOICES.get(code, code), _ATTR_CHOICES_ES.get(code, code)
                ),
            )
            or _ATTR_DEFAULT
        )
    with controls[1]:
        min_provinces = st.number_input(
            t("Min. provinces", "Mín. provincias"),
            min_value=1, max_value=24, value=1, step=1,
            key="cand-min-prov", help=t(_HELP_MIN_PROVINCES_EN, _HELP_MIN_PROVINCES_ES),
        )
    inherited = _inherited()
    tradable_only = bool(inherited["tradable"])
    support_on = bool(inherited["support"])
    connected_on = bool(inherited["connected"])
    top_k = int(inherited["top_k"])
    beta = float(inherited["beta"])
    weight = float(inherited["weight"])
    # The connectedness guard is a per-**industry** property (φ is national), so it is
    # resolved once here and passed as a class-level exclusion — applied after scoring and
    # before the top-k cut, exactly where the explorer applies it. Built as the set of
    # *ineligible* classes, so a class the industry-space export does not cover is left
    # alone rather than silently dropped (``connected_eligible``'s NaN semantics).
    thin_classes: set[str] = set()
    if connected_on:
        support_frame = load_density_support()
        thin_flag = support_frame["phi_mass_noself"] < THIN_NEIGHBORHOOD_THRESHOLD
        thin_classes = set(
            support_frame.loc[~connected_eligible(thin_flag), "codigo_clase"]
        )

    provinces = sorted(p for p in density["id_code"].unique() if p != "99")
    columns, abbrs, names, groups = _province_columns(geo, provinces)
    result = province_shortlist_matrix(
        profile, density, columns, var,
        beta=float(beta), weight=float(weight), top_k=int(top_k),
        universe=menu if tradable_only else None,
        # One implementation of the guard for both surfaces: this grid and the
        # explorer's Section 3 call the same eligibility rule, so a cell cannot be a
        # candidate here and ineligible there.
        emp=load_labor_class_employment(),
        min_estab=MIN_ESTAB,
        min_plazas=MIN_PLAZAS,
        apply_support=bool(support_on),
        ineligible_classes=thin_classes,
    )
    marks: pd.DataFrame = result["marks"]
    if marks.empty:
        st.info(
            t(
                "No province has a scorable industry under these settings.",
                "Ninguna provincia tiene una industria puntuable con estos parámetros.",
            )
        )
        return

    breadth: pd.Series = result["breadth"]
    state_mix: pd.Series = result["state_mix"]
    matrix: pd.DataFrame = result["state_matrix"]

    # Rows ordered by breadth, then by best score anywhere: the top of the grid is the industries
    # the whole country could plausibly move into, the bottom the genuinely place-specific ones.
    best_score = marks.groupby("codigo_clase")["score"].max()
    order = (
        pd.DataFrame({"breadth": breadth, "best": best_score})
        .sort_values(["breadth", "best"], ascending=[False, False])
    )
    kept = order[order["breadth"] >= int(min_provinces)]
    n_hidden = len(order) - len(kept)
    if kept.empty:
        st.warning(
            t(
                f"No industry is a candidate in {int(min_provinces)} or more provinces at "
                f"these settings — the widest reaches {int(order['breadth'].max())}. Lower "
                "the threshold.",
                f"Ninguna industria es candidata en {int(min_provinces)} provincias o más "
                f"con estos parámetros; la más amplia llega a "
                f"{int(order['breadth'].max())}. Baje el umbral.",
            )
        )
        return
    matrix = matrix.reindex(index=kept.index)

    display_codes = dict(zip(marks["codigo_clase"], marks["display_code"]))
    class_names = dict(zip(marks["codigo_clase"], marks["class_name"].astype(str)))
    row_labels = {
        code: f"{display_codes.get(code, code)} · {class_names.get(code, '')[:38]}"
        for code in matrix.index
    }
    rank_lookup = {
        (row["codigo_clase"], row["id_code"]): int(row["rank_in_state"])
        for _, row in marks.iterrows()
    }
    st.plotly_chart(
        make_candidate_heatmap(
            matrix, STATE_COLORS, state_labels(), row_labels, abbrs, names,
            breadth=breadth.to_dict(), rank_lookup=rank_lookup, column_groups=groups,
        ),
        width="stretch",
        key=(
            f"cand-heatmap-{var}-{tradable_only}-{support_on}-{connected_on}"
            f"-{top_k}-{beta}-{weight}-{min_provinces}"
        ),
    )

    n_marks = len(marks)
    st.caption(
        t(
            f"**{len(matrix):,} industries × {len(columns)} provinces, {n_marks:,} candidate "
            f"marks** — the union of all 24 shortlists at {int(top_k)} per presence state. "
            "Columns are grouped by region (Costa, Sierra, Amazonía, Isla) and rows are "
            "ordered by how many provinces an industry reaches, so the broad opportunities "
            "are at the top and the place-specific ones at the bottom. An empty cell means "
            "the industry did not reach that province's shortlist — not a zero, and not "
            "missing data.",
            f"**{len(matrix):,} industrias × {len(columns)} provincias, {n_marks:,} marcas "
            f"de candidatura**: la unión de las 24 listas cortas con {int(top_k)} por estado "
            "de presencia. Las columnas se agrupan por región (Costa, Sierra, Amazonía, "
            "Insular) y las filas se ordenan por a cuántas provincias llega una industria, "
            "así que las oportunidades amplias quedan arriba y las específicas de un lugar "
            "abajo. Una celda vacía significa que la industria no llegó a la lista corta de "
            "esa provincia: no es un cero ni un dato faltante.",
        )
        + (
            t(
                f" **{n_hidden:,} industries** are hidden by the *min. provinces* threshold "
                f"of {int(min_provinces)}.",
                f" El umbral de *mín. provincias* de {int(min_provinces)} oculta "
                f"**{n_hidden:,} industrias**.",
            )
            if n_hidden
            else ""
        )
    )

    # With the filter off, the rows it *would* remove are on screen — so name the widest
    # of them from the displayed grid rather than asserting a pair in a tooltip. (The
    # tooltip used to claim the two broadest rows overall were temp agencies and uranium
    # mining; measured, uranium is sixth and four in-menu industries are broader. A
    # derived caption cannot go stale that way, and the reader can check it by looking.)
    if not tradable_only:
        off_menu = [code for code in matrix.index if code not in menu]
        if off_menu:
            widest = breadth.reindex(off_menu).dropna().sort_values(ascending=False)
            named = ", ".join(
                f"**{display_codes.get(code, code)} {class_names.get(code, '')[:34].strip()}** "
                f"({int(n)})"
                for code, n in widest.head(3).items()
            )
            st.caption(
                t(
                    f"**{len(off_menu):,} of the {len(matrix):,} industries shown would be "
                    "removed by the tradable focus**, which is **off** on the explorer. The "
                    "widest of them, with the number of provinces each reaches: "
                    f"{named}. That is the argument for the filter — these are rows a "
                    "reader scanning for breadth would otherwise read as national "
                    "opportunities.",
                    f"**El enfoque en transables quitaría {len(off_menu):,} de las "
                    f"{len(matrix):,} industrias mostradas**, y está **desactivado** en el "
                    "explorador. Las más amplias, con el número de provincias que alcanza "
                    f"cada una: {named}. Ese es el argumento a favor del filtro: son filas "
                    "que quien busque amplitud leería como oportunidades nacionales.",
                )
            )

    # With the guard off, the rows it *would* remove are on screen — so name the widest of
    # them from the grid in front of the reader, the same idiom as the tradable caption
    # above. It costs a second matrix build, and only in the non-default state.
    if not support_on:
        gated = province_shortlist_matrix(
            profile, density, columns, var,
            beta=float(beta), weight=float(weight), top_k=int(top_k),
            universe=menu if tradable_only else None,
            emp=load_labor_class_employment(),
            min_estab=MIN_ESTAB, min_plazas=MIN_PLAZAS, apply_support=True,
            ineligible_classes=thin_classes,
        )
        thin_only = [code for code in matrix.index if code not in set(gated["breadth"].index)]
        if thin_only:
            widest = breadth.reindex(thin_only).dropna().sort_values(ascending=False)
            named = ", ".join(
                f"**{display_codes.get(code, code)} {class_names.get(code, '')[:34].strip()}** "
                f"({int(n)})"
                for code, n in widest.head(3).items()
            )
            st.caption(
                t(
                    f"**Support is not required: {len(thin_only):,} of the {len(matrix):,} "
                    "industries shown are candidates *only* because of it** — every province "
                    f"marking them has both under {MIN_ESTAB} establishments and under "
                    f"{MIN_PLAZAS:,.0f} FTE in them. The widest, with the provinces each "
                    f"reaches: {named}. Turn the support guard back on **on the explorer** "
                    "and those rows disappear.",
                    f"**No se exige actividad real: {len(thin_only):,} de las "
                    f"{len(matrix):,} industrias mostradas son candidatas *solo* por eso**; "
                    f"cada provincia que las marca tiene a la vez menos de {MIN_ESTAB} "
                    f"establecimientos y menos de {MIN_PLAZAS:,.0f} plazas equivalentes en "
                    "ellas. Las más amplias, con las provincias que alcanza cada una: "
                    f"{named}. Reactive el filtro de actividad real **en el explorador** y "
                    "esas filas desaparecen.",
                )
            )

    # The finding this page exists for, quantified on whatever the current settings are.
    n_mixed = int((state_mix.reindex(matrix.index).fillna(0) > 1).sum())
    n_single = int((breadth.reindex(matrix.index).fillna(0) == 1).sum())
    st.caption(
        t(
            f"**{n_mixed:,} of the {len(matrix):,} industries shown change presence state "
            "across provinces** — the same industry is a new entry in one place and "
            "something to deepen in another, which is the one reading this grid has and the "
            f"explorer cannot. **{n_single:,} are a candidate in exactly one province**: the "
            "place-specific end of the range. Colors are what the province has of the "
            "industry *today*, not how good the opportunity is — blue and amber are the "
            "diversification story, green is deepening.\n\n"
            "Attractiveness is a **national** figure identical in every column, so a "
            "well-paid industry tends to rank highly wherever its feasibility is decent: "
            "read a wide row as partly a national ranking. The province-specific half of the "
            "story is always the **feasibility** mix and the **colors**.",
            f"**{n_mixed:,} de las {len(matrix):,} industrias mostradas cambian de estado de "
            "presencia entre provincias**: la misma industria es una entrada nueva en un "
            "lugar y algo que profundizar en otro, que es la lectura que solo esta matriz "
            f"permite y el explorador no. **{n_single:,} son candidatas en exactamente una "
            "provincia**, el extremo específico de lugar del rango. Los colores indican lo "
            "que la provincia tiene de la industria *hoy*, no qué tan buena es la "
            "oportunidad: el azul y el ámbar son la historia de diversificación y el verde "
            "es profundización.\n\n"
            "El atractivo es una cifra **nacional** idéntica en todas las columnas, así que "
            "una industria bien pagada tiende a quedar arriba donde su factibilidad sea "
            "decente: lea una fila amplia en parte como un ordenamiento nacional. La mitad "
            "específica de cada provincia es siempre la mezcla de **factibilidad** y los "
            "**colores**.",
        )
    )
    # Per-province totals: the column reading, and the honest note about short columns.
    totals = marks.groupby("id_code")["codigo_clase"].nunique().reindex(columns).fillna(0).astype(int)
    expected = 3 * int(top_k)
    short = totals[totals < expected]
    with st.expander(
        t("How many candidates each province contributes",
          "Cuántas candidatas aporta cada provincia"),
        expanded=False,
    ):
        table = pd.DataFrame({
            t("Province", "Provincia"): [names.get(code, code) for code in columns],
            t("Candidates", "Candidatas"): totals.to_numpy(),
            t("Of which shown", "De las cuales mostradas"): [
                int(matrix[code].notna().sum()) if code in matrix.columns else 0
                for code in columns
            ],
        })
        st.dataframe(table, width="stretch", hide_index=True)
        st.caption(
            t(
                f"A province contributes up to {expected} industries ({int(top_k)} per "
                "state × 3 states). Fewer means a state has fewer than that many scorable "
                "industries — or none at all, which is a finding: a province with no absent "
                "industries has no extensive margin, so diversification there means "
                "deepening what it already has. ",
                f"Una provincia aporta hasta {expected} industrias ({int(top_k)} por estado "
                "× 3 estados). Menos significa que un estado tiene menos industrias "
                "puntuables que eso, o ninguna, lo cual es un hallazgo: una provincia sin "
                "industrias ausentes no tiene margen extensivo, así que diversificar ahí "
                "significa profundizar lo que ya tiene. ",
            )
            + (
                t(
                    f"{len(short)} of {len(columns)} provinces fall short of {expected} "
                    "here.",
                    f"{len(short)} de {len(columns)} provincias no alcanzan {expected} "
                    "aquí.",
                )
                if len(short)
                else t(
                    f"Every province reaches the full {expected} at these settings.",
                    f"Todas las provincias alcanzan las {expected} con estos parámetros.",
                )
            )
        )

    # The bridge back: the grid says which industries and where; the explorer says why.
    default_label = names.get(DEFAULT_BOTTOM_UP_PROVINCE, DEFAULT_BOTTOM_UP_PROVINCE)
    st.caption(
        t(
            "To ask *why* any of these is a candidate — what the province already has that "
            "is closest to it, what is missing, and how much of the score rests on a single "
            "link — open the **Province diversification explorer** and search for it there "
            f"(it opens on {default_label.split(' (')[0]}) — it also owns the settings this "
            "grid is running at. This page deliberately shows no capability panel and marks "
            "no cell with the thin-neighbourhood flag: those are per-industry readings, and "
            "a grid of 1,800 cells is the wrong place for them. The connectedness guard is "
            "still *applied* here — inherited from the explorer, and the same rule — it is "
            "only never displayed per cell.",
            "Para preguntar *por qué* alguna de estas es candidata, es decir qué tiene ya la "
            "provincia que se le parezca más, qué le falta y cuánto del puntaje se apoya en "
            "un solo eslabón, abra el **Explorador por provincia** y búsquela ahí (se abre "
            f"en {default_label.split(' (')[0]}); además es el dueño de los parámetros con "
            "los que corre esta matriz. Esta página deliberadamente no muestra panel de "
            "capacidades ni marca ninguna celda con la señal de vecindario delgado: son "
            "lecturas por industria, y una matriz de 1.800 celdas es el lugar equivocado "
            "para ellas. El filtro de conectividad sí se *aplica* aquí, heredado del "
            "explorador y con la misma regla; solo nunca se muestra celda por celda.",
        )
    )

    # The grid on screen *is* the preview, so no table is repeated here — just the frame
    # behind it, derived live at the settings the explorer owns rather than served from
    # any stored file, so a download always matches what is displayed. The long form,
    # not the matrix: a cell here carries a state and a rank, which one wide grid of
    # colours cannot express in a single value.
    download = marks[marks["codigo_clase"].isin(matrix.index)][[
        "id_code", "codigo_clase", "display_code", "class_name", "gsector_label",
        "state", "score", "x", "rca", "density", "rank_in_state",
    ]].rename(columns={"x": "feasibility", "density": "density_noself"})
    download.insert(1, "id_name", [names.get(code, code) for code in download["id_code"]])
    st.download_button(
        t("Download these candidates (CSV)", "Descargar estas candidatas (CSV)"),
        data=download.sort_values(["id_code", "score"], ascending=[True, False]).to_csv(
            index=False, encoding="utf-8-sig"
        ),
        file_name=f"candidate_industries_by_province_{var}_k{int(top_k)}.csv",
        mime="text/csv",
        key="cand-download",
    )
    st.caption(
        t(
            "Long form — one row per (province, industry) candidate: `id_code`, `id_name`, "
            "`codigo_clase`, `display_code`, `class_name`, `gsector_label`, `state`, "
            "`score`, `feasibility`, `density_noself`, `rca` and `rank_in_state`. Rows are "
            "the industries left after the *min. provinces* threshold, so the file matches "
            "the grid on screen.",
            "Formato largo, una fila por candidata (provincia, industria): `id_code`, "
            "`id_name`, `codigo_clase`, `display_code`, `class_name`, `gsector_label`, "
            "`state`, `score`, `feasibility`, `density_noself`, `rca` y `rank_in_state`. "
            "Las filas son las industrias que quedan tras el umbral de *mín. provincias*, "
            "así que el archivo coincide con la matriz en pantalla.",
        )
    )
