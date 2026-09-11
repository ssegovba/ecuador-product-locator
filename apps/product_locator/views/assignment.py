"""Top-down page B: all 50 opportunities × the provinces the lens assigns, at once.

Page A answers "where could *this* product be made?". This page generalises that
answer: it replays Page A's ranking for all 50 opportunities under the **same
parameters** and counts the result.

    cell(i, p) = # of the 50 products for which industry i is mapped
                 AND province p clears the assignment bar on that product

A product is assigned to **every province whose weighted presence reaches
parity** — the national average in the location-quotient view — rather than to a
fixed number of them, so how many provinces a product reaches is a finding. A row
can carry many marks and a cell can exceed 1, both meaningful ("this industry is
a top pick for three different products here"). **The blank columns are the
point**: they are the provinces no opportunity ever reaches, which is precisely
the motivation for the bottom-up lens.

The grid spans **23 provinces**, not 24: Galápagos is never assigned by this lens
(`config.TD_EXCLUDED_PROVINCES`) and its column is dropped rather than shown
empty, because empty already means the opposite thing here — reached by nothing,
rather than never considered. Its observations still count in every measure the
grid is built from.

The page is **ungated** — it renders fully with no opportunity selected — and
shares Page A's parameter state, so the badge above the heatmap always states
the settings that produced it.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import state
from config import (
    COLOR_BADGE_BG,
    COLOR_BADGE_RULE,
    COLOR_TEXT_BODY,
    COLOR_TEXT_ON_FILL,
    DEFAULT_ROUTE_PRESET,
    MARGIN_COLORS,
    MARGIN_ABBR,
    MARGIN_LABELS,
    MARGIN_LABELS_EN,
    MIN_CROSSWALK_WEIGHT,
    MIN_ESTAB,
    MIN_PLAZAS,
    PRESENCE_CUT_CHOICES,
    ROUTE_PRESET_LABELS,
    ROUTE_PRESET_LABELS_ES,
    ROUTE_PRESETS,
    TD_EXCLUDED_PROVINCES,
    TD_EXCLUSION_NOTE,
    TD_EXCLUSION_NOTE_ES,
)
from data_loader import load_class_labels, load_opportunities, load_province_rca
from metrics import assignment_matrix, parity_threshold, presence_frame
from lang import coerce_choice, t
from viz import (
    make_assignment_heatmap,
    make_opportunity_heatmap,
    presence_bases,
    presence_view,
)

# Product-name budget inside the heatmap hover (see viz._HOVER_LABEL_CHARS).
_HOVER_NAME_CHARS = 34

_FOCUS_PROVINCES = {"CHIMBORAZO", "SANTA ELENA", "COTOPAXI", "TUNGURAHUA", "PASTAZA"}

_HELP_PRODUCT_EN = (
    "Bands this product's industries (rows) and the provinces its ranking assigns it to "
    "(columns), and outlines the cells where the two meet — the marks this one product "
    "contributes to the matrix."
)
_HELP_PRODUCT_ES = (
    "Resalta las industrias de este producto (filas) y las provincias a las que su "
    "ordenamiento lo asigna (columnas), y marca con contorno las celdas donde ambas se "
    "cruzan, es decir las marcas que este producto aporta a la matriz."
)


def _route_labels() -> dict[str, str]:
    """The route presets' display labels in the current language."""
    return {
        code: t(ROUTE_PRESET_LABELS[code], ROUTE_PRESET_LABELS_ES[code])
        for code in ROUTE_PRESETS
    }


def _segmented(label: str, options: list, labels: dict, key: str, fallback):
    """See views/opportunity._segmented: coerce_choice guards the language flip."""
    coerce_choice(key, options, labels)
    choice = st.segmented_control(label, options, format_func=lambda v: labels.get(v, v), key=key)
    return choice if choice is not None else fallback


def render() -> None:
    state.init_state()

    st.title(t("All opportunities × provinces", "Oportunidades × provincias"))
    st.markdown(
        t(
            "The same ranking as the previous page, replayed for **all 50 opportunities** "
            "under the same settings, in two steps. First **where each opportunity lands** "
            "— one row per product, so a column total is simply how many of the 50 that "
            "province wins. Then the same allocation **seen through the industries** those "
            "products need, which is the form that compares with the province lens. Where a "
            "column is blank in both, no opportunity's ranking reaches that province at all "
            "— which is the case for starting from the province instead.",
            "El mismo ordenamiento de la página anterior, repetido para **las 50 "
            "oportunidades** con los mismos parámetros, en dos pasos. Primero **dónde cae "
            "cada oportunidad**: una fila por producto, de modo que el total de una columna "
            "es simplemente cuántas de las 50 gana esa provincia. Luego la misma asignación "
            "**vista a través de las industrias** que esos productos necesitan, que es la "
            "forma comparable con el enfoque de provincia. Donde una columna queda vacía en "
            "ambas, el ordenamiento de ninguna oportunidad llega a esa provincia, y ese es "
            "justamente el argumento para partir de la provincia.",
        )
    )
    st.caption(t(TD_EXCLUSION_NOTE, TD_EXCLUSION_NOTE_ES))

    opps = load_opportunities()
    rca = load_province_rca()
    class_labels = load_class_labels()

    # Controls bound to the same session keys as Page A, so the two stay in sync
    # whichever page the user touches.
    control_columns = st.columns([3, 2, 3, 2])
    with control_columns[0]:
        view = _segmented(
            t("Presence measured as", "Presencia medida como"), ["lq", "scale"],
            {key: str(presence_view(key)["label"]) for key in ("lq", "scale")},
            "td-view", "lq",
        )
    with control_columns[1]:
        base = _segmented(
            t("Counting", "Contando"), ["plazas", "estab"], presence_bases(),
            "td-base", "plazas",
        )
    with control_columns[2]:
        preset = _segmented(
            t("Route weight", "Peso de las rutas"), list(ROUTE_PRESETS), _route_labels(),
            "td-route", DEFAULT_ROUTE_PRESET,
        )
    with control_columns[3]:
        coerce_choice("td-cut", PRESENCE_CUT_CHOICES)
        cut = st.select_slider(
            t("Assign where presence is at least", "Asignar donde la presencia sea al menos"),
            options=PRESENCE_CUT_CHOICES, key="td-cut",
            format_func=lambda v: t(f"x{v:g} parity", f"x{v:g} paridad"),
        )
    guard = st.toggle(
        t(
            f"Require real activity: drop cells with fewer than {MIN_ESTAB} "
            f"establishments **and** fewer than {MIN_PLAZAS:.0f} FTE",
            f"Exigir actividad real: descartar celdas con menos de {MIN_ESTAB} "
            f"establecimientos **y** menos de {MIN_PLAZAS:.0f} plazas equivalentes",
        ),
        key="td-support",
    )
    weight = float(ROUTE_PRESETS.get(preset, ROUTE_PRESETS[DEFAULT_ROUTE_PRESET]))

    frames = presence_frame(rca, view, base, bool(guard), MIN_ESTAB, MIN_PLAZAS)
    # Parity over all 24 provinces, excluded ones included — the excluded province is
    # out of the answer, not out of the measure (see config.TD_EXCLUDED_PROVINCES).
    parity = parity_threshold(view, len(frames['presence'].columns))
    threshold = parity * float(cut)
    result = assignment_matrix(
        opps, frames, weight, threshold, MIN_CROSSWALK_WEIGHT, TD_EXCLUDED_PROVINCES
    )
    # An excluded province comes back as an all-zero column. Drop it here, once, before
    # anything reads the columns: on this page a blank column already *means* something
    # else — "no opportunity's ranking reaches that province at all", the reading the
    # header caption spells out — and leaving Galápagos in would put it in that group
    # under a rule that never considered it.
    excluded_here = [c for c in result["matrix"].columns if c in TD_EXCLUDED_PROVINCES]
    if excluded_here:
        result = dict(result)
        result["matrix"] = result["matrix"].drop(columns=excluded_here)
        result["product_matrix"] = result["product_matrix"].drop(columns=excluded_here)
        result["opportunities_per_province"] = result["opportunities_per_province"].drop(
            index=excluded_here, errors="ignore"
        )
    matrix = result["matrix"]

    # Order rows by broad sector, then code — the project-wide industry reading order.
    meta = rca.drop_duplicates("codigo_clase").set_index("codigo_clase")
    order = (
        pd.DataFrame({"code": matrix.index})
        .assign(gsector=lambda d: d["code"].map(meta["gsector_label"]).fillna(""))
        .sort_values(["gsector", "code"])["code"].tolist()
    )
    matrix = matrix.loc[order]

    # Product selector — empty by default; highlights one product's rows and columns.
    products = (
        opps.drop_duplicates("opportunity_hs4_code")
        .sort_values(["margen", "rank"])[
            ["opportunity_hs4_code", "opportunity_product_name", "margen", "rank"]
        ]
    )
    product_labels = {
        row.opportunity_hs4_code: (
            f"{MARGIN_ABBR[row.margen]} #{row.rank} · "
            f"HS {row.opportunity_hs4_code} — {row.opportunity_product_name}"
        )
        for row in products.itertuples()
    }
    # The same label with the *name* trimmed, for the heatmap hover: plotly has no
    # wrap property, so a 69-character name would stretch the box. Trimming the
    # name rather than the whole label keeps the margin/rank/HS prefix intact.
    hover_labels = {
        row.opportunity_hs4_code: (
            f"{MARGIN_ABBR[row.margen]} #{row.rank} · "
            f"HS {row.opportunity_hs4_code} — {str(row.opportunity_product_name)[:_HOVER_NAME_CHARS]}"
        )
        for row in products.itertuples()
    }
    search_columns = st.columns([14, 1], vertical_alignment="bottom")
    with search_columns[1]:
        if st.button(
            "✕", key="td-heat-clear",
            help=t("Clear the highlighted product", "Quitar el producto resaltado"),
            disabled=st.session_state.get("td-heat-product") is None,
        ):
            st.session_state["td-heat-product"] = None
            st.rerun()
    with search_columns[0]:
        coerce_choice("td-heat-product", products["opportunity_hs4_code"].tolist())
        focus = st.selectbox(
            t("Highlight one opportunity", "Resaltar una oportunidad"),
            options=products["opportunity_hs4_code"].tolist(),
            index=None,
            format_func=lambda c: product_labels.get(c, c),
            placeholder=t("Type an HS code or a product name to search",
                          "Escriba un código HS o el nombre de un producto"),
            key="td-heat-product",
            help=t(_HELP_PRODUCT_EN, _HELP_PRODUCT_ES),
        )

    # The parameter badge: without it the page silently changes meaning
    # depending on where the user has been.
    badge = " · ".join([
        str(presence_view(view)["short"]),
        presence_bases()[base],
        t(
            f"support guard {'on' if guard else 'OFF'}",
            f"filtro de actividad {'activado' if guard else 'DESACTIVADO'}",
        ),
        t(
            f"route {_route_labels()[preset]} (w = {weight:.2f})",
            f"rutas {_route_labels()[preset]} (w = {weight:.2f})",
        ),
        t(f"bar x{cut:g} parity", f"barra x{cut:g} paridad"),
    ])
    st.markdown(
        f"<div style='background:{COLOR_BADGE_BG};border-left:4px solid {COLOR_BADGE_RULE};padding:6px 10px;"
        f"font-size:0.82em;color:{COLOR_TEXT_BODY};margin-bottom:0.4em'><b>"
        + t("Settings behind this matrix:", "Parámetros detrás de esta matriz:")
        + f"</b> {badge}</div>",
        unsafe_allow_html=True,
    )

    geo = rca.drop_duplicates("id_code").set_index("id_code")
    row_labels = {
        code: f"{code} · {(class_labels.get(code) or str(meta.loc[code, 'class_name']))[:44]}"
        for code in matrix.index
    }
    column_labels = {code: str(geo.loc[code, "id_abbr"]) for code in matrix.columns}
    column_names = {code: str(geo.loc[code, "id_name"]).strip() for code in matrix.columns}

    # Cell drill-down (see below): with no product focused, a complete
    # industry + province pair bands that one cell instead.
    pair_industry = st.session_state.get("td-cell-industry")
    pair_province = st.session_state.get("td-cell-province")
    pair_active = bool(not focus and pair_industry and pair_province)
    if focus:
        highlight_rows = result["industries_by_product"].get(focus, [])
        highlight_columns = result["assigned"].get(focus, [])
    elif pair_active:
        highlight_rows, highlight_columns = [pair_industry], [pair_province]
    else:
        highlight_rows, highlight_columns = [], []

    # ------------------------------------------------------------------ #
    # View 1 — where each OPPORTUNITY is allocated. The primitive: the       #
    # industry grid below is this one pushed through the crosswalk.          #
    # ------------------------------------------------------------------ #
    product_matrix = result["product_matrix"]
    won = result["opportunities_per_province"]
    totals_map = {code: int(won.get(code, 0)) for code in product_matrix.columns}
    order_products = products.sort_values(["margen", "rank"])["opportunity_hs4_code"].tolist()
    product_matrix = product_matrix.loc[order_products]

    st.subheader(t("Where does each opportunity land?", "¿Dónde cae cada oportunidad?"))
    st.caption(
        t(
            "One row per opportunity, ordered by margin then priority rank. The number is "
            "the province's **rank** among the provinces that qualified, so the number of "
            "marks in a row is itself a finding — how widely that opportunity's industry "
            "mix is already present — and a **column total is a count of opportunities**.",
            "Una fila por oportunidad, ordenadas por margen y luego por prioridad. El "
            "número es la **posición** de la provincia entre las que calificaron, así que "
            "la cantidad de marcas en una fila es en sí misma un hallazgo (qué tan "
            "extendida está ya la combinación de industrias de esa oportunidad), y el "
            "**total de una columna es un conteo de oportunidades**.",
        )
    )
    st.plotly_chart(
        make_opportunity_heatmap(
            product_matrix,
            {code: product_labels.get(code, code)[:52] for code in product_matrix.index},
            column_labels, column_names, province_totals=totals_map,
            highlight_rows=[focus] if focus else [],
            highlight_columns=result["assigned"].get(focus, []) if focus else [],
        ),
        width="stretch",
        key=f"td-prodheat-{view}-{base}-{guard}-{preset}-{cut}-{focus}",
    )
    ranked_totals = won.sort_values(ascending=False)
    st.caption(
        t("**Opportunities won per province** — ",
          "**Oportunidades ganadas por provincia**: ")
        + " · ".join(
            f"{column_names[code]} {int(value)}"
            for code, value in ranked_totals[ranked_totals > 0].items()
        )
        + t(
            ". This is the headline count: how many of the "
            f"{len(product_matrix.index)} opportunities put that province in their bar.",
            ". Este es el conteo principal: cuántas de las "
            f"{len(product_matrix.index)} oportunidades ponen a esa provincia en su barra.",
        )
    )

    # ------------------------------------------------------------------ #
    # View 2 — the same allocation seen through the industries, which is    #
    # what compares against the bottom-up lens.                             #
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader(t("…and which industries does that put to work?",
                   "…y ¿qué industrias pone eso en juego?"))
    st.caption(
        t(
            "The same allocation, pushed through the HS→ISIC crosswalk: each "
            "opportunity's marks are spread over the industries that make it (and, on the "
            "extensive margin, over the industries behind its anchors). This is the view "
            "that lines up against the **Province diversification explorer**, because it "
            "speaks in industries rather than products. ⚠ A column total here counts "
            "**industry marks, not opportunities** — one opportunity mapping to four "
            "industries contributes four marks, so these totals run roughly double the "
            "counts above.",
            "La misma asignación, pasada por la concordancia entre el Sistema Armonizado y "
            "la CIIU: las marcas de cada oportunidad se reparten entre las industrias que "
            "la elaboran y, en el margen extensivo, entre las industrias detrás de sus "
            "anclas. Esta es la vista que se compara con el **Explorador por provincia**, "
            "porque habla en industrias y no en productos. ⚠ Aquí el total de una columna "
            "cuenta **marcas de industria, no oportunidades**: una oportunidad que se "
            "vincula con cuatro industrias aporta cuatro marcas, así que estos totales son "
            "aproximadamente el doble de los de arriba.",
        )
    )
    st.plotly_chart(
        make_assignment_heatmap(
            matrix, row_labels, column_labels, column_names,
            highlight_rows=highlight_rows, highlight_columns=highlight_columns,
            cell_products=result["cell_products"], product_labels=hover_labels,
            province_totals=totals_map, n_products=len(product_matrix.index),
        ),
        width="stretch",
        key=(
            f"td-heatmap-{view}-{base}-{guard}-{preset}-{cut}-{focus}"
            f"-{pair_industry if pair_active else ''}-{pair_province if pair_active else ''}"
        ),
    )

    if focus:
        margin = str(products.set_index("opportunity_hs4_code").loc[focus, "margen"])
        st.markdown(
            f"<span style='background:{MARGIN_COLORS[margin]};color:{COLOR_TEXT_ON_FILL};padding:2px 9px;"
            f"border-radius:9px;font-size:0.8em'>"
            f"{t(MARGIN_LABELS_EN[margin], MARGIN_LABELS[margin])}</span> "
            f"**HS {focus} · {product_labels[focus].split('— ')[-1]}**"
            + t(
                f" — {len(highlight_rows)} "
                f"industr{'y' if len(highlight_rows) == 1 else 'ies'} × "
                f"{len(highlight_columns)} province"
                f"{'' if len(highlight_columns) == 1 else 's'} "
                f"({', '.join(column_names[c] for c in highlight_columns) or 'none assigned'})"
                f" = {len(highlight_rows) * len(highlight_columns)} outlined cells.",
                f": {len(highlight_rows)} "
                f"industria{'' if len(highlight_rows) == 1 else 's'} × "
                f"{len(highlight_columns)} "
                f"provincia{'' if len(highlight_columns) == 1 else 's'} "
                f"({', '.join(column_names[c] for c in highlight_columns) or 'ninguna asignada'})"
                f" = {len(highlight_rows) * len(highlight_columns)} celdas con contorno.",
            ),
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------ #
    # Cell drill-down. Hover names the opportunities behind a count; this is  #
    # the copyable, touch-friendly counterpart — and the only way to read a   #
    # cell without a mouse. It is not a click handler because it cannot be:   #
    # go.Heatmap is not a selectable plotly trace.                            #
    # ------------------------------------------------------------------ #
    st.markdown(t("**Read one cell**", "**Leer una celda**"))
    cell_columns = st.columns([8, 8, 1], vertical_alignment="bottom")
    with cell_columns[2]:
        if st.button(
            "✕", key="td-cell-clear",
            help=t("Clear the cell selection", "Quitar la selección de celda"),
            disabled=not (pair_industry or pair_province),
        ):
            st.session_state["td-cell-industry"] = None
            st.session_state["td-cell-province"] = None
            st.rerun()
    with cell_columns[0]:
        coerce_choice("td-cell-industry", list(matrix.index))
        chosen_industry = st.selectbox(
            t("Industry", "Industria"),
            options=list(matrix.index),
            index=None,
            format_func=lambda code: row_labels.get(code, code),
            placeholder=t("Type a CIIU code or name",
                          "Escriba un código o nombre CIIU"),
            key="td-cell-industry",
        )
    with cell_columns[1]:
        coerce_choice("td-cell-province", list(matrix.columns))
        chosen_province = st.selectbox(
            t("Province", "Provincia"),
            options=list(matrix.columns),
            index=None,
            format_func=lambda code: column_names.get(code, code),
            placeholder=t("Type a province name",
                          "Escriba el nombre de una provincia"),
            key="td-cell-province",
        )
    if chosen_industry and chosen_province:
        hits = sorted(result["cell_products"].get((chosen_industry, chosen_province), []))
        heading = (
            f"**{row_labels.get(chosen_industry, chosen_industry)}** × "
            f"**{column_names.get(chosen_province, chosen_province)}**"
        )
        if hits:
            st.markdown(
                heading
                + t(
                    f" — **{len(hits)}** of the 50 opportunities put this province in "
                    "their bar *and* map to this industry:",
                    f": **{len(hits)}** de las 50 oportunidades ponen a esta provincia en "
                    "su barra *y* se vinculan con esta industria:",
                )
            )
            rows_by_hs4 = products.set_index("opportunity_hs4_code")
            for hs4 in hits:
                margin = str(rows_by_hs4.loc[hs4, "margen"])
                st.markdown(
                    f"<span style='background:{MARGIN_COLORS[margin]};color:{COLOR_TEXT_ON_FILL};"
                    f"padding:1px 7px;border-radius:9px;font-size:0.72em'>"
                    f"{MARGIN_ABBR[margin]} #{int(rows_by_hs4.loc[hs4, 'rank'])}</span> "
                    f"&nbsp;**HS {hs4}**"
                    + t(" — ", " · ")
                    + f"{rows_by_hs4.loc[hs4, 'opportunity_product_name']}",
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                heading
                + t(
                    " — empty. Either this industry is not mapped to any opportunity that "
                    "reaches this province, or the province does not clear the assignment "
                    "bar on the opportunities that do map to it.",
                    ": vacía. O esta industria no se vincula con ninguna oportunidad que "
                    "llegue a esta provincia, o la provincia no supera la barra de "
                    "asignación en las oportunidades que sí se vinculan con ella.",
                )
            )
        if focus:
            st.caption(
                t(
                    "The heatmap bands are showing the highlighted **opportunity** above; "
                    "clear it to band this cell instead.",
                    "Las franjas del mapa de calor están mostrando la **oportunidad** "
                    "resaltada arriba; quítela para resaltar esta celda en su lugar.",
                )
            )
    elif chosen_industry or chosen_province:
        st.caption(
            t(
                "Pick both an industry and a province to list the opportunities in that "
                "cell.",
                "Elija una industria y una provincia para listar las oportunidades de esa "
                "celda.",
            )
        )

    totals = matrix.sum(axis=0).sort_values(ascending=False)
    nonzero = int((matrix > 0).sum().sum())
    empty = [column_names[c] for c in totals[totals == 0].index]
    st.caption(
        t(
            f"{matrix.shape[0]} industries × {matrix.shape[1]} provinces · "
            f"{nonzero} of {matrix.size} cells carry a mark "
            f"({nonzero / matrix.size:.1%}); the busiest cell serves "
            f"{int(matrix.to_numpy().max())} opportunities. Rows are the industries "
            f"clearing the {MIN_CROSSWALK_WEIGHT:.0%} crosswalk-weight floor — below it an "
            "industry is HS4 aggregation residue that would otherwise earn a mark on a "
            "weight of 0.006%.",
            f"{matrix.shape[0]} industrias × {matrix.shape[1]} provincias · "
            f"{nonzero} de {matrix.size} celdas tienen una marca "
            f"({nonzero / matrix.size:.1%}); la celda más cargada sirve a "
            f"{int(matrix.to_numpy().max())} oportunidades. Las filas son las industrias "
            f"que superan el piso de peso de concordancia de {MIN_CROSSWALK_WEIGHT:.0%}; "
            "por debajo de ese piso una industria es residuo de la agregación a cuatro "
            "dígitos y obtendría una marca con un peso de 0,006%.",
        )
    )
    if empty:
        focus_empty = [name for name in empty if name in _FOCUS_PROVINCES]
        message = t(
            f"**{len(empty)} province{'' if len(empty) == 1 else 's'} receive nothing**: "
            f"{', '.join(empty)}. No opportunity's ranking reaches them — which is the "
            "case for starting from the province instead, in the **Province "
            "diversification explorer**.",
            f"**{len(empty)} provincia{'' if len(empty) == 1 else 's'} no "
            f"recibe{'' if len(empty) == 1 else 'n'} nada**: {', '.join(empty)}. El "
            "ordenamiento de ninguna oportunidad llega a "
            f"{'ella' if len(empty) == 1 else 'ellas'}, y ese es justamente el argumento "
            "para partir de la provincia, en el **Explorador por provincia**.",
        )
        if focus_empty:
            message += t(
                f" {', '.join(focus_empty)} "
                f"{'is a focus province' if len(focus_empty) == 1 else 'are focus provinces'}"
                " for this project.",
                f" {', '.join(focus_empty)} "
                f"{'es una provincia prioritaria' if len(focus_empty) == 1 else 'son provincias prioritarias'}"
                " de este proyecto.",
            )
        if view == "scale":
            message += t(
                " On the **Scale** view emptiness largely reflects province size: a share "
                "of the national total ranks the biggest provinces first almost by "
                "construction.",
                " En la vista de **Escala** el vacío refleja en buena medida el tamaño de "
                "la provincia: una participación en el total nacional ordena primero a las "
                "provincias más grandes casi por construcción.",
            )
        st.caption(message)
    st.caption(
        t("**Industry-mark totals** (not opportunities — see above) — ",
          "**Totales de marcas de industria** (no de oportunidades, ver arriba): ")
        + " · ".join(
            f"{column_names[code]} {int(value)}" for code, value in totals[totals > 0].items()
        )
        + "."
    )

    # The grid on screen *is* the preview, so no table is repeated here — just the
    # frame behind it, derived live at the settings in the badge above rather than
    # served from any stored file, so a download always matches what is displayed.
    download = matrix.rename_axis("codigo_clase").reset_index()
    download.insert(1, "industry", [row_labels.get(code, code) for code in matrix.index])
    st.download_button(
        t("Download this grid (CSV)", "Descargar esta matriz (CSV)"),
        data=download.to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"opportunity_assignment_{view}_{base}_{preset}_cut{cut}.csv",
        mime="text/csv",
        key="td-download",
    )
    st.caption(
        t(
            "One row per industry: `codigo_clase`, `industry`, then one column per "
            "province (2-digit `id_code`) carrying the **industry-mark count** — how many "
            "of the 50 opportunities put that province in their bar *and* map to that "
            "industry.",
            "Una fila por industria: `codigo_clase`, `industry` y luego una columna por "
            "provincia (`id_code` de 2 dígitos) con el **conteo de marcas de industria**, "
            "es decir cuántas de las 50 oportunidades ponen a esa provincia en su barra *y* "
            "se vinculan con esa industria.",
        )
    )
