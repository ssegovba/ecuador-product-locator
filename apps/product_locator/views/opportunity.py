"""Top-down page A: one diversification opportunity, and where it could be made.

Replaces the former 4-page product wizard. An upstream analysis identified **50
concrete opportunities** for Ecuador — 20 intensive margin, 30 extensive — all at
HS4 in the 2022 classification, already scored for feasibility and
attractiveness. Because that set is fixed and pre-vetted, the top-down job
collapses to something much simpler than a wizard: map an opportunity to its
industries, find where those industries are already **observably present**, and
rank the provinces on that.

**Feasibility here is observed presence and nothing else** — no relatedness
density, no proximity, no composite score. The upstream analysis already did the
feasibility × attractiveness work; this page does not redo it.

Like the rest of the app it **consumes precomputed exports and never
recomputes** RCA, shares or employment: `oportunidades/oportunidades_industrias.csv`
(from `utils/build_oportunidades_industrias.py`) and
`industry_space/province_class_rca_multibase.csv`. The arithmetic performed here
is a weighted sum and a weighted average — nothing else.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import state
from config import (
    COLOR_TEXT_MUTED,
    COLOR_TEXT_ON_FILL,
    MARGIN_COLORS,
    MARGIN_LABELS,
    MARGIN_LABELS_EN,
    MATERIAL_SHARE,
    DEFAULT_ROUTE_PRESET,
    MIN_CROSSWALK_WEIGHT,
    MIN_ESTAB,
    MIN_PLAZAS,
    PRESENCE_CUT_CHOICES,
    ROUTE_PRESET_LABELS,
    ROUTE_PRESET_LABELS_ES,
    ROUTE_PRESETS,
    TD_EXCLUDED_MAP_SUFFIX,
    TD_EXCLUDED_MAP_SUFFIX_ES,
    TD_EXCLUDED_PROVINCES,
    TD_EXCLUSION_NOTE,
    TD_EXCLUSION_NOTE_ES,
)
from data_loader import (
    load_class_labels,
    load_opportunities,
    load_province_geojson,
    load_province_rca,
)
from metrics import (
    blend_routes,
    parity_threshold,
    presence_frame,
    product_presence,
    qualifying_provinces,
)
from lang import coerce_choice, t
from viz import (
    fmt_usd,
    make_presence_choropleth,
    make_presence_ranking_bar,
    presence_bases,
    presence_view,
)

_CARDS_PER_ROW = 5

_HELP_VIEW_EN = (
    "**Specialization** — the location quotient: the province's share of its own "
    "employment in the industry divided by the industry's national share, so ×1 is the "
    "national average. It asks *where is this industry disproportionately important*. "
    "· **Scale** — the province's share of the industry's national total: *where is it "
    "biggest*. Specialization is the default because ranking on scale alone mostly "
    "re-discovers the two largest provinces: across these 54 industries Pichincha is "
    "#1 for 30 of them and Guayas for 16, and the #1 province holds a median 48% of "
    "national employment."
)
_HELP_VIEW_ES = (
    "**Especialización**: el cociente de localización, es decir la participación de la "
    "industria en el empleo de la provincia dividida por su participación nacional, de "
    "modo que ×1 es el promedio nacional. Pregunta *dónde es esta industria "
    "desproporcionadamente importante*. · **Escala**: la participación de la provincia "
    "en el total nacional de la industria, es decir *dónde es más grande*. La "
    "especialización es la opción por defecto porque ordenar solo por escala redescubre "
    "sobre todo a las dos provincias más grandes: entre estas 54 industrias Pichincha "
    "ocupa el primer lugar en 30 y Guayas en 16, y la provincia líder concentra una "
    "mediana del 48% del empleo nacional."
)

_HELP_BASE_EN = (
    "**Employment** — registered full-time-equivalent positions, counted where the "
    "establishment operates. · **Establishments** — the number of local units. The two "
    "tell different stories: an industry can hold many positions in few large units, or "
    "few positions across many small ones."
)
_HELP_BASE_ES = (
    "**Empleo**: plazas equivalentes registradas, contabilizadas donde opera el "
    "establecimiento. · **Establecimientos**: el número de unidades locales. Los dos "
    "cuentan historias distintas, porque una industria puede concentrar muchas plazas "
    "en pocas unidades grandes, o pocas plazas repartidas en muchas pequeñas."
)

_HELP_SUPPORT_EN = (
    f"A province × industry cell is dropped when it has **both** fewer than {MIN_ESTAB} "
    f"establishments **and** fewer than {MIN_PLAZAS:.0f} FTE — an AND, so a cell survives "
    "on either enough units or enough workers. 60% of the cells behind these industries "
    "carry fewer than 5 positions, and the largest location quotient built on under 5 "
    "positions is ×12.1; unfiltered, one province ranks 4th nationally for sewing "
    "machines on a single job. Filtered cells contribute zero, and the caption says how "
    "many were removed."
)
_HELP_SUPPORT_ES = (
    f"Una celda provincia × industria se descarta cuando tiene **a la vez** menos de "
    f"{MIN_ESTAB} establecimientos **y** menos de {MIN_PLAZAS:.0f} plazas equivalentes. "
    "Es una conjunción, así que basta con suficientes unidades o suficientes "
    "trabajadores para sobrevivir. El 60% de las celdas detrás de estas industrias "
    "tiene menos de 5 plazas, y el mayor cociente de localización construido sobre "
    "menos de 5 plazas llega a ×12,1: sin filtrar, una provincia aparece cuarta del "
    "país en máquinas de coser por un solo puesto de trabajo. Las celdas filtradas "
    "aportan cero, y la nota al pie indica cuántas se retiraron."
)

_HELP_ROUTE_EN = (
    "How much of the score comes from the industries that make the product itself versus "
    "the industries behind its **anchors** — products Ecuador already exports "
    "competitively that sit close to this one in the product space. Presets rather than a "
    "slider: the weight moves the tail of the ranking far more than its head, so it reads "
    "honestly as a sensitivity check and not as a tuned parameter. Intensive-margin "
    "opportunities have no anchors, so the control does not appear for them."
)
_HELP_ROUTE_ES = (
    "Cuánto del puntaje proviene de las industrias que elaboran el producto mismo "
    "frente a las industrias detrás de sus **anclas**, es decir productos que el "
    "Ecuador ya exporta de forma competitiva y que se ubican cerca de este en el "
    "espacio de productos. Son opciones fijas y no un deslizador: el peso mueve mucho "
    "más la cola del ordenamiento que su cabeza, de manera que se lee honestamente "
    "como una prueba de sensibilidad y no como un parámetro calibrado. Las "
    "oportunidades del margen intensivo no tienen anclas, así que para ellas el "
    "control no aparece."
)

_HELP_CUT_EN = (
    "**The assignment bar, not a count.** A province is assigned this product when its "
    "weighted presence reaches **parity** — the score it would post if nothing about it were "
    "special. In the specialization view parity is ×1, the national average, which makes this "
    "the same *RCA ≥ 1* test used everywhere else in the project, applied to the product's "
    "whole industry mix rather than to one industry. In the scale view parity is an equal "
    "share of the industry across the 24 provinces, because a weighted average of national "
    "shares sums to 1 across all of them and so can never approach 1 for any single one. "
    "How many provinces qualify is therefore a **finding** — 2 to 9 of them, median 4 — not "
    "a number you set. The multiplier tightens or loosens the bar as a sensitivity check; a "
    "province with no observable presence at all is never assigned at any setting."
)
_HELP_CUT_ES = (
    "**Es la barra de asignación, no un conteo.** A una provincia se le asigna este "
    "producto cuando su presencia ponderada alcanza la **paridad**, es decir el "
    "puntaje que registraría si nada en ella fuera especial. En la vista de "
    "especialización la paridad es ×1, el promedio nacional, lo que convierte a esta "
    "en la misma prueba *RCA ≥ 1* que se usa en todo el proyecto, aplicada a la "
    "combinación completa de industrias del producto y no a una sola industria. En la "
    "vista de escala la paridad es una participación igual de la industria entre las "
    "24 provincias, porque un promedio ponderado de participaciones nacionales suma 1 "
    "entre todas ellas y por lo tanto nunca puede acercarse a 1 en una sola. Cuántas "
    "provincias califican es entonces un **hallazgo** (entre 2 y 9, con mediana de 4) "
    "y no una cifra que usted fije. El multiplicador ajusta la barra como prueba de "
    "sensibilidad; una provincia sin ninguna presencia observable no se asigna en "
    "ningún ajuste."
)


def _fmt_int(value: float) -> str:
    return f"{float(value):,.0f}"


def _stack(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concat what is non-empty; an empty frame when nothing is (pd.concat of an
    empty list raises, and both routes are legitimately empty for a product whose
    industries all sit below the crosswalk-weight floor)."""
    parts = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _fmt_presence(value: float, view: str) -> str:
    """A presence value in the units of the current view."""
    return f"{value:.4%}" if view == "scale" else f"{value:.3f}"


def _score_explainer(
    detail: pd.DataFrame,
    plot_df: pd.DataFrame,
    province_name: dict[str, str],
    class_labels: dict[str, str],
    view: str,
    base: str,
    weight: float,
    preset: str,
    has_anchors: bool,
    n_anchors: int,
) -> None:
    """The arithmetic behind one province's score, on the numbers now on screen.

    The province-by-province table shows *what* each score is; a first-time reader
    still cannot see *how* the weights turn into it. This walks one province
    through the three steps — weight, presence, blend — with the real figures for
    the current controls, so the number in the bar chart can be reconstructed by
    hand.
    """
    ranked = plot_df[plot_df["score"] > 0].sort_values("score", ascending=False)
    if detail.empty or ranked.empty:
        return

    with st.expander(
        t(
            "How is this score built? — worked through on one province",
            "¿Cómo se construye este puntaje? Paso a paso en una provincia",
        ),
        expanded=False,
    ):
        options = ranked["id_code"].tolist()
        # An explicit key + coerce_choice: without a key this control's state hangs
        # off an element id that changes with the language, and the restore then
        # fails on a province label formatted in the other one.
        coerce_choice("td-explain-province", options)
        chosen = st.selectbox(
            t("Work through", "Ver el detalle de"),
            options=options,
            index=0,
            key="td-explain-province",
            format_func=lambda code: (
                f"{province_name.get(code, code)}"
                + t(" — score ", ": puntaje ")
                + f"{_fmt_presence(float(ranked.set_index('id_code').loc[code, 'score']), view)}"
            ),
            help=t(
                "Defaults to the top-ranked province. Pick any other to see why it "
                "scores less.",
                "Por defecto es la provincia mejor ubicada. Elija cualquier otra para ver "
                "por qué obtiene menos.",
            ),
        )
        place = province_name.get(chosen, chosen)
        row = ranked.set_index("id_code").loc[chosen]
        cells = detail[detail["id_code"] == chosen]

        measure = (
            t(
                "its **location quotient**: the province's share of its own "
                f"{'employment' if base == 'plazas' else 'establishments'} in that "
                "industry ÷ the industry's national share. ×1 is the national average, "
                "×2 means twice as concentrated here as in Ecuador as a whole",
                "su **cociente de localización**: la participación de esa industria en "
                f"{'el empleo' if base == 'plazas' else 'los establecimientos'} de la "
                "provincia ÷ la participación nacional de la industria. ×1 es el promedio "
                "nacional y ×2 significa el doble de concentración aquí que en el "
                "conjunto del Ecuador",
            )
            if view == "lq"
            else t(
                "the province's **share of that industry's national total** — its size "
                "in the industry, not its specialization",
                "la **participación de la provincia en el total nacional de esa "
                "industria**, es decir su tamaño en la industria y no su especialización",
            )
        )
        st.markdown(
            t(
                "**Step 1 — the product is mapped to industries.** The HS→ISIC crosswalk "
                "splits the product across the industries that make it, with weights "
                "summing to 1",
                "**Paso 1: el producto se vincula con industrias.** La concordancia entre "
                "el Sistema Armonizado y la CIIU reparte el producto entre las industrias "
                "que lo elaboran, con pesos que suman 1",
            )
            + (
                t(
                    f". Its {n_anchors} anchor{'s' if n_anchors != 1 else ''} are mapped "
                    "the same way, and because each anchor carries its own weights "
                    "summing to 1 they are **averaged** — the weights below are already "
                    f"divided by {n_anchors}.",
                    f". Sus {n_anchors} anclas se vinculan de la misma manera y, como cada "
                    "ancla trae sus propios pesos que suman 1, se **promedian**: los pesos "
                    f"de abajo ya están divididos por {n_anchors}.",
                )
                if has_anchors
                else "."
            )
            + t(
                f"\n\n**Step 2 — each industry's presence in {place}** is {measure}.",
                f"\n\n**Paso 2: la presencia de cada industria en {place}** es {measure}.",
            )
            + t(
                "\n\n**Step 3 — multiply and add.** Weight × presence, summed within "
                "each route:",
                "\n\n**Paso 3: multiplicar y sumar.** Peso × presencia, sumado dentro de "
                "cada ruta:",
            )
        )

        blocks = ["producto"] + (["ancla"] if has_anchors else [])
        titles = {
            "producto": t("Industries that make the product",
                          "Industrias que elaboran el producto"),
            "ancla": t("Industries behind the anchors",
                       "Industrias detrás de las anclas"),
        }
        lines: list[str] = []
        for route in blocks:
            part = cells[cells["route"] == route].sort_values("contribution", ascending=False)
            if part.empty:
                continue
            lines.append(f"{titles[route]}:")
            for item in part.itertuples():
                name = (class_labels.get(item.isic4_code) or "").rstrip(".")[:42]
                here = t(
                    f"{item.num_establecimientos:,.0f} estab · "
                    f"{item.plazas_equiv:,.0f} FTE here",
                    f"{item.num_establecimientos:,.0f} estab. · "
                    f"{item.plazas_equiv:,.0f} plazas aquí",
                )
                if item.presence_used > 0:
                    note = f"  ({here})"
                elif item.presence_raw > 0:
                    note = t(
                        f"  ← removed by the support guard: only {here}, so its "
                        f"{_fmt_presence(item.presence_raw, view)} does not count",
                        f"  ← retirada por el filtro de actividad real: solo {here}, así "
                        f"que su {_fmt_presence(item.presence_raw, view)} no cuenta",
                    )
                elif item.num_establecimientos > 0 or item.plazas_equiv > 0:
                    note = t(
                        "  ← registered here but reports no "
                        f"{'FTE' if base == 'plazas' else 'units'} ({here})",
                        "  ← registrada aquí pero no reporta "
                        f"{'plazas' if base == 'plazas' else 'unidades'} ({here})",
                    )
                else:
                    note = t("  ← absent from this province",
                             "  ← ausente en esta provincia")
                lines.append(
                    f"  {item.weight:8.4f} × {_fmt_presence(item.presence_used, view):>9}"
                    f" = {_fmt_presence(item.contribution, view):>9}   "
                    f"{item.isic4_code} {name}{note}"
                )
            subtotal = float(part["contribution"].sum())
            label = (
                t("direct route", "ruta directa")
                if route == "producto"
                else t("anchor route", "ruta de anclas")
            )
            lines.append(f"  {'':>8}   {'':>9}   {'─' * 9}")
            lines.append(f"  {'':>8}   {'':>9}   {_fmt_presence(subtotal, view):>9}   = {label}")
            lines.append("")
        st.code("\n".join(lines), language=None)

        if has_anchors:
            preset_label = _route_labels()[preset]
            st.markdown(
                t(
                    f"**Step 4 — blend the two routes** with the *{preset_label}* preset "
                    f"(w = {weight:.2f}):\n\n"
                    f"`score = {weight:.2f} × {_fmt_presence(float(row['direct']), view)} "
                    f"+ {1 - weight:.2f} × {_fmt_presence(float(row['anchor']), view)} "
                    f"= {_fmt_presence(float(row['score']), view)}`\n\n"
                    f"That is {place}'s bar in the ranking below.",
                    f"**Paso 4: combinar las dos rutas** con la opción *{preset_label}* "
                    f"(w = {weight:.2f}):\n\n"
                    f"`puntaje = {weight:.2f} × {_fmt_presence(float(row['direct']), view)} "
                    f"+ {1 - weight:.2f} × {_fmt_presence(float(row['anchor']), view)} "
                    f"= {_fmt_presence(float(row['score']), view)}`\n\n"
                    f"Esa es la barra de {place} en el ordenamiento de abajo.",
                )
            )
        else:
            st.markdown(
                t(
                    "This opportunity has no anchors, so the direct route **is** the "
                    f"score: `{_fmt_presence(float(row['score']), view)}` — {place}'s bar "
                    "in the ranking below.",
                    "Esta oportunidad no tiene anclas, así que la ruta directa **es** el "
                    f"puntaje: `{_fmt_presence(float(row['score']), view)}`, la barra de "
                    f"{place} en el ordenamiento de abajo.",
                )
            )
        st.caption(
            t(
                "An industry that is absent, or that the support guard removed, "
                "contributes **0** and the remaining weights are **not** rescaled to "
                "compensate — so a province missing one of the product's industries "
                "scores lower, which is the intended reading. Every number here follows "
                "the controls above; change one and this walk-through changes with it.",
                "Una industria ausente, o retirada por el filtro de actividad real, aporta "
                "**0**, y los pesos restantes **no** se reescalan para compensar. Por eso "
                "una provincia a la que le falta una de las industrias del producto "
                "obtiene menos puntaje, que es la lectura buscada. Todas las cifras de "
                "aquí siguen los controles de arriba: cambie uno y este recorrido cambia "
                "con él.",
            )
        )


def _route_labels() -> dict[str, str]:
    """The route presets' display labels in the current language.

    The presets are keyed by code (see config.ROUTE_PRESETS), so the label is a
    pure display concern and can be resolved per run without touching the value
    persisted in session state.
    """
    return {
        code: t(ROUTE_PRESET_LABELS[code], ROUTE_PRESET_LABELS_ES[code])
        for code in ROUTE_PRESETS
    }


def _segmented(label: str, options: list, labels: dict, key: str, help_text: str, fallback):
    """Segmented control bound directly to a shared session key (no `default=`:
    the key is pre-seeded by state.init_state, and passing both warns).

    ``coerce_choice`` first: this control's label and its option labels both
    translate, so a language flip can leave the key holding a display label
    rather than a code (see ``lang.coerce_choice``).
    """
    coerce_choice(key, options, labels)
    choice = st.segmented_control(
        label, options, format_func=lambda v: labels.get(v, v), key=key, help=help_text
    )
    return choice if choice is not None else fallback


def _opportunity_cards(
    opps: pd.DataFrame, selected: str | None, collapsible: bool = True
) -> None:
    """The 50-card picker, grouped by margin and colored by margin only.

    ``collapsible`` shows only the first row of each margin and puts the rest
    behind a per-margin expander, so opening the app does not dump 50 products on
    the reader. It is turned **off** when the picker is itself rendered inside the
    "choose a different opportunity" expander — Streamlit cannot nest expanders,
    and by then the user has explicitly asked to browse the full list.
    """
    products = (
        opps[opps["route"] == "producto"]
        .groupby(
            ["margen", "rank", "opportunity_hs4_code", "opportunity_product_name",
             "opportunity_sector"],
            as_index=False,
        )
        .agg(
            # The allocation weights sum to 1 within the producto route, so this
            # total reconstructs Ecuador's own exports of the HS4 exactly.
            exports=("export_value_allocated", "sum"),
            market=("opportunity_accessible_market_busd", "first"),
        )
    )

    def draw(chunk: pd.DataFrame, color: str) -> None:
        columns = st.columns(_CARDS_PER_ROW)
        for column, item in zip(columns, chunk.itertuples()):
            with column, st.container(border=True):
                is_selected = item.opportunity_hs4_code == selected
                st.markdown(
                    f"<span style='background:{color};color:{COLOR_TEXT_ON_FILL};padding:1px 7px;"
                    f"border-radius:9px;font-size:0.70em;white-space:nowrap'>"
                    f"#{item.rank}</span> "
                    f"<span style='font-size:0.74em;color:{COLOR_TEXT_MUTED}'>HS {item.opportunity_hs4_code}"
                    f"</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='font-size:0.86em;line-height:1.25;min-height:3.4em;"
                    f"margin-top:0.25em'><b>{item.opportunity_product_name}</b></div>",
                    unsafe_allow_html=True,
                )
                # Two different quantities, deliberately side by side: what
                # Ecuador ships today, and how big the market it could ship into
                # is. The second is the reason the opportunity is on the list.
                market = (
                    t(f" · market ${item.market:,.1f}B",
                      f" · mercado ${item.market:,.1f}\u00a0mil\u00a0M")
                    if pd.notna(item.market)
                    else ""
                )
                st.markdown(
                    f"<div style='font-size:0.72em;color:{COLOR_TEXT_MUTED};margin-bottom:0.4em'>"
                    f"{item.opportunity_sector} · "
                    + t("exports ", "exportaciones ")
                    + f"{fmt_usd(item.exports)}{market}</div>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    t("✓ Selected", "✓ Seleccionada")
                    if is_selected
                    else t("Select", "Seleccionar"),
                    key=f"td-card-{item.opportunity_hs4_code}",
                    type="primary" if is_selected else "secondary",
                    disabled=is_selected,
                    width="stretch",
                ):
                    st.session_state["opportunity_hs4"] = item.opportunity_hs4_code
                    st.rerun()

    for margin in ["intensivo", "extensivo"]:
        block = products[products["margen"] == margin].sort_values("rank")
        color = MARGIN_COLORS[margin]
        st.markdown(
            f"<div style='margin:0.6em 0 0.2em 0;font-weight:600'>"
            f"<span style='color:{color}'>■</span> "
            f"{t(MARGIN_LABELS_EN[margin], MARGIN_LABELS[margin])} "
            f"<span style='font-weight:400;color:{COLOR_TEXT_MUTED}'>· "
            + t(f"{len(block)} opportunities, ranked by priority",
                f"{len(block)} oportunidades, ordenadas por prioridad")
            + "</span></div>",
            unsafe_allow_html=True,
        )
        chunks = [block.iloc[i:i + _CARDS_PER_ROW] for i in range(0, len(block), _CARDS_PER_ROW)]
        if not collapsible:
            for chunk in chunks:
                draw(chunk, color)
            continue
        draw(chunks[0], color)
        rest = chunks[1:]
        if not rest:
            continue
        hidden = int(sum(len(chunk) for chunk in rest))
        first_hidden = int(rest[0]["rank"].min())
        last_hidden = int(rest[-1]["rank"].max())
        with st.expander(
            t(f"Show the other {hidden} — #{first_hidden}–#{last_hidden}",
              f"Ver las otras {hidden}: #{first_hidden} a #{last_hidden}"),
            expanded=False,
        ):
            for chunk in rest:
                draw(chunk, color)


# Displayed column headers. Functions rather than constants because they are
# resolved per run: a module-level constant would freeze at import in whichever
# language was current then. They name *display* columns only -- nothing keyed on
# them leaves the page, and the download paths build their own frames.
def _COL_INDUSTRY() -> str:
    return t("Industry", "Industria")


def _COL_WEIGHT() -> str:
    return t("Weight", "Peso")


def _COL_COUNTED() -> str:
    return t("Counted", "Contada")


def _COL_FTE_NAT() -> str:
    return t("FTE nationally", "Plazas equiv. (nacional)")


def _COL_ESTAB_NAT() -> str:
    return t("Establishments nationally", "Establecimientos (nacional)")


def _industry_table(industries: pd.DataFrame, class_labels: dict[str, str]) -> pd.DataFrame:
    """One route's industries, ready for st.dataframe."""
    table = industries.copy()
    names = [
        class_labels.get(code) or str(fallback)
        for code, fallback in zip(table["isic4_code"], table["isic4_name"])
    ]
    table[_COL_INDUSTRY()] = (
        table["isic4_code"] + " · " + pd.Series(names, index=table.index)
    )
    table[_COL_COUNTED()] = ~table["below_floor"]
    return table.rename(
        columns={
            "crosswalk_weight_used": _COL_WEIGHT(),
            "national_plazas": _COL_FTE_NAT(),
            "national_estab": _COL_ESTAB_NAT(),
        }
    )[[_COL_INDUSTRY(), _COL_WEIGHT(), _COL_COUNTED(), _COL_FTE_NAT(), _COL_ESTAB_NAT()]]


def _industry_column_config() -> dict:
    """Column config for the industry tables, keyed by the current-language headers."""
    return {
        _COL_WEIGHT(): st.column_config.NumberColumn(
            format="%.3f",
            help=t(
                "Share of the product allocated to this industry by the HS→ISIC "
                "crosswalk — Ecuador's own within-HS4 export mix where it clears the "
                "$1M support floor, otherwise the US reference. Sums to 1 within a "
                "route.",
                "Parte del producto asignada a esta industria por la concordancia entre "
                "el Sistema Armonizado y la CIIU: la propia composición exportadora del "
                "Ecuador dentro de la partida cuando supera el piso de soporte de 1 "
                "millón de dólares, y la referencia de Estados Unidos en caso "
                "contrario. Suma 1 dentro de cada ruta.",
            ),
        ),
        _COL_COUNTED(): st.column_config.CheckboxColumn(
            help=t(
                f"Industries below a crosswalk weight of {MIN_CROSSWALK_WEIGHT:.0%} are "
                "HS4 aggregation residue (aircraft manufacturing turns up for seats at a "
                "weight of 0.006%). They are shown with their weight rather than hidden, "
                "but contribute nothing to the ranking.",
                f"Las industrias por debajo de un peso de concordancia de "
                f"{MIN_CROSSWALK_WEIGHT:.0%} son residuo de la agregación a cuatro "
                "dígitos (la fabricación de aeronaves aparece por los asientos con un "
                "peso de 0,006%). Se muestran con su peso en lugar de ocultarse, pero no "
                "aportan nada al ordenamiento.",
            ),
        ),
        _COL_FTE_NAT(): st.column_config.NumberColumn(format="%.0f"),
        _COL_ESTAB_NAT(): st.column_config.NumberColumn(format="%.0f"),
    }


def render() -> None:
    state.init_state()

    st.title(t("Opportunity & provinces", "Oportunidad y provincias"))
    st.markdown(
        t(
            "Fifty diversification opportunities have been identified for Ecuador — 20 on "
            "the **intensive margin** (products it already exports competitively and could "
            "deepen) and 30 on the **extensive margin** (genuinely new products). Pick one "
            "and this page maps it to the industries that make it, then ranks the "
            "provinces by how much of those industries is **already there**.",
            "Se han identificado cincuenta oportunidades de diversificación para el "
            "Ecuador: 20 en el **margen intensivo** (productos que ya exporta de forma "
            "competitiva y podría profundizar) y 30 en el **margen extensivo** (productos "
            "genuinamente nuevos). Elija una y esta página la vincula con las industrias "
            "que la elaboran, y luego ordena las provincias según cuánto de esas "
            "industrias **ya está ahí**.",
        )
    )

    opps = load_opportunities()
    rca = load_province_rca()
    class_labels = load_class_labels()
    selected = st.session_state.get("opportunity_hs4")

    if selected is None:
        _opportunity_cards(opps, selected)
        st.info(
            t(
                "Pick an opportunity above to see where in Ecuador it could be produced.",
                "Elija una oportunidad arriba para ver dónde en el Ecuador podría "
                "producirse.",
            )
        )
        return

    header = opps[opps["opportunity_hs4_code"] == selected].iloc[0]
    margin = str(header["margen"])
    title_columns = st.columns([18, 1], vertical_alignment="center")
    with title_columns[0]:
        st.markdown(
            f"<span style='background:{MARGIN_COLORS[margin]};color:{COLOR_TEXT_ON_FILL};padding:2px 9px;"
            f"border-radius:9px;font-size:0.8em'>"
            f"{t(MARGIN_LABELS_EN[margin], MARGIN_LABELS[margin])} · "
            f"#{int(header['rank'])}"
            f"</span>&nbsp;&nbsp;<b style='font-size:1.15em'>HS {selected} · "
            f"{header['opportunity_product_name']}</b>"
            f"<span style='color:{COLOR_TEXT_MUTED}'>"
            + t(" — ", " · ")
            + f"{header['opportunity_sector']}</span>",
            unsafe_allow_html=True,
        )
    with title_columns[1]:
        if st.button(
            "✕",
            key="td-clear",
            help=t("Clear the selected opportunity",
                   "Quitar la oportunidad seleccionada"),
        ):
            state.clear_opportunity()
            st.rerun()
    with st.expander(
        t("Choose a different opportunity", "Elegir otra oportunidad"), expanded=False
    ):
        # Not collapsible in here: Streamlit cannot nest expanders, and opening
        # this one is already the "show me everything" gesture.
        _opportunity_cards(opps, selected, collapsible=False)

    # ------------------------------------------------------------------ #
    # Controls — one set, driving every figure on the page.               #
    # ------------------------------------------------------------------ #
    st.divider()
    has_anchors = bool(((opps["opportunity_hs4_code"] == selected) & (opps["route"] == "ancla")).any())
    control_columns = st.columns([3, 2, 3, 2] if has_anchors else [3, 2, 2, 3])
    with control_columns[0]:
        view = _segmented(
            t("Presence measured as", "Presencia medida como"),
            ["lq", "scale"],
            {key: str(presence_view(key)["label"]) for key in ("lq", "scale")},
            "td-view", t(_HELP_VIEW_EN, _HELP_VIEW_ES), "lq",
        )
    with control_columns[1]:
        base = _segmented(
            t("Counting", "Contando"),
            ["plazas", "estab"],
            presence_bases(),
            "td-base",
            t(_HELP_BASE_EN, _HELP_BASE_ES),
            "plazas",
        )
    if has_anchors:
        with control_columns[2]:
            preset = _segmented(
                t("Route weight", "Peso de las rutas"),
                list(ROUTE_PRESETS), _route_labels(),
                "td-route", t(_HELP_ROUTE_EN, _HELP_ROUTE_ES), DEFAULT_ROUTE_PRESET,
            )
    else:
        preset = st.session_state.get("td-route") or DEFAULT_ROUTE_PRESET
    with control_columns[3 if has_anchors else 2]:
        coerce_choice("td-cut", PRESENCE_CUT_CHOICES)
        cut = st.select_slider(
            t("Assign where presence is at least", "Asignar donde la presencia sea al menos"),
            options=PRESENCE_CUT_CHOICES, key="td-cut",
            format_func=lambda v: t(f"x{v:g} parity", f"x{v:g} paridad"),
            help=t(_HELP_CUT_EN, _HELP_CUT_ES),
        )
    guard = st.toggle(
        t(
            f"Require real activity: drop cells with fewer than {MIN_ESTAB} "
            f"establishments **and** fewer than {MIN_PLAZAS:.0f} FTE",
            f"Exigir actividad real: descartar celdas con menos de {MIN_ESTAB} "
            f"establecimientos **y** menos de {MIN_PLAZAS:.0f} plazas equivalentes",
        ),
        key="td-support",
        help=t(_HELP_SUPPORT_EN, _HELP_SUPPORT_ES),
    )
    weight = float(ROUTE_PRESETS.get(preset, ROUTE_PRESETS[DEFAULT_ROUTE_PRESET]))

    frames = presence_frame(rca, view, base, bool(guard), MIN_ESTAB, MIN_PLAZAS)
    direct = product_presence(opps, frames, selected, "producto", MIN_CROSSWALK_WEIGHT, MATERIAL_SHARE)
    anchor = product_presence(opps, frames, selected, "ancla", MIN_CROSSWALK_WEIGHT, MATERIAL_SHARE)
    blend = (
        blend_routes(direct["presence"], anchor["presence"], weight)
        if has_anchors
        else {"score": direct["presence"], "n_observable_direct": direct["n_observable"],
              "n_observable_anchor": 0, "effective_w": 1.0}
    )
    score = blend["score"]

    # ------------------------------------------------------------------ #
    # Section 2 — the industries this opportunity maps to.                #
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader(t("Which industries make this product?",
                   "¿Qué industrias elaboran este producto?"))

    st.markdown(t("**Industries that make the product itself**",
                  "**Industrias que elaboran el producto mismo**"))
    st.dataframe(
        _industry_table(direct["industries"], class_labels),
        width="stretch", hide_index=True, column_config=_industry_column_config(),
    )

    if has_anchors:
        st.markdown(t("**Industries behind its anchors**",
                      "**Industrias detrás de sus anclas**"))
        st.caption(
            t(
                "An **anchor** is a product Ecuador already exports competitively that "
                "sits close to this one in the product space but in a sparse neighbourhood "
                "of it — so the jump leans on capabilities the country demonstrably has, "
                "rather than on the low-complexity outskirts. ⚠ That relatedness was "
                "established at the **national** level, on Ecuador's product space and its "
                "national exports. Reading it province by province assumes the relation "
                "also holds locally; there is no provincial evidence for it here.",
                "Un **ancla** es un producto que el Ecuador ya exporta de forma competitiva "
                "y que se ubica cerca de este en el espacio de productos, pero en un "
                "vecindario poco denso, de modo que el salto se apoya en capacidades que el "
                "país demostrablemente tiene y no en la periferia de baja complejidad. ⚠ "
                "Esa relación se estableció a nivel **nacional**, sobre el espacio de "
                "productos del Ecuador y sus exportaciones nacionales. Leerla provincia por "
                "provincia supone que la relación también se sostiene localmente, y aquí no "
                "hay evidencia provincial de ello.",
            )
        )
        anchors = anchor["industries"]
        for anchor_code, group in anchors.groupby("anchor_hs4_code"):
            first = group.iloc[0]
            st.markdown(
                f"↳ **HS {anchor_code} · {first['anchor_product_name']}**"
                + t(
                    f" — proximity to this opportunity "
                    f"{first['anchor_to_candidate_proximity']:.3f}, density percentile "
                    f"{first['anchor_density_percentile']:.1f}",
                    f": proximidad a esta oportunidad "
                    f"{first['anchor_to_candidate_proximity']:.3f}, percentil de densidad "
                    f"{first['anchor_density_percentile']:.1f}",
                )
            )
            st.dataframe(
                _industry_table(group, class_labels),
                width="stretch", hide_index=True, column_config=_industry_column_config(),
                key=f"td-anchor-table-{selected}-{anchor_code}",
            )
        st.caption(
            t(
                f"The {anchor['n_anchors']} anchors are **averaged**, not added: each "
                "carries its own weights summing to 1, so adding them would multiply the "
                "anchor route by the number of anchors — bookkeeping, not evidence.",
                f"Las {anchor['n_anchors']} anclas se **promedian**, no se suman: cada una "
                "trae sus propios pesos que suman 1, así que sumarlas multiplicaría la ruta "
                "de anclas por el número de anclas, lo cual sería contabilidad y no "
                "evidencia.",
            )
        )

    # An industry reached by BOTH routes. This is NOT double counting — the blend
    # is a convex combination whose weights total 1 (verified across all 30
    # anchored opportunities), so such an industry takes a larger share of a fixed
    # budget rather than adding mass. What it does cost is *independence*: the two
    # routes stop being two channels of evidence, and the route-weight preset
    # moves the ranking less than it appears to. That is worth saying out loud.
    if has_anchors and not direct["industries"].empty and not anchor["industries"].empty:
        direct_w = (
            direct["industries"].drop_duplicates("isic4_code")
            .set_index("isic4_code")["effective_weight"]
        )
        anchor_w = (
            anchor["industries"].drop_duplicates("isic4_code")
            .set_index("isic4_code")["effective_weight"]
        )
        shared = [c for c in direct_w.index if c in anchor_w.index and anchor_w[c] > 0 and direct_w[c] > 0]
        if shared:
            pooled = weight * direct_w.reindex(
                direct_w.index.union(anchor_w.index)
            ).fillna(0.0) + (1 - weight) * anchor_w.reindex(
                direct_w.index.union(anchor_w.index)
            ).fillna(0.0)
            total = float(pooled.sum())
            shared_share = float(pooled.reindex(shared).sum()) / total if total else 0.0
            parts = [
                f"**{code} {(class_labels.get(code) or '')[:40].rstrip('.')}**"
                + t(
                    f" — {direct_w[code]:.3f} directly and {anchor_w[code]:.3f} through an "
                    f"anchor, so {weight:.2f}×{direct_w[code]:.3f} + "
                    f"{1 - weight:.2f}×{anchor_w[code]:.3f} = "
                    f"**{float(pooled[code]):.3f}** of the score",
                    f": {direct_w[code]:.3f} de forma directa y {anchor_w[code]:.3f} a "
                    f"través de un ancla, así que {weight:.2f}×{direct_w[code]:.3f} + "
                    f"{1 - weight:.2f}×{anchor_w[code]:.3f} = "
                    f"**{float(pooled[code]):.3f}** del puntaje",
                )
                for code in shared
            ]
            message = (
                t("↔ **The same industry is reached by both routes.** ",
                  "↔ **La misma industria se alcanza por las dos rutas.** ")
                + "; ".join(parts)
                + t(
                    ". It is not counted twice — the two routes are averaged, not added, "
                    "and their weights still total 1, so it simply takes a larger share of "
                    "that budget.",
                    ". No se cuenta dos veces: las dos rutas se promedian y no se suman, y "
                    "sus pesos siguen totalizando 1, así que simplemente toma una porción "
                    "mayor de ese presupuesto.",
                )
            )
            if shared_share >= 0.5:
                message += t(
                    f" ⚠ Because it carries **{shared_share:.0%}** of the combined weight, "
                    "the two routes are largely the *same* evidence for this opportunity — "
                    "expect the route-weight preset to move the ranking very little.",
                    f" ⚠ Como concentra el **{shared_share:.0%}** del peso combinado, las "
                    "dos rutas son en buena medida la *misma* evidencia para esta "
                    "oportunidad: espere que el peso de las rutas mueva muy poco el "
                    "ordenamiento.",
                )
            st.info(message)

    below = _stack(direct["industries"], anchor["industries"]) if has_anchors else direct["industries"]
    n_below = int(below["below_floor"].sum()) if not below.empty else 0
    if n_below:
        st.caption(
            t(
                f"{n_below} industr{'y is' if n_below == 1 else 'ies are'} below the "
                f"{MIN_CROSSWALK_WEIGHT:.0%} crosswalk-weight floor (unticked above): shown "
                "with their weight, but not counted — at HS4 a product picks up traces of "
                "industries that do not make it.",
                f"{n_below} industria{'' if n_below == 1 else 's'} "
                f"{'está' if n_below == 1 else 'están'} por debajo del piso de peso de "
                f"concordancia de {MIN_CROSSWALK_WEIGHT:.0%} (sin marcar arriba): se "
                "muestran con su peso pero no se cuentan, porque a cuatro dígitos un "
                "producto arrastra rastros de industrias que no lo elaboran.",
            )
        )

    # ------------------------------------------------------------------ #
    # Section 3 — where those industries already are.                     #
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader(t("Where are those industries already present?",
                   "¿Dónde están ya presentes esas industrias?"))

    provinces = list(frames["presence"].columns)
    geo = rca.drop_duplicates("id_code").set_index("id_code")
    province_name = {code: str(geo.loc[code, "id_name"]).strip() for code in provinces}
    detail = _stack(direct["detail"], anchor["detail"]) if has_anchors else direct["detail"]
    # An industry can be reached by BOTH routes of the same opportunity (its own
    # and one of its anchors'). Presence, employment, establishments and the
    # support guard are properties of a (province, industry) cell, not of a
    # route — so everything counted per cell is deduplicated first.
    cells = detail.drop_duplicates(["id_code", "isic4_code"]) if not detail.empty else detail
    observable = (
        detail[detail["presence_used"] > 0].groupby("id_code")["isic4_code"].nunique()
        if not detail.empty
        else pd.Series(dtype=int)
    )
    plot_df = pd.DataFrame({
        "id_code": provinces,
        "id_name": [province_name[code] for code in provinces],
        "score": score.reindex(provinces).to_numpy(),
        "direct": direct["presence"].reindex(provinces).to_numpy(),
        "anchor": anchor["presence"].reindex(provinces).to_numpy(),
        "n_industries": observable.reindex(provinces).fillna(0).astype(int).to_numpy(),
    })
    size = frames["plazas"].reindex(
        sorted(set(detail["isic4_code"])) if not detail.empty else []
    ).sum(axis=0).reindex(provinces).fillna(0.0)
    # Parity is deliberately taken over **all** provinces, excluded ones included: the
    # scale view's bar is an equal share of the industry, and `share_national_*` is
    # denominated on all 24. Excluding a province from the answer does not re-base the
    # measure it is scored on.
    parity = parity_threshold(view, len(provinces))
    threshold = parity * float(cut)
    picks = qualifying_provinces(score, threshold, size, TD_EXCLUDED_PROVINCES)

    # The map keeps every province — an excluded one is greyed, not dropped, so a reader
    # looking for Galápagos finds it rather than wondering where it went. Only the score
    # is withheld: it is the one figure the lens will never act on there. Its
    # observations (industries, FTE, establishments) are real and stay on the hover.
    excluded_here = [code for code in provinces if code in TD_EXCLUDED_PROVINCES]
    map_df = plot_df.copy()
    if excluded_here:
        greyed = map_df["id_code"].isin(excluded_here)
        map_df.loc[greyed, "score"] = float("nan")
        map_df.loc[greyed, "id_name"] = map_df.loc[greyed, "id_name"] + t(
            TD_EXCLUDED_MAP_SUFFIX, TD_EXCLUDED_MAP_SUFFIX_ES
        )
        # Everything below the map ranks provinces, and a province the rule can never
        # pick has no rank — so the bar, the worked example and the detail table see
        # only the assignable ones.
        plot_df = plot_df[~plot_df["id_code"].isin(excluded_here)].reset_index(drop=True)

    st.plotly_chart(
        make_presence_choropleth(load_province_geojson(), map_df, view, top_codes=picks),
        width="stretch",
        key=f"td-map-{selected}-{view}-{base}-{guard}-{preset}-{cut}",
    )
    view_label = str(presence_view(view)["label"]).lower()
    base_label = presence_bases()[base].lower()
    st.caption(
        t(
            f"Weighted presence of the product's industries, {view_label}, measured in "
            f"{base_label}. Outlined: the {len(picks)} province"
            f"{'' if len(picks) == 1 else 's'} this product is assigned to. Employment is "
            "a **stock** the industries hold today — registered formal FTE positions "
            "counted where the establishment operates — never jobs this opportunity would "
            "create.",
            f"Presencia ponderada de las industrias del producto, {view_label}, medida en "
            f"{base_label}. Con contorno: "
            f"{'la provincia' if len(picks) == 1 else f'las {len(picks)} provincias'} a "
            f"{'la' if len(picks) == 1 else 'las'} que se asigna este producto. El empleo "
            "es un **acervo** que las industrias tienen hoy, es decir plazas equivalentes "
            "formales registradas donde opera el establecimiento, y nunca empleos que esta "
            "oportunidad crearía.",
        )
    )
    if excluded_here:
        st.caption(t(TD_EXCLUSION_NOTE, TD_EXCLUSION_NOTE_ES))

    _score_explainer(
        detail, plot_df, province_name, class_labels, view, base, weight, preset,
        has_anchors, anchor["n_anchors"],
    )

    # Province × industry detail, collapsed.
    with st.expander(
        t("Province by province — every score, side by side",
          "Provincia por provincia: todos los puntajes, uno al lado del otro"),
        expanded=False,
    ):
        if detail.empty:
            st.info(
                t(
                    "No industry of this opportunity clears the crosswalk-weight floor.",
                    "Ninguna industria de esta oportunidad supera el piso de peso de "
                    "concordancia.",
                )
            )
        else:
            wide = cells.pivot_table(
                index="id_code", columns="isic4_code", values="presence_used", aggfunc="max"
            )
            totals = cells.groupby("id_code").agg(
                fte=("plazas_equiv", "sum"),
                establishments=("num_establecimientos", "sum"),
                thin=("thin", "sum"),
            )
            table = plot_df.set_index("id_code").join(wide).join(totals)
            col = {
                "province": t("Province", "Provincia"),
                "score": t("Score", "Puntaje"),
                "direct": t("Direct route", "Ruta directa"),
                "anchor": t("Anchor route", "Ruta de anclas"),
                "observable": t("Industries observable", "Industrias observables"),
                "fte": t("FTE (product's industries)",
                         "Plazas equiv. (industrias del producto)"),
                "estab": t("Establishments (product's industries)",
                           "Establecimientos (industrias del producto)"),
                "thin": t("⚠ Low-support cells", "⚠ Celdas de bajo soporte"),
            }
            table = table.rename(columns={
                "id_name": col["province"], "score": col["score"],
                "direct": col["direct"], "anchor": col["anchor"],
                "n_industries": col["observable"], "fte": col["fte"],
                "establishments": col["estab"], "thin": col["thin"],
            }).sort_values(col["score"], ascending=False)
            industry_columns = [c for c in wide.columns]
            ordered = [
                col["province"], col["score"], col["direct"], col["anchor"],
                *industry_columns, col["observable"], col["fte"], col["estab"],
                col["thin"],
            ]
            if not has_anchors:
                ordered.remove(col["anchor"])
            st.dataframe(
                table[ordered], width="stretch", hide_index=True,
                column_config={
                    col["score"]: st.column_config.NumberColumn(format="%.3f"),
                    col["direct"]: st.column_config.NumberColumn(format="%.3f"),
                    col["anchor"]: st.column_config.NumberColumn(format="%.3f"),
                    col["fte"]: st.column_config.NumberColumn(
                        format="%.0f",
                        help=t(
                            "Registered FTE positions this province holds across the "
                            "product's counted industries — a stock they hold today, not "
                            "jobs the opportunity would create. Includes cells the guard "
                            "removed.",
                            "Plazas equivalentes registradas que esta provincia tiene en las "
                            "industrias contadas del producto: un acervo que tiene hoy, no "
                            "empleos que la oportunidad crearía. Incluye las celdas que "
                            "retiró el filtro.",
                        ),
                    ),
                    col["estab"]: st.column_config.NumberColumn(
                        format="%.0f",
                        help=t(
                            "Local units this province holds across the product's counted "
                            "industries. Includes cells the guard removed.",
                            "Unidades locales que esta provincia tiene en las industrias "
                            "contadas del producto. Incluye las celdas que retiró el filtro.",
                        ),
                    ),
                    col["thin"]: st.column_config.NumberColumn(
                        format="%.0f",
                        help=t(
                            "How many of the product's industries this province holds with "
                            f"**both** fewer than {MIN_ESTAB} establishments **and** fewer "
                            f"than {MIN_PLAZAS:.0f} FTE. With the support guard on, each of "
                            "those contributes 0 to the score however large its location "
                            "quotient looks — a ratio built on one or two jobs is noise, not "
                            "evidence. A province can therefore show FTE and establishments "
                            "here while still scoring 0.",
                            "Cuántas de las industrias del producto tiene esta provincia con "
                            f"**a la vez** menos de {MIN_ESTAB} establecimientos **y** menos "
                            f"de {MIN_PLAZAS:.0f} plazas equivalentes. Con el filtro de "
                            "actividad real activado, cada una de ellas aporta 0 al puntaje "
                            "por más alto que parezca su cociente de localización, porque una "
                            "razón construida sobre uno o dos puestos es ruido y no "
                            "evidencia. Por eso una provincia puede mostrar plazas y "
                            "establecimientos aquí y aun así obtener 0.",
                        ),
                    ),
                    **{
                        code: st.column_config.NumberColumn(
                            format="%.2f",
                            help=f"{code} · {class_labels.get(code, '')}"
                            + t(
                                " — presence as scored (0 when absent or filtered by the "
                                "support guard)",
                                ": presencia tal como entró al puntaje (0 cuando está "
                                "ausente o la retiró el filtro de actividad real)",
                            ),
                        )
                        for code in industry_columns
                    },
                },
            )
            formula = (
                t(
                    f"**Score = {weight:.2f} × Direct route + {1 - weight:.2f} × Anchor "
                    f"route** (the *{_route_labels()[preset]}* preset).",
                    f"**Puntaje = {weight:.2f} × Ruta directa + {1 - weight:.2f} × Ruta de "
                    f"anclas** (opción *{_route_labels()[preset]}*).",
                )
                if has_anchors
                else t(
                    "**Score = the direct route** — this opportunity has no anchors.",
                    "**Puntaje = la ruta directa**, porque esta oportunidad no tiene anclas.",
                )
            )
            st.caption(
                formula
                + t(
                    " Each industry column is that industry's presence in the province "
                    "**as it entered the score** — zero where the industry is absent or "
                    "where the support guard removed it. Weights are not renormalized when "
                    "an industry is missing: a province that lacks one of the product's "
                    "industries genuinely has less of the capability, and renormalizing "
                    "would hide exactly that.",
                    " Cada columna de industria es la presencia de esa industria en la "
                    "provincia **tal como entró al puntaje**, y vale cero donde la "
                    "industria está ausente o donde la retiró el filtro de actividad real. "
                    "Los pesos no se renormalizan cuando falta una industria: una provincia "
                    "a la que le falta una de las industrias del producto genuinamente tiene "
                    "menos de la capacidad, y renormalizar ocultaría justamente eso.",
                )
            )

    # ------------------------------------------------------------------ #
    # Section 4 — the ranking.                                            #
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader(t("Which provinces are best placed?",
                   "¿Qué provincias están mejor ubicadas?"))
    st.plotly_chart(
        make_presence_ranking_bar(plot_df, view, top_codes=picks, show_anchor=has_anchors),
        width="stretch",
        key=f"td-rank-{selected}-{view}-{base}-{guard}-{preset}-{cut}",
    )

    notes: list[str] = []
    # Counted over the provinces the rule can actually pick, so "assigned to 4 of 9 with
    # observable presence" compares like with like — an excluded province with presence
    # would otherwise inflate the denominator of a ratio it can never be in the numerator of.
    n_scorable = int((score.drop(index=excluded_here, errors="ignore") > 0).sum())
    parity_name = (
        t("the national average", "el promedio nacional")
        if view == "lq"
        else t("an equal share of the industry",
               "una participación igual de la industria")
    )
    # At the default the bar *is* parity; above it the multiplier has to be named,
    # or the caption claims 2.000 is the national average.
    bar = (
        f"{parity_name} ({_fmt_presence(threshold, view)})"
        if float(cut) == 1.0
        else f"x{cut:g} {parity_name} ({_fmt_presence(threshold, view)})"
    )
    if not picks:
        notes.append(
            t(
                f"**No province reaches {bar}** on this product's industry mix, so it is "
                f"assigned nowhere — {n_scorable} have some observable presence, none of it "
                "at parity",
                f"**Ninguna provincia alcanza {bar}** en la combinación de industrias de "
                f"este producto, así que no se asigna a ninguna parte: {n_scorable} tienen "
                "alguna presencia observable, pero ninguna en paridad",
            )
        )
    else:
        named = ", ".join(province_name[p] for p in picks)
        notes.append(
            t(
                f"Assigned to the **{len(picks)}** province"
                f"{'' if len(picks) == 1 else 's'} at or above {bar} — "
                f"**{named}** — of {n_scorable} with any observable presence",
                f"Asignado a **{len(picks)}** "
                f"{'provincia' if len(picks) == 1 else 'provincias'} en o por encima de "
                f"{bar} (**{named}**), de {n_scorable} con alguna presencia observable",
            )
        )
    if has_anchors:
        notes.append(
            t(
                "the product's own industries are observable in "
                f"{blend['n_observable_direct']} of 24 provinces and the anchors' in "
                f"{blend['n_observable_anchor']}, so the nominal route weight of "
                f"{weight:.2f} delivers an effective {blend['effective_w']:.2f} — "
                "intended: where a route cannot be observed it should count for less, and "
                "rescaling it back up would imply coverage we do not have",
                "las industrias propias del producto son observables en "
                f"{blend['n_observable_direct']} de 24 provincias y las de las anclas en "
                f"{blend['n_observable_anchor']}, así que el peso nominal de rutas de "
                f"{weight:.2f} entrega uno efectivo de {blend['effective_w']:.2f}. Es "
                "intencional: donde una ruta no puede observarse debe contar menos, y "
                "reescalarla hacia arriba implicaría una cobertura que no tenemos",
            )
        )
    st.caption("; ".join(notes) + ".")

    # A province can top the ranking on anchors alone, with none of the industries
    # that actually make the product observable in it. That is what the weights
    # say, and the settled arithmetic stands — but it is the single most
    # misreadable result on the page, so it is named rather than left to be
    # reverse-engineered from the expander.
    if has_anchors and weight > 0:
        anchor_only = [p for p in picks if direct["presence"].get(p, 0.0) <= 0]
        if anchor_only:
            names = ", ".join(province_name[p] for p in anchor_only)
            direct_only_label = _route_labels()["direct"]
            st.warning(
                t(
                    f"⚠ **{names}** {'is' if len(anchor_only) == 1 else 'are'} assigned on "
                    "the **anchors alone**: none of the industries that make this product "
                    f"is observable {'there' if len(anchor_only) == 1 else 'in them'}, so "
                    "the direct route contributes 0 and the whole score comes from "
                    "industries related to the product rather than from the product's own. "
                    f"Switch the route weight to **{direct_only_label}** to see the ranking "
                    "on the product's own industries, and open the walk-through above to "
                    "see which anchor industry is carrying the score.",
                    f"⚠ **{names}** se "
                    f"{'asigna' if len(anchor_only) == 1 else 'asignan'} **solo por las "
                    "anclas**: ninguna de las industrias que elaboran este producto es "
                    f"observable {'ahí' if len(anchor_only) == 1 else 'en ellas'}, así que "
                    "la ruta directa aporta 0 y todo el puntaje viene de industrias "
                    "relacionadas con el producto y no de las suyas propias. Cambie el peso "
                    f"de las rutas a **{direct_only_label}** para ver el ordenamiento sobre "
                    "las industrias propias del producto, y abra el recorrido de arriba "
                    "para ver qué industria de ancla está sosteniendo el puntaje.",
                )
            )

    # Counted over DISTINCT cells: an industry on both routes would otherwise be
    # reported twice, implying the guard cost more evidence than it did.
    removed = (
        int((cells["thin"] & ((cells["plazas_equiv"] > 0) | (cells["num_establecimientos"] > 0))).sum())
        if not cells.empty
        else 0
    )
    if guard and removed:
        material = (
            _stack(direct["material_removed"], anchor["material_removed"])
            if has_anchors
            else direct["material_removed"]
        )
        if not material.empty:
            material = material.drop_duplicates(["id_code", "isic4_code"])
        message = t(
            f"The support guard removed **{removed}** province × industry cell"
            f"{'' if removed == 1 else 's'} for this product — each with fewer than "
            f"{MIN_ESTAB} establishments *and* fewer than {MIN_PLAZAS:.0f} FTE.",
            f"El filtro de actividad real retiró **{removed}** "
            f"{'celda' if removed == 1 else 'celdas'} provincia × industria para este "
            f"producto, cada una con menos de {MIN_ESTAB} establecimientos *y* menos de "
            f"{MIN_PLAZAS:.0f} plazas equivalentes.",
        )
        if not material.empty:
            named = material.drop_duplicates(["id_code", "isic4_code"])
            parts = [
                f"{province_name.get(row.id_code, row.id_code)} × {row.isic4_code} "
                + t(
                    f"({_fmt_int(row.plazas_equiv)} FTE, {row.share_national:.0%} of the "
                    "industry nationally)",
                    f"({_fmt_int(row.plazas_equiv)} plazas, {row.share_national:.0%} de la "
                    "industria a nivel nacional)",
                )
                for row in named.itertuples()
            ]
            message += t(
                f" ⚠ {len(parts)} of them hold at least {MATERIAL_SHARE:.0%} of their "
                f"industry's national employment: {'; '.join(parts)}.",
                f" ⚠ {len(parts)} de ellas concentran al menos el {MATERIAL_SHARE:.0%} del "
                f"empleo nacional de su industria: {'; '.join(parts)}.",
            )
        st.caption(message)
    elif not guard:
        st.caption(
            t(
                "⚠ The support guard is **off**: every cell counts, including "
                "single-establishment ones. 60% of the cells behind these industries carry "
                "fewer than 5 positions, and a location quotient built on one job can reach "
                "×12.",
                "⚠ El filtro de actividad real está **desactivado**: cuentan todas las "
                "celdas, incluidas las de un solo establecimiento. El 60% de las celdas "
                "detrás de estas industrias tiene menos de 5 plazas, y un cociente de "
                "localización construido sobre un solo puesto puede llegar a ×12.",
            )
        )
