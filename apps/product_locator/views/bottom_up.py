"""Bottom-up page: province diversification explorer.

The complement of the top-down lens. That one starts from an opportunity and
asks where it could be made; this one starts from a **province** and asks which
industries make sense to diversify into — no product angle anywhere, and no
product selection required (the page is deliberately ungated and never reads or
writes ``selected_province``).

Three sections, each answering a different question about one province:

1. **What is this economy made of?** — a composition treemap of the registered
   economy (employment or establishments, at three CIIU levels).
2. **What distinguishes it?** — relative presence: the province's share of an
   industry divided by the peer's (Ecuador, or the other provinces of its
   region), the Metroverse comparison.
3. **What is worth diversifying into?** — a feasibility × attractiveness plane:
   the top industries per presence state by a user-weighted rank-space score.
   Feasibility is an **explicit β mix** of the industry's own presence (relative
   presence) and its related capabilities (related-only relatedness density),
   which is also the plotted x-axis, so the plot and the ranking cannot disagree.
   Two eligibility guards decide what may be **recommended**, both default on and
   both applied after scoring and before the top-k cut, so neither moves a
   survivor's numbers: ``bu-support`` (real activity behind the presence — the
   same floors as Section 2 and the top-down lens) and ``bu-connected`` (enough
   proximity mass for the density half of feasibility to be measurable).
There was a fourth — the shortlist priced against Ecuador's own frontier as a
potential/gap treemap. It **moved** to *The candidate industries list* (the app's
third lens) on 2026-08-27, unchanged in method: "how much could each of these be worth
here?" is a better question asked of the province's committed list than of a 30-row
candidate pool no downstream artifact consumes. The benchmark functions it used stay in
``metrics.py``, where the list's assembly and the appendix's worked example still call
them. See ``notes/reference/product-locator/candidate_list_page.md``.

One page-level **tradable focus** toggle (``bu-tradable``, default on since v4 —
it matches the committed province candidate list, the candidates grid and the
builder) restricts what the page may recommend to industries whose output can
leave the province, mining/quarrying/hunting excluded. It is a **choice set, not an information set**: it changes membership,
ranks and scores, and no raw value anywhere.

The page **consumes precomputed notebook exports and never recomputes** density,
RCA or the labor metrics: ``relatedness_density/`` (from
``notebooks/relatedness_density.ipynb``) and ``industry_labor_profile/`` (from
``notebooks/industry_labor_profile.ipynb``). The arithmetic here is share
ratios, region aggregation and percentile ranks — nothing else.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import state
from config import (
    BU_DEFAULT_BETA,
    BU_DEFAULT_CONNECTED,
    BU_DEFAULT_SUPPORT,
    BU_DEFAULT_TOP_K,
    BU_DEFAULT_TRADABLE,
    BU_DEFAULT_WEIGHT,
    DEFAULT_BOTTOM_UP_PROVINCE,
    GSECTOR_COLORS,
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
    load_proximity_matrix,
    load_tradable_menu,
)
from metrics import (
    capability_neighbors,
    density_contributions,
    feasibility_mix,
    opportunity_score,
    relative_presence,
    connected_eligible,
    support_eligible,
)
from lang import coerce_choice, t
from viz import (
    fmt_number,
    make_attractiveness_feasibility_scatter,
    make_composition_treemap,
    make_density_contribution_bar,
    make_relative_presence_bars,
    state_labels,
)

NATIONAL = "99"
_TOP_N = 10  # industries in each of the relative-presence chart's two blocks

# CIIU level control. Sections 1 and 2 carry their own copy: the two questions
# are naturally read at different granularities, so the controls stay separate.
# Keyed by code, never by display text: these are persisted in session state
# (bu-level, bu-rp-level), so a translated label as the key would break the
# lookup or orphan the pick when the language changes. Labels are resolved at
# the widget through format_func.
_LEVELS = {"section": "Sections", "division": "Divisions", "class": "Industries"}
_LEVELS_ES = {"section": "Secciones", "division": "Divisiones", "class": "Industrias"}
_LEVEL_DEFAULT = "division"
_HELP_LEVELS_EN = (
    "How finely the economy is cut: 20 CIIU sections, 87 divisions, or the 412 "
    "4-digit industries."
)
_HELP_LEVELS_ES = (
    "Con qué detalle se corta la economía: 20 secciones CIIU, 87 divisiones o las 412 "
    "industrias de 4 dígitos."
)

# Composition variable control. Labels -> columns of the composition export.
_VARS = {"plazas_equiv": "Employment", "n_estab": "Establishments"}
_VARS_ES = {"plazas_equiv": "Empleo", "n_estab": "Establecimientos"}
_VAR_DEFAULT = "plazas_equiv"
_HELP_VARS_EN = (
    "Employment — full-time-equivalent positions registered in the province "
    "(where the units operate). · Establishments — the number of local units. "
    "The two tell genuinely different stories: an industry can hold many "
    "positions in few large units, or few positions across many small ones."
)
_HELP_VARS_ES = (
    "Empleo: plazas equivalentes registradas en la provincia, donde operan las "
    "unidades. · Establecimientos: el número de unidades locales. Los dos cuentan "
    "historias genuinamente distintas, porque una industria puede concentrar muchas "
    "plazas en pocas unidades grandes, o pocas plazas repartidas en muchas pequeñas."
)

# Attractiveness variable control (Section 3), mirroring Page 4's wording.
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
    "Average wage paid — what a full-time-equivalent worker earns per year in this "
    "industry (2024 business registry). · Average jobs per firm — how many jobs a "
    "typical employing firm in this industry carries: the employment ticket of an "
    "entrant. Both are national values (the registry books firms to their "
    "headquarters province)."
)
_HELP_ATTR_ES = (
    "Salario promedio pagado: lo que gana al año una plaza equivalente en esta "
    "industria (registro de empresas 2024). · Empleo promedio por empresa: cuántos "
    "puestos sostiene una empresa empleadora típica de esta industria, es decir el "
    "boleto de empleo de quien entra. Ambos son valores nacionales, porque el registro "
    "asigna cada empresa a la provincia de su sede."
)

_HELP_TRADABLE = (
    "Off, the page describes the **whole registered economy**. On, it only "
    "recommends industries that pass two tests. First, the output can plausibly "
    "be **sold outside the province** — so a province is not told to diversify "
    "into more hairdressers or local government. Second, a private entrant could "
    "actually **cause the industry to appear** there: out come the "
    "**state-allocated** ones, where siting is a ministry or budget decision "
    "(hospitals, universities, power generation), and the **regulated networks**, "
    "where entry is a national licence or charter and a province's establishments "
    "are branch nodes (telecoms, broadcasting, banks and insurers). It also drops "
    "**mining, quarrying and hunting**, where what a province can do is geology "
    "rather than capability. Logging and fishing stay: those are renewable "
    "harvests with real capability clusters behind them.\n\n"
    "It changes **what can be recommended, never the numbers**. "
    "Every density, relative presence and share below is exactly the same either "
    "way; only which industries are eligible for a ranking changes."
)
_HELP_TRADABLE_ES = (
    "Desactivado, la página describe **toda la economía registrada**. Activado, solo "
    "recomienda industrias que pasan dos pruebas. Primero, que su producción pueda "
    "plausiblemente **venderse fuera de la provincia**, de modo que a una provincia no "
    "se le diga que diversifique hacia más peluquerías o más administración pública. "
    "Segundo, que un entrante privado pueda realmente **hacer que la industria "
    "aparezca** ahí: quedan fuera las de **asignación estatal**, donde la ubicación es "
    "una decisión ministerial o presupuestaria (hospitales, universidades, generación "
    "eléctrica), y las **redes reguladas**, donde entrar exige una licencia o concesión "
    "nacional y los establecimientos de una provincia son nodos de sucursal "
    "(telecomunicaciones, radiodifusión, bancos y aseguradoras). También descarta "
    "**minas, canteras y caza**, donde lo que una provincia puede hacer es geología y "
    "no capacidad. La silvicultura y la pesca se quedan: son cosechas renovables con "
    "verdaderos conglomerados de capacidades detrás.\n\n"
    "Cambia **qué puede recomendarse, nunca las cifras**. Todas las densidades, "
    "presencias relativas y participaciones de abajo son exactamente las mismas en "
    "ambos casos; solo cambia qué industrias son elegibles para un ordenamiento."
)

_HELP_BETA = (
    "What **feasible** should mean here. Each industry is ranked twice — by how "
    "much of it the province already has (**relative presence**) and by how much "
    "of what it needs the *rest* of the local economy provides (**related-only "
    "relatedness density**) — and the two percentile ranks are mixed with this "
    "slider. Slide to 1.0 to mean 'we already do this', 0.0 to mean 'the "
    "capabilities are here even though the industry is not'. The two genuinely "
    "disagree: in most provinces they correlate only weakly, which is the whole "
    "reason the mix is yours to set rather than buried in one number."
)
_HELP_BETA_ES = (
    "Qué debe significar **factible** aquí. Cada industria se ordena dos veces, por "
    "cuánto de ella ya tiene la provincia (**presencia relativa**) y por cuánto de lo "
    "que necesita se lo aporta el *resto* de la economía local (**densidad de "
    "industrias relacionadas**), y los dos percentiles se mezclan con este deslizador. "
    "Llévelo a 1,0 para decir 'esto ya lo hacemos' y a 0,0 para decir 'las capacidades "
    "están aquí aunque la industria no'. Los dos discrepan de verdad: en la mayoría de "
    "las provincias apenas se correlacionan, y esa es toda la razón por la que la "
    "mezcla la fija usted en lugar de quedar enterrada en una sola cifra."
)

_HELP_WEIGHT = (
    "How the shortlist is ordered. Each industry carries a **feasibility** score "
    "(the mix above) and a percentile rank on the chosen **attractiveness** "
    "metric nationally, and the two are mixed with this weight. Raw units cannot "
    "be added (a density of 0.6 against $12,000 a year), so the mixing happens in "
    "rank space. Slide to 1.0 for feasibility only, 0.0 for attractiveness only. "
    "The shortlist follows every control on this section, including this one: the "
    "score orders the list, it is not a measurement of anything."
)
_HELP_WEIGHT_ES = (
    "Cómo se ordena la lista corta. Cada industria lleva un puntaje de "
    "**factibilidad** (la mezcla de arriba) y un percentil nacional en la métrica de "
    "**atractivo** elegida, y los dos se mezclan con este peso. Las unidades crudas no "
    "pueden sumarse (una densidad de 0,6 frente a 12.000 dólares al año), así que la "
    "mezcla ocurre en el espacio de rangos. Llévelo a 1,0 para solo factibilidad y a "
    "0,0 para solo atractivo. La lista corta sigue todos los controles de esta "
    "sección, incluido este: el puntaje ordena la lista, no mide nada."
)

_HELP_ROBUST_EN = (
    "Ticked when the industry makes the top list under **both** attractiveness "
    "metrics — high pay *and* many jobs per firm — at the current settings. An "
    "unticked row is not worse; it is a pick that depends on which of the two you "
    "care about."
)
_HELP_ROBUST_ES = (
    "Marcada cuando la industria entra en la lista con **ambas** métricas de atractivo "
    "(paga alta *y* muchos puestos por empresa) con los parámetros actuales. Una fila "
    "sin marca no es peor: es una elección que depende de cuál de las dos le importe."
)

_HELP_GUARD = (
    "A handful of units produces wild ratios, so industries with too little activity "
    "here are held out of the ranking. An industry is **too thin only when it fails "
    f"both floors** — fewer than {MIN_ESTAB} establishments **and** fewer than "
    f"{MIN_PLAZAS:,} full-time-equivalent positions in this province. Either one is "
    "enough to survive: a single large operation is real evidence, and so are five "
    "one-person firms.\n\nThis is the **same guard the top-down lens applies**, so a "
    "(province, industry) cell that counts as observable there counts here too."
)
_HELP_GUARD_ES = (
    "Un puñado de unidades produce razones desbocadas, así que las industrias con muy "
    "poca actividad aquí quedan fuera del ordenamiento. Una industria es **demasiado "
    "delgada solo cuando falla los dos pisos**: menos de "
    f"{MIN_ESTAB} establecimientos **y** menos de {MIN_PLAZAS:,} plazas equivalentes "
    "en esta provincia. Cualquiera de los dos basta para sobrevivir: una sola "
    "operación grande es evidencia real, y también lo son cinco empresas de una "
    "persona.\n\nEs el **mismo filtro que aplica el enfoque descendente**, así que una "
    "celda (provincia, industria) que cuenta como observable allá cuenta aquí también."
)

# The tooltip of the ``bu-connected`` guard. Was the ``⚠ Thin`` column's help text
# until v4 turned the flag into a default-on filter: with the filter on every row in
# the table is well-connected, so the column was constant-False and carried no
# information, and its explanation belongs on the control that now acts on it.
_HELP_CONNECTED = (
    "A few industries are barely connected to anything else in the economy — the "
    "proximity mass linking them to all other industries is under "
    f"{THIN_NEIGHBORHOOD_THRESHOLD}. Their related-only density is then a ratio of "
    "two tiny numbers, with that tiny mass as the whole denominator, so it moves on "
    "almost nothing. Read those rows off their own presence instead.\n\nOn, they are "
    "held out of the **shortlist** — we do not recommend them anywhere, and the "
    "committed province candidate list drops them before it is built. This changes "
    "**only which industries can be picked**: they stay on the plot as grey context "
    "dots, keep their exported presence state, and every surviving row's feasibility, "
    "score and raw value is identical either way. Turn it off to put them back on the "
    "list — the caption below then names the ones that made it."
)
_HELP_CONNECTED_ES = (
    "Unas pocas industrias están apenas conectadas con el resto de la economía: la "
    "masa de proximidad que las une a todas las demás industrias es menor a "
    f"{THIN_NEIGHBORHOOD_THRESHOLD}. Su densidad de industrias relacionadas es "
    "entonces una razón entre dos números diminutos, con esa masa mínima como todo el "
    "denominador, así que se mueve por casi nada. Lea esas filas por su propia "
    "presencia.\n\nActivado, quedan fuera de la **lista corta**: no las recomendamos "
    "en ninguna parte, y la lista comprometida de industrias candidatas las descarta "
    "antes de construirse. Esto cambia **solo qué industrias pueden elegirse**: siguen "
    "en el gráfico como puntos grises de contexto, conservan su estado de presencia "
    "exportado, y la factibilidad, el puntaje y el valor crudo de cada fila "
    "sobreviviente son idénticos en ambos casos. Desactívelo para devolverlas a la "
    "lista; la nota de abajo nombra entonces las que entraron."
)

_HELP_SUPPORT = (
    "Whether an industry may be **recommended** at all. An industry the province "
    f"already has is held out of the shortlist when it fails **both** floors — under "
    f"{MIN_ESTAB} establishments **and** under {MIN_PLAZAS:,} full-time-equivalent "
    "positions here — because a two-establishment 'specialization' is a ratio, not an "
    "industry. Either floor on its own is enough to survive: one large operation is "
    "evidence, and so are five one-person firms.\n\nIndustries the province does **not** "
    "have are never held out — zero presence is what an entry candidate is. This is the "
    "same guard as the one on relative presence above and in the top-down lens, and it "
    "changes **only which industries can be picked**: every feasibility, score and raw "
    "value below is identical either way, and the held-out industries stay on the plot as "
    "grey context dots with their exported presence state intact."
)
_HELP_SUPPORT_ES = (
    "Si una industria puede ser **recomendada** en absoluto. Una industria que la "
    "provincia ya tiene queda fuera de la lista corta cuando falla **los dos** pisos, "
    f"menos de {MIN_ESTAB} establecimientos **y** menos de {MIN_PLAZAS:,} plazas "
    "equivalentes aquí, porque una 'especialización' de dos establecimientos es una "
    "razón y no una industria. Cualquiera de los dos pisos por sí solo basta para "
    "sobrevivir: una operación grande es evidencia, y también lo son cinco empresas de "
    "una persona.\n\nLas industrias que la provincia **no** tiene nunca quedan fuera, "
    "porque presencia cero es justamente lo que es un candidato de entrada. Es el "
    "mismo filtro que el de la presencia relativa de arriba y que el del enfoque "
    "descendente, y cambia **solo qué industrias pueden elegirse**: toda "
    "factibilidad, puntaje y valor crudo de abajo es idéntico en ambos casos, y las "
    "industrias excluidas siguen en el gráfico como puntos grises de contexto con su "
    "estado de presencia exportado intacto."
)

_HELP_BOTH_BASES = (
    "The presence states on this page are **establishment-based** — how many local units "
    "the province has of an industry against the national pattern. Ticked when the "
    "industry lands in the **same state** under the employment-based measure too, so the "
    "reading does not depend on which base you count in.\n\nAn unticked row is not worse, "
    "and it can go either way: an industry can be specialized in units and merely present "
    "in positions (many small operations), or the reverse (a few large employers). "
    "Industries the province is missing entirely agree trivially — the signal lives in the "
    "present and specialized rows."
)
_HELP_BOTH_BASES_ES = (
    "Los estados de presencia de esta página se basan en **establecimientos**: "
    "cuántas unidades locales tiene la provincia de una industria frente al patrón "
    "nacional. Marcada cuando la industria cae en el **mismo estado** también con la "
    "medida basada en empleo, de modo que la lectura no depende de en qué base se "
    "cuente.\n\nUna fila sin marca no es peor, y puede ir en cualquier dirección: una "
    "industria puede estar especializada en unidades y solo presente en plazas (muchas "
    "operaciones pequeñas), o al revés (unos pocos empleadores grandes). Las "
    "industrias que la provincia no tiene coinciden trivialmente, así que la señal "
    "vive en las filas presentes y especializadas."
)

def _prov_label_factory(geo: pd.DataFrame):
    lookup = (
        geo[["id_code", "id_name", "region"]]
        .dropna(subset=["id_code"])
        .drop_duplicates("id_code")
        .set_index("id_code")
    )

    def label(code: str) -> str:
        if code == NATIONAL:
            return t("Ecuador — national", "Ecuador (nacional)")
        if code in lookup.index:
            row = lookup.loc[code]
            # id_name is UPPER-case Spanish (and one carries a trailing space).
            return f"{str(row['id_name']).strip()} ({row['region']})"
        return code

    return label, lookup


def _labels(en: dict[str, str], es: dict[str, str]) -> dict[str, str]:
    """A {code: label} control map in the current language.

    The code keys are what land in session state, so only the values move.
    """
    return {code: t(en[code], es[code]) for code in en}


def _segmented(
    label: str,
    labels: dict[str, str],
    default: str,
    key: str,
    help_text: str,
    disabled: bool = False,
) -> str:
    """A segmented control over a {code: display label} mapping.

    Returns the **code**, which is also what lands in session state. The code
    keys are only half the guarantee: ``coerce_choice`` is what repairs the
    stored value when a language flip leaves a display label behind it.
    """
    coerce_choice(key, list(labels), labels)
    choice = st.segmented_control(
        label, list(labels.keys()), default=default, key=key, help=help_text,
        disabled=disabled, format_func=lambda code: labels.get(code, code),
    )
    return choice or default


def _composition_slice(comp: pd.DataFrame, geography: str, level: str) -> pd.DataFrame:
    """Industry rows of one geography at one CIIU level, Spanish-named."""
    df = comp[(comp["id_code"] == geography) & (comp["level"] == level)].copy()
    df["name"] = df["name_es"].fillna(df["name_en"])
    return df


def _shortlist_codes(
    plane: pd.DataFrame, var: str, beta: float, weight: float, top_k: int,
    apply_support: bool, apply_connected: bool,
) -> set[str]:
    """The top-``top_k``-per-presence-state shortlist under one attractiveness metric.

    Replayed on a copy so the page can ask "would the *other* metric have picked
    this too?" without disturbing the plane it is plotting. Everything else — the
    menu filter, β, the weight and **both eligibility guards** — is held at the
    current setting, which is what makes the answer a robustness check on the metric
    choice alone. The guards in particular have to travel: a ✓ against an ungated
    other-metric list would be checking robustness against a list the page would
    never show, which is the bug the support guard fixed and the same trap the
    connectedness guard would fall into.
    """
    frame = plane.copy()
    frame["x"] = feasibility_mix(frame, var, float(beta))
    frame["score"] = opportunity_score(frame, var, float(weight), feasibility_col="x")
    frame = frame.dropna(subset=["score"])
    if apply_support:
        frame = frame[
            support_eligible(
                frame["state"], frame["n_estab_prov"], frame["plazas_equiv_prov"],
                MIN_ESTAB, MIN_PLAZAS,
            )
        ]
    if apply_connected:
        frame = frame[connected_eligible(frame["thin_neighborhood"])]
    if frame.empty:
        return set()
    picks = (
        frame.sort_values("score", ascending=False)
        .groupby("state", group_keys=False)
        .head(int(top_k))
    )
    return set(picks["codigo_clase"])


def _capability_expander(
    row: pd.Series,
    place: str,
    prov_density: pd.DataFrame,
    labels: dict[str, str],
    display_codes: dict[str, str],
) -> None:
    """The 'why is this feasible here?' panel for one industry.

    Two readings of the same neighbourhood. **What the province has that is like
    this** ranks the industry's closest relatives by raw proximity and marks what
    exists locally — the capability story a reader can act on. **Where the score
    comes from** decomposes the density itself into `φ × presence` shares, which
    is the arithmetic and doubles as a falsifier: when one odd neighbour carries
    almost the whole score, the pick is an artefact and this is where it shows.

    The decomposition is always the **related-only** one (``include_self=False``),
    matching the only density the page carries since the feasibility mix replaced
    the with-self view: own presence is the β half of the mix and is read off the
    industry's relative presence, so counting it again here as a φ = 1 self-term
    would double-count it and, in a specialized industry, drown out every genuine
    contributor at a median 79% of the bar.
    """
    code = str(row["codigo_clase"])
    phi = load_proximity_matrix()
    if code not in phi.index:
        return

    with st.expander(
        t(
            f"Why is {row['display_code']} feasible in {place}? — the capabilities "
            "behind the score",
            f"¿Por qué {row['display_code']} es factible en {place}? Las capacidades "
            "detrás del puntaje",
        ),
        expanded=False,
    ):
        neighbors = capability_neighbors(phi.loc[code], prov_density, focal=code, top_n=10)
        neighbors["name"] = neighbors["codigo_clase"].map(labels).fillna("")
        neighbors["display_code"] = (
            neighbors["codigo_clase"].map(display_codes).fillna(neighbors["codigo_clase"])
        )
        # Split on *specialization*, not on mere existence: in a diversified
        # province almost every industry exists somewhere, so an exists/does-not
        # split would put everything in one column and say nothing.
        strong = neighbors[neighbors["state"] == "specialized"]
        weak = neighbors[neighbors["state"] != "specialized"]

        def _strong_line(item: pd.Series) -> str:
            return f"- `{item['display_code']}` **{item['name']}**" + t(
                f" — RCA {item['rca']:.2f}, "
                f"{fmt_number(item['intensity'], 0)} establishments",
                f": RCA {item['rca']:.2f}, "
                f"{fmt_number(item['intensity'], 0)} establecimientos",
            )

        def _weak_line(item: pd.Series) -> str:
            head = f"- `{item['display_code']}` **{item['name']}**"
            if not bool(item["has_it"]):
                return head + t(f" — absent from {place}", f": ausente en {place}")
            return head + t(
                f" — present but below national weight (RCA {item['rca']:.2f}, "
                f"{fmt_number(item['intensity'], 0)} establishments)",
                f": presente pero por debajo del peso nacional (RCA {item['rca']:.2f}, "
                f"{fmt_number(item['intensity'], 0)} establecimientos)",
            )

        cols = st.columns(2)
        with cols[0]:
            st.markdown(
                t(f"**Strengths {place} can build on**",
                  f"**Fortalezas sobre las que {place} puede construir**")
            )
            st.markdown(
                "\n".join(_strong_line(n) for _, n in strong.iterrows())
                if not strong.empty
                else t(
                    "_None of its ten closest industries is specialized here._",
                    "_Ninguna de sus diez industrias más cercanas está especializada "
                    "aquí._",
                )
            )
        with cols[1]:
            st.markdown(t("**Weak or missing links**", "**Eslabones débiles o ausentes**"))
            st.markdown(
                "\n".join(_weak_line(n) for _, n in weak.iterrows())
                if not weak.empty
                else t(
                    "_All ten of its closest industries are specialized here._",
                    "_Las diez industrias más cercanas están especializadas aquí._",
                )
            )
        st.caption(
            t(
                "Its ten closest industries by **proximity** — how often the two are "
                "produced together inside the same firm, nationwide — split by what this "
                "province has of each: specialized on the left, present-but-thin or absent "
                "on the right. The right column is usually the actionable one. Ranked by "
                "similarity, not by how much each adds to the score below: otherwise a "
                "merely-similar industry the province happens to be big in would outrank "
                "the one that actually shares the capabilities.",
                "Sus diez industrias más cercanas por **proximidad**, es decir con qué "
                "frecuencia ambas se producen juntas dentro de la misma empresa a nivel "
                "nacional, separadas según lo que esta provincia tiene de cada una: "
                "especializadas a la izquierda, presentes pero delgadas o ausentes a la "
                "derecha. La columna derecha suele ser la accionable. Están ordenadas por "
                "similitud y no por cuánto aporta cada una al puntaje de abajo, porque de "
                "lo contrario una industria apenas parecida en la que la provincia resulta "
                "grande superaría a la que de verdad comparte las capacidades.",
            )
        )

        st.divider()
        contrib = density_contributions(
            phi.loc[code], prov_density, focal=code, top_n=10_000, include_self=False
        )
        cumulative = contrib["contribution_share"].cumsum()
        n_half = int((cumulative < 0.5).sum() + 1)
        top = contrib.head(8).copy()
        top["name"] = top["codigo_clase"].map(labels).fillna("")
        top["display_code"] = (
            top["codigo_clase"].map(display_codes).fillna(top["codigo_clase"])
        )
        lookup = prov_density.drop_duplicates("codigo_clase").set_index("codigo_clase")
        top["state"] = top["codigo_clase"].map(lookup["state"])
        top["intensity"] = pd.to_numeric(top["codigo_clase"].map(lookup["intensity"]), errors="coerce")

        st.markdown(
            t(
                "**Half of the related-capability half of this industry's feasibility "
                f"comes from {n_half} industr{'y' if n_half == 1 else 'ies'}.** Its own "
                f"presence in {place} is excluded here — that is the other half of the "
                "mix, and it is the relative presence on its card above.",
                "**La mitad de la mitad de capacidades relacionadas de la factibilidad de "
                f"esta industria viene de {n_half} "
                f"industria{'' if n_half == 1 else 's'}.** Su propia presencia en "
                f"{place} queda excluida aquí, porque es la otra mitad de la mezcla y es "
                "la presencia relativa de su ficha de arriba.",
            )
        )
        st.plotly_chart(
            make_density_contribution_bar(
                top, STATE_COLORS, state_labels(), focal_code=code, province_label=place
            ),
            width="stretch",
            key=f"bu-contrib-{code}",
        )
        st.caption(
            t(
                "Each bar is one **other** industry's share of the related-only "
                "relatedness density: its proximity to the focal industry times how "
                "present it is in this province. Colors are what the province has of that "
                "contributor. A density concentrated in one or two bars — especially "
                "unrelated-looking ones — is a sign the number rests on very little.",
                "Cada barra es la parte que aporta **otra** industria a la densidad de "
                "industrias relacionadas: su proximidad a la industria focal multiplicada "
                "por cuán presente está en esta provincia. Los colores indican lo que la "
                "provincia tiene de ese contribuyente. Una densidad concentrada en una o "
                "dos barras, sobre todo si parecen no tener relación, es señal de que la "
                "cifra se apoya en muy poco.",
            )
        )


def render() -> None:
    state.init_state()  # ungated on purpose: no product selection is required

    st.title(t("Province diversification explorer", "Explorador por provincia"))
    st.markdown(
        t(
            "Start from a **province** instead of a product. What is its registered "
            "economy made of, what does it do differently from the rest of the country, "
            "and which industries are both **plausible** and **worth having** there? "
            "Everything below reads the 2024 business registry (REEM) and the relatedness "
            "analysis — no product selection needed.",
            "Parta de una **provincia** en lugar de un producto. ¿De qué está hecha su "
            "economía registrada, qué hace distinto del resto del país y qué industrias "
            "son a la vez **plausibles** y **valiosas** ahí? Todo lo de abajo lee el "
            "registro de empresas (REEM) de 2024 y el análisis de relación entre "
            "industrias, sin necesidad de elegir un producto.",
        )
    )

    comp = load_labor_composition()
    geo = load_province_geodata()
    prov_label, prov_lookup = _prov_label_factory(geo)

    options = sorted(comp["id_code"].unique())  # 24 provinces + '99'
    # Seed the default once; from then on the widget's own key carries the value,
    # so the ✕ below can empty it.
    if "bu-province" not in st.session_state:
        st.session_state["bu-province"] = (
            DEFAULT_BOTTOM_UP_PROVINCE if DEFAULT_BOTTOM_UP_PROVINCE in options else options[0]
        )
    # Clear (✕) beside the box, so a new province can be typed without deleting
    # the old one character by character. Rendered BEFORE the selectbox, so
    # resetting the widget's session key is allowed (mirrors Page 1).
    #
    # The menu filter sits here rather than inside a section because it drives all
    # three: it is the page's answer to "which industries are we even willing to
    # recommend?", not a per-figure display option.
    prov_col, prov_clear, filter_col = st.columns([10, 1, 7], vertical_alignment="bottom")
    with filter_col:
        tradable_only = st.toggle(
            t("Focus on tradable industries", "Enfocarse en industrias transables"),
            value=BU_DEFAULT_TRADABLE,
            key="bu-tradable",
            help=t(_HELP_TRADABLE, _HELP_TRADABLE_ES),
        )
    with prov_clear:
        if st.button(
            "✕", key="bu-province-clear",
            help=t("Clear the selected province", "Quitar la provincia seleccionada"),
            disabled=st.session_state.get("bu-province") is None,
        ):
            st.session_state["bu-province"] = None
            st.rerun()
    with prov_col:
        coerce_choice("bu-province", options)
        province = st.selectbox(
            t("Province", "Provincia"),
            options=options,
            index=None,
            format_func=prov_label,
            placeholder=t("Type a province name to search",
                          "Escriba el nombre de una provincia"),
            key="bu-province",
            help=t(
                "Drives all three sections. 'Ecuador — national' shows the national "
                "composition; relative presence and provincial feasibility are undefined "
                "for the country as a whole.",
                "Gobierna las tres secciones. 'Ecuador (nacional)' muestra la "
                "composición del país; la presencia relativa y la factibilidad "
                "provincial no están definidas para el país en su conjunto.",
            ),
        )
    if province is None:
        st.info(t("Pick a province to start.", "Elija una provincia para empezar."))
        return
    is_national = province == NATIONAL
    geography_label = prov_label(province)
    place = "Ecuador" if is_national else geography_label.split(" (")[0]

    # The menu of recommendable industries, derived from the tradability labels
    # and never hardcoded (see data_loader.load_tradable_menu). ``universe`` is the
    # page's own class universe, so the caption's "N of M" counts what this page
    # actually carries rather than what ISIC contains.
    menu = load_tradable_menu()
    universe = set(comp.loc[comp["level"] == "class", "code"])
    menu_here = menu & universe
    if tradable_only:
        st.caption(
            t(
                "**Tradable focus is on.** Rankings below are restricted to the "
                f"**{len(menu_here):,} of {len(universe):,}** industries whose output can "
                "plausibly leave the province *and* that a private entrant could cause to "
                "appear there — mining and quarrying excluded as geology, state-allocated "
                "and regulated-network industries excluded as not enterable. No number "
                "changes — only which industries are eligible to be recommended.",
                "**El enfoque en transables está activado.** Los ordenamientos de abajo se "
                f"restringen a las **{len(menu_here):,} de {len(universe):,}** industrias "
                "cuya producción puede plausiblemente salir de la provincia *y* que un "
                "entrante privado podría hacer aparecer ahí: minas y canteras quedan fuera "
                "por ser geología, y las industrias de asignación estatal y de redes "
                "reguladas por no ser de libre entrada. Ninguna cifra cambia, solo qué "
                "industrias son elegibles para ser recomendadas.",
            )
        )

    # ------------------------------------------------------------------ #
    # Section 1 — what the economy is made of.                            #
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader(
        t(f"What is {place}'s economy made of?",
          f"¿De qué está hecha la economía de {place}?")
    )
    st.caption(
        t(
            "Every registered industry as a tile, grouped into its broad sector and sized "
            "by the chosen variable. Click a sector to drill in; the breadcrumb above the "
            "tiles walks back out.",
            "Cada industria registrada es un mosaico, agrupada en su sector amplio y "
            "dimensionada según la variable elegida. Haga clic en un sector para entrar en "
            "detalle; la ruta sobre los mosaicos permite regresar.",
        )
    )

    comp_cols = st.columns([2, 2, 3])
    with comp_cols[0]:
        # The menu is a list of 4-digit industries, so it cannot select a division
        # or a section: a division holds tradable and non-tradable classes at once,
        # and a tile summing both would make the slice caption below a lie. With the
        # filter on the treemap therefore reads at class level and says so.
        level1 = _segmented(
            t("Detail", "Detalle"), _labels(_LEVELS, _LEVELS_ES), _LEVEL_DEFAULT,
            "bu-level", t(_HELP_LEVELS_EN, _HELP_LEVELS_ES),
            disabled=tradable_only,
        )
        if tradable_only:
            level1 = "class"
    with comp_cols[1]:
        var1 = _segmented(
            t("Measured in", "Medido en"), _labels(_VARS, _VARS_ES), _VAR_DEFAULT,
            "bu-comp-var", t(_HELP_VARS_EN, _HELP_VARS_ES),
        )

    industries = _composition_slice(comp, province, level1)
    total_all = float(industries[var1].sum())
    if tradable_only:
        industries = industries[industries["code"].isin(menu_here)]
    if industries.empty:
        st.info(
            t("No registered **tradable** activity for this geography.",
              "No hay actividad **transable** registrada para esta geografía.")
            if tradable_only
            else t("No registered activity for this geography.",
                   "No hay actividad registrada para esta geografía.")
        )
    else:
        st.plotly_chart(
            make_composition_treemap(industries, var1, GSECTOR_COLORS, geography_label),
            width="stretch",
            key=f"bu-treemap-{province}-{level1}-{var1}-{tradable_only}",
        )
        total = float(industries[var1].sum())
        if var1 == "plazas_equiv":
            st.caption(
                t(
                    f"{total:,.0f} full-time-equivalent **positions registered** in "
                    f"{place} (2024). These are formal positions the registry sees — "
                    "nationally 2.93M, about 36% of Ecuador's employed — counted where "
                    "the establishment operates, not where its firm is headquartered.",
                    f"{total:,.0f} **plazas equivalentes registradas** en {place} (2024). "
                    "Son plazas formales que ve el registro (2,93 millones a nivel "
                    "nacional, cerca del 36% de los ocupados del Ecuador), contabilizadas "
                    "donde opera el establecimiento y no donde tiene su sede la empresa.",
                )
            )
        else:
            st.caption(
                t(
                    f"{total:,.0f} active **establishments** (local units — headquarters "
                    f"and branches alike) registered in {place} in 2024, counted where "
                    "they operate.",
                    f"{total:,.0f} **establecimientos** activos (unidades locales, tanto "
                    f"sedes como sucursales) registrados en {place} en 2024, "
                    "contabilizados donde operan.",
                )
            )
        if tradable_only:
            # Tile sizes and hover shares are the shipped whole-economy figures,
            # never renormalized to the slice — so the slice's own weight has to be
            # stated, or the treemap reads as the whole economy.
            share = total / total_all if total_all else 0.0
            st.caption(
                t(
                    f"Showing the **{len(industries):,} tradable industries** only — "
                    f"{share:.1%} of {place}'s "
                    f"{'registered FTE' if var1 == 'plazas_equiv' else 'establishments'}. "
                    "Tile sizes and hover shares are still the whole-economy figures (a "
                    "tile reading 4% means 4% of the province, not 4% of this slice). "
                    "Detail is fixed at industry level while the filter is on: the "
                    "tradable list is a list of 4-digit industries, and a division mixes "
                    "tradable with non-tradable.",
                    f"Se muestran solo las **{len(industries):,} industrias transables**, "
                    f"el {share:.1%} de "
                    f"{'las plazas registradas' if var1 == 'plazas_equiv' else 'los establecimientos'} "
                    f"de {place}. El tamaño de los mosaicos y las participaciones del "
                    "cursor siguen siendo las cifras de toda la economía (un mosaico que "
                    "marca 4% significa 4% de la provincia, no 4% de este subconjunto). "
                    "El detalle queda fijo en el nivel de industria mientras el filtro "
                    "está activo: la lista de transables es una lista de industrias de 4 "
                    "dígitos, y una división mezcla transables con no transables.",
                )
            )

    # ------------------------------------------------------------------ #
    # Section 2 — relative presence against a peer economy.               #
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader(t(f"What does {place} do differently?",
                   f"¿Qué hace {place} de manera distinta?"))

    if is_national:
        st.info(
            t(
                "Relative presence compares a province with a peer economy — pick a "
                "province above to see it. (Ecuador can only be compared with itself.)",
                "La presencia relativa compara una provincia con una economía de "
                "referencia; elija una provincia arriba para verla. (El Ecuador solo "
                "puede compararse consigo mismo.)",
            )
        )
    else:
        st.caption(
            t(
                "**Relative presence** = the industry's share of the province ÷ its share "
                "of the peer economy. Above ×1 the province is more concentrated in that "
                "industry than the peer, below ×1 less. It is the same location-quotient "
                "formula as the RCA behind the presence colors further down, so the two "
                "always agree.",
                "**Presencia relativa** = la participación de la industria en la provincia "
                "÷ su participación en la economía de referencia. Por encima de ×1 la "
                "provincia está más concentrada en esa industria que la referencia, y por "
                "debajo de ×1 menos. Es la misma fórmula de cociente de localización que "
                "el RCA detrás de los colores de presencia de más abajo, así que las dos "
                "siempre coinciden.",
            )
        )

        region = (
            prov_lookup.loc[province, "region"] if province in prov_lookup.index else None
        )
        region_peers = [
            code
            for code in prov_lookup.index[prov_lookup["region"] == region]
            if code != province
        ]

        rp_cols = st.columns([2, 2, 2])
        with rp_cols[0]:
            # Same reason as Section 1: a 4-digit menu cannot make a division
            # eligible or ineligible, so the filter pins the detail to class level.
            level2 = _segmented(
                t("Detail", "Detalle"), _labels(_LEVELS, _LEVELS_ES), _LEVEL_DEFAULT,
                "bu-rp-level", t(_HELP_LEVELS_EN, _HELP_LEVELS_ES),
                disabled=tradable_only,
            )
            if tradable_only:
                level2 = "class"
        with rp_cols[1]:
            var2 = _segmented(
                t("Measured in", "Medido en"), _labels(_VARS, _VARS_ES), _VAR_DEFAULT,
                "bu-rp-var", t(_HELP_VARS_EN, _HELP_VARS_ES),
            )
        with rp_cols[2]:
            peer_labels = {
                "ecuador": "Ecuador",
                "region": t("Regional peers", "Pares regionales"),
            }
            coerce_choice("bu-peer", list(peer_labels), peer_labels)
            peer_choice = st.segmented_control(
                t("Compared with", "Comparado con"),
                list(peer_labels),
                default="ecuador",
                key="bu-peer",
                format_func=lambda code: peer_labels[code],
                help=t(
                    "Ecuador — the whole country. · Regional peers — the **other** "
                    "provinces of this province's region (Costa / Sierra / Amazonia / "
                    "Isla). The province itself is always excluded from its peer group, or "
                    "a large province would be compared partly with itself.",
                    "Ecuador: todo el país. · Pares regionales: las **otras** provincias de "
                    "la región de esta provincia (Costa / Sierra / Amazonía / Insular). La "
                    "provincia misma siempre queda excluida de su grupo de pares, porque "
                    "de lo contrario una provincia grande se compararía en parte consigo "
                    "misma.",
                ),
            )
        use_region = (peer_choice or "ecuador") == "region"
        if use_region and not region_peers:
            st.info(
                t(
                    f"{place} is the only province in the {region} region, so it has no "
                    "regional peer group. Showing the comparison with Ecuador instead.",
                    f"{place} es la única provincia de la región {region}, así que no "
                    "tiene grupo de pares regionales. Se muestra la comparación con el "
                    "Ecuador.",
                )
            )
            use_region = False
        peer_codes = region_peers if use_region else None
        peer_label = (
            t(f"the rest of {region}", f"el resto de {region}") if use_region else "Ecuador"
        )

        guard_on = st.toggle(
            t(
                f"Hide industries with under {MIN_ESTAB} establishments and under "
                f"{MIN_PLAZAS:,} FTE here",
                f"Ocultar industrias con menos de {MIN_ESTAB} establecimientos y menos de "
                f"{MIN_PLAZAS:,} plazas equivalentes aquí",
            ),
            value=True,
            key="bu-guard",
            help=t(_HELP_GUARD, _HELP_GUARD_ES),
        )

        rca_df = None
        if level2 == "class" and var2 == "n_estab" and not use_region:
            # The one combination with a canonical RCA: consume the shipped column.
            density_all = load_province_density()
            rca_df = density_all[density_all["id_code"] == province]

        rp = relative_presence(comp, province, level2, var2, peer_codes, rca_df)
        rp["name"] = rp["name_es"].fillna(rp["name_en"])

        absent = rp["n_estab"] <= 0
        # The shared two-dimensional guard: thin only when **both** floors fail.
        # `value` is the plotted variable and is FTE only on the employment view, so
        # employment is read from the export's own plazas column either way — the
        # guard must not change meaning when the user flips the display variable.
        fte_here = pd.to_numeric(rp["plazas_equiv_guard"], errors="coerce").fillna(0.0)
        thin = (
            (~absent)
            & (rp["n_estab"] < MIN_ESTAB)
            & (fte_here < MIN_PLAZAS)
        )
        # The filter restricts **ranking eligibility only**: rp itself (the shipped
        # RCA at the canonical combination, the share ratio everywhere else) is
        # untouched, so a filtered-out industry keeps exactly the value it had.
        off_menu = ~rp["code"].isin(menu_here) if tradable_only else pd.Series(False, index=rp.index)
        eligible = rp[~absent & (~thin if guard_on else True) & ~off_menu]
        plottable = eligible[(eligible["rp"] > 0) & (eligible["rp"] < float("inf"))]
        n_zero = int(((eligible["rp"] <= 0) | eligible["rp"].isna()).sum())
        n_inf = int((eligible["rp"] == float("inf")).sum())

        if plottable.empty:
            st.info(
                t(
                    "No industry in this province clears the filters for this comparison.",
                    "Ninguna industria de esta provincia supera los filtros para esta "
                    "comparación.",
                )
            )
        else:
            # Ties are real at class level: an industry whose entire national
            # activity sits in this province hits the ceiling ratio
            # 1 / (province's share of the country), and several can reach it at
            # once. Break those ties by size, so the more material industry wins
            # the slot rather than whichever happened to sort first.
            top = plottable.sort_values(
                ["rp", "value"], ascending=[False, False]
            ).head(min(_TOP_N, len(plottable)))
            rest = plottable.drop(index=top.index)
            bottom = rest.sort_values(["rp", "value"], ascending=[True, False]).head(
                min(_TOP_N, len(rest))
            )
            st.plotly_chart(
                make_relative_presence_bars(
                    top, bottom, var2, GSECTOR_COLORS,
                    peer_label=peer_label, province_label=place,
                ),
                width="stretch",
                key=f"bu-rp-{province}-{level2}-{var2}-{peer_label}-{guard_on}-{tradable_only}",
            )

            # Everything the ranking leaves out is counted, never silently dropped.
            notes = [
                t(
                    f"Ranked over {len(plottable):,} of {len(rp):,} industries",
                    f"Ordenadas sobre {len(plottable):,} de {len(rp):,} industrias",
                )
            ]
            if int(absent.sum()):
                notes.append(
                    t(
                        f"{int(absent.sum()):,} are absent from {place} altogether (a "
                        "ratio of zero, which a log axis cannot show)",
                        f"{int(absent.sum()):,} están del todo ausentes en {place} (una "
                        "razón de cero, que un eje logarítmico no puede mostrar)",
                    )
                )
            if guard_on and int(thin.sum()):
                notes.append(
                    t(
                        f"{int(thin.sum()):,} have both fewer than {MIN_ESTAB} "
                        f"establishments and fewer than {MIN_PLAZAS:,} FTE here, and are "
                        "hidden by the ranking guard",
                        f"{int(thin.sum()):,} tienen a la vez menos de {MIN_ESTAB} "
                        f"establecimientos y menos de {MIN_PLAZAS:,} plazas equivalentes "
                        "aquí, y el filtro del ordenamiento las oculta",
                    )
                )
            n_off_menu = int((off_menu & ~absent & (~thin if guard_on else True)).sum())
            if n_off_menu:
                notes.append(
                    t(
                        f"{n_off_menu:,} are present here but not tradable outside the "
                        "province (or are extractive) and are held out of the ranking by "
                        "the tradable focus — their relative presence is unchanged, only "
                        "their eligibility",
                        f"{n_off_menu:,} están presentes aquí pero no son transables fuera "
                        "de la provincia (o son extractivas) y el enfoque en transables las "
                        "deja fuera del ordenamiento; su presencia relativa no cambia, solo "
                        "su elegibilidad",
                    )
                )
            if n_zero:
                notes.append(
                    t(
                        f"{n_zero:,} have establishments here but report no FTE",
                        f"{n_zero:,} tienen establecimientos aquí pero no reportan plazas",
                    )
                    if var2 == "plazas_equiv"
                    else t(
                        f"{n_zero:,} have no share to compare",
                        f"{n_zero:,} no tienen participación que comparar",
                    )
                )
            if n_inf:
                notes.append(
                    t(
                        f"{n_inf:,} are present here but absent from {peer_label} "
                        "(an unbounded ratio, equally unplottable)",
                        f"{n_inf:,} están presentes aquí pero ausentes en {peer_label} "
                        "(una razón sin límite, igualmente no graficable)",
                    )
                )
            st.caption("; ".join(notes) + ".")
            if level2 == "class" and var2 == "n_estab" and not use_region:
                st.caption(
                    t(
                        "At this combination the chart plots the **shipped RCA** from the "
                        "relatedness analysis — the identical column that colors the "
                        "presence states in the next section, not a recomputation of it.",
                        "En esta combinación el gráfico usa el **RCA publicado** del "
                        "análisis de relación entre industrias, es decir la misma columna "
                        "que colorea los estados de presencia de la siguiente sección y no "
                        "un recálculo de ella.",
                    )
                )
            st.caption(
                t(
                    "The employment and establishment views are different stories, not a "
                    "rescaling of each other (they agree only at ρ ≈ 0.56 across "
                    "industries): an industry can be over-represented in units and "
                    "under-represented in the positions those units hold.",
                    "Las vistas de empleo y de establecimientos cuentan historias "
                    "distintas, no son un reescalamiento una de la otra (coinciden solo con "
                    "ρ ≈ 0,56 entre industrias): una industria puede estar "
                    "sobrerrepresentada en unidades y subrepresentada en las plazas que "
                    "esas unidades sostienen.",
                )
            )

    # ------------------------------------------------------------------ #
    # Section 3 — feasibility x attractiveness opportunities.             #
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader(
        t(f"What is worth diversifying into in {place}?",
          f"¿Hacia qué vale la pena diversificar en {place}?")
    )

    if is_national:
        st.info(
            t(
                "Relatedness density is a property of a province — pick a province above "
                "to see its opportunity plane.",
                "La densidad de industrias relacionadas es una propiedad de una "
                "provincia; elija una arriba para ver su plano de oportunidades.",
            )
        )
        return

    st.caption(
        t(
            "Each dot is an industry: **feasibility** on the x-axis — a mix of how much of "
            "the industry the province already has and how much of what it needs the rest "
            "of the local economy provides, which you set under *Advanced settings* — and "
            "**attractiveness** on the y-axis (what it pays, or the jobs a typical firm "
            "holds, measured nationally). Grey dots are the province's whole opportunity "
            "space; the highlighted ones are the best-scoring industries in each presence "
            "state. Dot size is the employment the industry holds nationally — a "
            "materiality screen, never jobs a new entrant would create.",
            "Cada punto es una industria: la **factibilidad** en el eje x, una mezcla entre "
            "cuánto de la industria ya tiene la provincia y cuánto de lo que necesita se lo "
            "aporta el resto de la economía local, que usted fija en *Parámetros "
            "avanzados*, y el **atractivo** en el eje y (lo que paga, o los puestos que "
            "sostiene una empresa típica, medidos a nivel nacional). Los puntos grises son "
            "todo el espacio de oportunidades de la provincia; los destacados son las "
            "industrias con mejor puntaje en cada estado de presencia. El tamaño del punto "
            "es el empleo que la industria sostiene a nivel nacional, un filtro de "
            "materialidad y nunca empleos que crearía un nuevo entrante.",
        )
    )

    ctrl = st.columns([3, 1])
    with ctrl[0]:
        var3 = _segmented(
            t("Attractiveness metric", "Métrica de atractivo"),
            _labels(_ATTR_CHOICES, _ATTR_CHOICES_ES), _ATTR_DEFAULT, "bu-attr",
            t(_HELP_ATTR_EN, _HELP_ATTR_ES),
        )
    with ctrl[1]:
        top_k = st.number_input(
            t("Per state", "Por estado"),
            min_value=1, max_value=15, value=BU_DEFAULT_TOP_K, step=1, key="bu-k",
            help=t(
                "How many industries to highlight in each of the three presence states.",
                "Cuántas industrias destacar en cada uno de los tres estados de presencia.",
            ),
        )
    # Same guard, same shape as Section 2's: a toggle the reader can flip and audit,
    # on the main surface because it decides what may be recommended rather than how
    # the recommendations are ordered.
    support_on = st.toggle(
        t(
            f"Only recommend industries with at least {MIN_ESTAB} establishments or "
            f"{MIN_PLAZAS:,.0f} FTE here",
            f"Recomendar solo industrias con al menos {MIN_ESTAB} establecimientos o "
            f"{MIN_PLAZAS:,.0f} plazas equivalentes aquí",
        ),
        value=BU_DEFAULT_SUPPORT,
        key="bu-support",
        help=t(_HELP_SUPPORT, _HELP_SUPPORT_ES),
    )
    # The second eligibility guard, same shape as the one above and deliberately so.
    # The support guard is about **evidence** (is there real activity behind this
    # presence?); this one is about **measurability** of the density half of
    # feasibility (is there enough proximity mass for the ratio to mean anything?).
    # Both are eligibility only — never a demotion and never a re-ranking.
    connected_on = st.toggle(
        t("Only recommend well-connected industries",
          "Recomendar solo industrias bien conectadas"),
        value=BU_DEFAULT_CONNECTED,
        key="bu-connected",
        help=t(_HELP_CONNECTED, _HELP_CONNECTED_ES),
    )
    # The two rank-space knobs live behind an expander: they change the ordering,
    # not the data, and putting them on the main surface implied the shortlist was
    # a tuned result rather than a reading of the plane.
    with st.expander(
        t("Advanced settings — what *feasible* means, and how the list is ordered",
          "Parámetros avanzados: qué significa *factible* y cómo se ordena la lista")
    ):
        adv = st.columns(2)
        with adv[0]:
            beta = st.slider(
                t("Feasibility mix — 0 = related capabilities, 1 = own presence",
                  "Mezcla de factibilidad: 0 = capacidades relacionadas, "
                  "1 = presencia propia"),
                min_value=0.0, max_value=1.0, value=BU_DEFAULT_BETA, step=0.05,
                key="bu-beta", help=t(_HELP_BETA, _HELP_BETA_ES),
            )
        with adv[1]:
            weight = st.slider(
                t("Weight on feasibility (vs attractiveness)",
                  "Peso de la factibilidad (frente al atractivo)"),
                min_value=0.0, max_value=1.0, value=BU_DEFAULT_WEIGHT, step=0.05,
                key="bu-weight", help=t(_HELP_WEIGHT, _HELP_WEIGHT_ES),
            )
    # One density on the page: the **related-only** one. Own presence enters
    # feasibility explicitly through β, so the with-self column would count it
    # twice at a ratio that varies arbitrarily by industry (see the as-built).
    value_col = "density_cont_noself"

    labels = load_class_labels()
    profile = load_labor_profile().copy()
    profile["class_name"] = profile["codigo_clase"].map(labels).fillna(profile["class_name"])
    # Section-letter-prefixed codes, for any class the capability panels touch.
    display_codes = dict(zip(profile["codigo_clase"], profile["display_code"]))
    density = load_province_density()
    # Full slice (kept for the capability expander, which needs every industry's
    # presence in the province, not only the ones on the plane).
    prov_density_full = density.loc[
        density["id_code"] == province,
        ["codigo_clase", "rca", "state", "intensity", value_col],
    ]
    prov_density = prov_density_full[["codigo_clase", "rca", "state", value_col]].rename(
        columns={value_col: "density"}
    )

    plane = profile.merge(prov_density, on="codigo_clase", how="inner")
    n_universe = len(plane)
    if tradable_only:
        # The plane **is** the menu when the filter is on — context dots included.
        # Restricting before the percentile ranks is what makes them "over the
        # plotted classes"; the raw density and RCA of every survivor are untouched.
        plane = plane[plane["codigo_clase"].isin(menu_here)].copy()
    # Establishment-booked employment located in this province — the same
    # attribution as plazas_equiv_estab_nat, so the two hover rows compare 1:1.
    prov_emp = load_labor_class_employment()
    prov_emp = prov_emp.loc[
        prov_emp["id_code"] == province, ["codigo_clase", "plazas_equiv", "n_estab"]
    ].rename(columns={"plazas_equiv": "plazas_equiv_prov", "n_estab": "n_estab_prov"})
    plane = plane.merge(prov_emp, on="codigo_clase", how="left")
    plane[["plazas_equiv_prov", "n_estab_prov"]] = plane[
        ["plazas_equiv_prov", "n_estab_prov"]
    ].fillna(0.0)
    # The **employment-base** presence state for the same province, from the
    # relatedness notebook's robustness export. It feeds the ✓ Both bases column and
    # nothing else: the states, the plane and every score stay establishment-based
    # (see the as-built's base ruling). A column, never a composite.
    plazas_states = load_province_density(base="plazas")
    plazas_states = plazas_states.loc[
        plazas_states["id_code"] == province, ["codigo_clase", "state", "rca"]
    ].rename(columns={"state": "state_plazas", "rca": "rca_plazas"})
    plane = plane.merge(plazas_states, on="codigo_clase", how="left")
    plane["both_bases"] = plane["state"] == plane["state_plazas"]
    # Thin-neighbourhood flag: how much proximity mass links this industry to the
    # rest of the economy at all. A class with almost none has a density that is
    # a ratio of two tiny numbers. Since v4 this is an **eligibility** input (the
    # ``bu-connected`` guard below) rather than a column: the flag is still read on
    # the searched industry's own card, which is where the reader can see how little
    # sits behind the number.
    support = load_density_support()
    plane = plane.merge(
        support[["codigo_clase", "phi_mass_noself", "n_neighbors_pos"]],
        on="codigo_clase", how="left",
    )
    plane["thin_neighborhood"] = plane["phi_mass_noself"] < THIN_NEIGHBORHOOD_THRESHOLD

    # The x-axis **is** the feasibility mix, so the plot and the score can never
    # disagree about which industry is more feasible.
    plane["x"] = feasibility_mix(plane, var3, float(beta))
    plane["score"] = opportunity_score(plane, var3, float(weight), feasibility_col="x")
    scored = plane.dropna(subset=["score"])
    if scored.empty:
        st.info(
            t(
                "No industry in this province carries both a density and this metric.",
                "Ninguna industria de esta provincia tiene a la vez una densidad y esta "
                "métrica.",
            )
        )
        return

    # Eligibility, not demotion: the held-out rows keep their exported state, stay on
    # the plane as context dots and stay in the percentile-rank universe above, so no
    # survivor's feasibility or score moves — only who may be recommended. Both guards
    # land **here**: after `feasibility_mix` / `opportunity_score` and before the
    # top-`k` cut, which is what makes the guarantee hold.
    support_mask = support_eligible(
        scored["state"], scored["n_estab_prov"], scored["plazas_equiv_prov"],
        MIN_ESTAB, MIN_PLAZAS,
    )
    eligible_mask = (
        support_mask if support_on else pd.Series(True, index=scored.index)
    )
    # A missing support row leaves `thin_neighborhood` False (NaN < threshold is
    # False), so a class the industry-space export does not cover is treated as
    # well-connected — the same reading the flag has always had, unchanged here so
    # that flipping this toggle cannot move membership for an unrelated reason. No
    # class on the menu plane is currently in that position.
    connected_mask = (
        connected_eligible(scored["thin_neighborhood"])
        if connected_on
        else pd.Series(True, index=scored.index)
    )
    recommendable = scored[eligible_mask & connected_mask]
    if recommendable.empty:
        st.info(
            t(
                "Every scorable industry in this province is held out by the eligibility "
                "guards. Turn one of them off above to see the ranking without it.",
                "Todas las industrias puntuables de esta provincia quedan fuera por los "
                "filtros de elegibilidad. Desactive uno arriba para ver el ordenamiento "
                "sin él.",
            )
        )
        return
    highlighted = (
        recommendable.sort_values("score", ascending=False)
        .groupby("state", group_keys=False)
        .head(int(top_k))
        .copy()
    )
    # Robust picks: the industries the other attractiveness metric would also have
    # chosen, at these same β / weight / filter settings. The pay-vs-volume choice
    # is the one this section exists to surface (26% of industries move more than
    # 25 percentile points between the two), so agreement is worth marking — but as
    # a column, not as a composite that would hide the disagreement.
    other_var = next(code for code in _ATTR_CHOICES if code != var3)
    robust_codes = _shortlist_codes(
        plane, other_var, beta, weight, int(top_k), support_on, connected_on
    )
    highlighted["robust"] = highlighted["codigo_clase"].isin(robust_codes)
    plane["is_matched"] = plane["codigo_clase"].isin(set(highlighted["codigo_clase"]))

    # Look up one industry wherever it sits — the bridge back to Section 2,
    # where a user meets an industry by name and wants to know what the plane
    # says about it. The shortlist cannot answer that: it is a top-k, so most
    # industries are on the plot as grey context dots.
    find_options = plane.sort_values("display_code")["codigo_clase"].tolist()
    # Flipping the filter on can strip the searched industry out of the options.
    # Reset before the widget renders (the ✕ button's pattern) — Streamlit raises
    # if a selectbox's stored value is not among its options.
    if st.session_state.get("bu-find") is not None and st.session_state["bu-find"] not in find_options:
        st.session_state["bu-find"] = None
    search_cols = st.columns([12, 1, 7], vertical_alignment="bottom")
    with search_cols[1]:
        if st.button(
            "✕", key="bu-find-clear",
            help=t("Clear the searched industry", "Quitar la industria buscada"),
            disabled=st.session_state.get("bu-find") is None,
        ):
            st.session_state["bu-find"] = None
            st.rerun()
    with search_cols[0]:
        coerce_choice("bu-find", find_options)
        find_labels = dict(
            zip(plane["codigo_clase"], plane["display_code"] + " — " + plane["class_name"])
        )
        focus_code = st.selectbox(
            t("Find an industry in the plot", "Buscar una industria en el gráfico"),
            options=find_options,
            index=None,
            format_func=lambda c: find_labels.get(c, c),
            placeholder=t("Type a code or a name to search",
                          "Escriba un código o un nombre"),
            key="bu-find",
            help=t(
                "Type a code or part of a name. The industry is ringed on the plot "
                "wherever it sits — including when the shortlist passed it over.",
                "Escriba un código o parte de un nombre. La industria queda marcada con un "
                "anillo en el gráfico donde sea que esté, incluso cuando la lista corta la "
                "haya pasado por alto.",
            ),
        )

    st.plotly_chart(
        make_attractiveness_feasibility_scatter(
            plane, var3, STATE_COLORS, state_labels(), province_name=place,
            label_points=len(highlighted) <= 21,
            focus_code=focus_code,
            x_title=t("Feasibility (percentile mix)",
                      "Factibilidad (mezcla de percentiles)"),
        ),
        width="stretch",
        key=f"bu-plane-{province}-{var3}-{tradable_only}-{top_k}-{beta}-{weight}-{focus_code}",
    )

    if focus_code is not None:
        row = plane[plane["codigo_clase"] == focus_code].iloc[0]
        on_plane = pd.notna(row["x"]) and pd.notna(row[var3])
        in_shortlist = bool(row["is_matched"])
        if not on_plane:
            st.info(
                t(
                    f"**{row['display_code']} · {row['class_name']}** has no wage or "
                    "jobs-per-firm figure (no employing firm reports both), so it cannot "
                    "be placed on this plane. Its related-only relatedness density in "
                    f"{place} is {row['density']:.3f} and its relative presence "
                    f"×{row['rca']:.2f}.",
                    f"**{row['display_code']} · {row['class_name']}** no tiene cifra de "
                    "salario ni de puestos por empresa (ninguna empresa empleadora reporta "
                    "ambas), así que no puede ubicarse en este plano. Su densidad de "
                    f"industrias relacionadas en {place} es {row['density']:.3f} y su "
                    f"presencia relativa ×{row['rca']:.2f}.",
                )
            )
        else:
            place_words = (
                t("It is in the shortlist above.", "Está en la lista corta de arriba.")
                if in_shortlist
                else t(
                    "It is **not** in the shortlist — the shortlist is the top "
                    f"{int(top_k)} of its presence state by the current weighting, not a "
                    "judgement that this industry is unpromising.",
                    "**No** está en la lista corta: la lista corta son las "
                    f"{int(top_k)} mejores de su estado de presencia con la ponderación "
                    "actual, no un juicio de que esta industria sea poco prometedora.",
                )
            )
            state_label = state_labels().get(row["state"], row["state"])
            x_rank = int((scored["x"] > row["x"]).sum()) + 1
            d_rank = int((scored["density"] > row["density"]).sum()) + 1
            st.markdown(
                t(
                    f"**{row['display_code']} · {row['class_name']}** "
                    f"({row['gsector_label']}) — {state_label}. "
                    f"Feasibility **{row['x']:.2f}** "
                    f"(rank {x_rank} of {len(scored)}), built from relative presence "
                    f"×{row['rca']:.2f} and related-only relatedness density "
                    f"{row['density']:.3f} (rank {d_rank} of {len(scored)}); "
                    f"average wage paid **${fmt_number(row['wage_per_fte'], 0)}** "
                    f"(P{fmt_number(row['wage_per_fte_pctile_gsector'], 0)} in its "
                    "sector), average jobs per firm "
                    f"**{fmt_number(row['fte_per_firm'], 1)}** "
                    f"(P{fmt_number(row['fte_per_firm_pctile_gsector'], 0)}). "
                    f"It holds {fmt_number(row['plazas_equiv_prov'], 0)} FTE across "
                    f"{fmt_number(row['n_estab_prov'], 0)} establishments here, of "
                    f"{fmt_number(row['plazas_equiv_estab_nat'], 0)} nationally. "
                    f"{place_words}",
                    f"**{row['display_code']} · {row['class_name']}** "
                    f"({row['gsector_label']}): {state_label}. "
                    f"Factibilidad **{row['x']:.2f}** "
                    f"(puesto {x_rank} de {len(scored)}), construida a partir de una "
                    f"presencia relativa de ×{row['rca']:.2f} y una densidad de "
                    f"industrias relacionadas de {row['density']:.3f} "
                    f"(puesto {d_rank} de {len(scored)}); salario promedio pagado "
                    f"**${fmt_number(row['wage_per_fte'], 0)}** "
                    f"(P{fmt_number(row['wage_per_fte_pctile_gsector'], 0)} en su "
                    "sector), empleo promedio por empresa "
                    f"**{fmt_number(row['fte_per_firm'], 1)}** "
                    f"(P{fmt_number(row['fte_per_firm_pctile_gsector'], 0)}). "
                    f"Sostiene {fmt_number(row['plazas_equiv_prov'], 0)} plazas "
                    f"equivalentes en {fmt_number(row['n_estab_prov'], 0)} "
                    "establecimientos aquí, de "
                    f"{fmt_number(row['plazas_equiv_estab_nat'], 0)} a nivel nacional. "
                    f"{place_words}",
                )
            )
        if bool(row["thin_neighborhood"]):
            st.warning(
                t(
                    f"⚠ **{row['display_code']} sits in a thin neighbourhood.** It has "
                    "measurable proximity to only "
                    f"{fmt_number(row['n_neighbors_pos'], 0)} industries and its total "
                    f"off-diagonal proximity mass is {row['phi_mass_noself']:.3f}, under "
                    f"the {THIN_NEIGHBORHOOD_THRESHOLD} floor. Its related-only density is "
                    "a ratio of two very small numbers — read that half of its feasibility "
                    "as noise rather than as evidence, and lean on its own presence "
                    "instead.",
                    f"⚠ **{row['display_code']} está en un vecindario delgado.** Tiene "
                    "proximidad medible con solo "
                    f"{fmt_number(row['n_neighbors_pos'], 0)} industrias y su masa total "
                    f"de proximidad fuera de la diagonal es {row['phi_mass_noself']:.3f}, "
                    f"por debajo del piso de {THIN_NEIGHBORHOOD_THRESHOLD}. Su densidad de "
                    "industrias relacionadas es una razón entre dos números muy pequeños: "
                    "lea esa mitad de su factibilidad como ruido y no como evidencia, y "
                    "apóyese en su propia presencia.",
                )
            )
        _capability_expander(row, place, prov_density_full, labels, display_codes)

    unscored = int(len(plane) - len(scored))
    st.caption(
        t(
            f"The **{int(top_k)} best-scoring industries in each presence state** are "
            "highlighted: blue (absent) and amber (present, not specialized) are the "
            "diversification story — industries the province could add; green "
            "(specialized) is where it could deepen what it already has. Ranking mixes "
            "feasibility and attractiveness in percentile space with the weight under "
            "*Advanced settings* — it orders the shortlist and is never shown as a "
            "measurement.",
            f"Se destacan las **{int(top_k)} industrias con mejor puntaje en cada estado "
            "de presencia**: el azul (ausente) y el ámbar (presente, no especializada) son "
            "la historia de diversificación, es decir industrias que la provincia podría "
            "añadir; el verde (especializada) es donde podría profundizar lo que ya tiene. "
            "El ordenamiento mezcla factibilidad y atractivo en espacio de percentiles con "
            "el peso de *Parámetros avanzados*: ordena la lista corta y nunca se presenta "
            "como una medición.",
        )
        + (
            t(
                f" {unscored} industries report neither pay nor jobs per firm (no "
                "employing firm files both), so they cannot be placed on this plane at all "
                "and can never be highlighted.",
                f" {unscored} industrias no reportan ni paga ni puestos por empresa "
                "(ninguna empresa empleadora presenta ambas), así que no pueden ubicarse "
                "en este plano y nunca pueden destacarse.",
            )
            if unscored
            else ""
        )
    )
    st.caption(
        t(
            f"**{len(plane):,} of {n_universe:,} industries shown.** ",
            f"**Se muestran {len(plane):,} de {n_universe:,} industrias.** ",
        )
        + (
            t(
                "The plane is the tradable menu: only industries whose output can leave "
                "the province, mining/quarrying/hunting excluded. Percentile ranks — and "
                "therefore feasibility, the score and the median crosshairs — run over "
                "these industries alone; every raw density, relative presence and wage is "
                "identical to the unfiltered view.",
                "El plano es el menú de transables: solo industrias cuya producción puede "
                "salir de la provincia, con minas, canteras y caza excluidas. Los "
                "percentiles, y por lo tanto la factibilidad, el puntaje y las líneas de "
                "mediana, se calculan solo sobre estas industrias; toda densidad, "
                "presencia relativa y salario crudo es idéntico a la vista sin filtro.",
            )
            if tradable_only
            else t(
                "The whole registered economy, tradable or not. Turn on **Focus on "
                "tradable industries** above to rank only what could be sold outside the "
                "province.",
                "Toda la economía registrada, transable o no. Active **Enfocarse en "
                "industrias transables** arriba para ordenar solo lo que podría venderse "
                "fuera de la provincia.",
            )
        )
    )

    # A "top k" implies there were more than k to choose from. Where there were
    # not, say so — a province with four absent industries in the whole registry
    # is not being shown its three best options, it is being shown all of them.
    available = recommendable["state"].value_counts()
    scope = (
        t("the tradable menu", "el menú de transables")
        if tradable_only
        else t("the entire registry", "todo el registro")
    )
    # With the guard on these are the *recommendable* classes, not merely the scorable
    # ones, and the two can differ by a lot — saying "scorable" here would contradict
    # the dots the plane is showing. Name what is counted, and where the rest went.
    qualifier = (
        t("recommendable", "recomendable") if support_on else t("scorable", "puntuable")
    )
    # Two reasons a state can be emptied, counted **disjointly** so `available` plus
    # the two equals the state's total and neither sentence can claim the other's
    # rows: the support guard bites first, connectedness on what is left of it.
    held_by_state = (
        scored.loc[~eligible_mask, "state"].value_counts()
        if support_on
        else pd.Series(dtype="int64")
    )
    thin_by_state = (
        scored.loc[eligible_mask & ~connected_mask, "state"].value_counts()
        if connected_on
        else pd.Series(dtype="int64")
    )
    for state_key in ["absent", "present_not_spec", "specialized"]:
        n_available = int(available.get(state_key, 0))
        n_state_held = int(held_by_state.get(state_key, 0))
        n_state_thin = int(thin_by_state.get(state_key, 0))
        clauses = []
        if n_state_held:
            clauses.append(
                t(
                    f"**{n_state_held}** "
                    + ("is" if n_state_held == 1 else "are")
                    + " present here but too thinly for the support guard to let "
                    + ("it" if n_state_held == 1 else "them")
                    + " be recommended",
                    f"**{n_state_held}** "
                    + ("está" if n_state_held == 1 else "están")
                    + " presente"
                    + ("" if n_state_held == 1 else "s")
                    + " aquí, pero de forma demasiado delgada para que el filtro de "
                    "actividad real permita recomendar"
                    + ("la" if n_state_held == 1 else "las"),
                )
            )
        if n_state_thin:
            clauses.append(
                t(
                    f"**{n_state_thin}** "
                    + ("sits" if n_state_thin == 1 else "sit")
                    + " in too thin a neighbourhood for the connectedness guard",
                    f"**{n_state_thin}** "
                    + ("está" if n_state_thin == 1 else "están")
                    + " en un vecindario demasiado delgado para el filtro de conectividad",
                )
            )
        guard_note = (
            t(
                f" A further {', and '.join(clauses)}.",
                f" Además, {', y '.join(clauses)}.",
            )
            if clauses
            else ""
        )
        label = state_labels().get(state_key, state_key).split(" (")[0].lower()
        if n_available == 0 and (n_state_held or n_state_thin):
            # The state is not empty — the guard emptied it. The margin sentences
            # below would assert the opposite ("every industry present here is
            # already specialized"), which is false when 96 of them are held out.
            reasons = []
            if n_state_held:
                reasons.append(
                    t(
                        f"**{n_state_held}** are present on both under {MIN_ESTAB} "
                        f"establishments and under {MIN_PLAZAS:,.0f} FTE here",
                        f"**{n_state_held}** están presentes con menos de {MIN_ESTAB} "
                        f"establecimientos y menos de {MIN_PLAZAS:,.0f} plazas "
                        "equivalentes aquí",
                    )
                )
            if n_state_thin:
                reasons.append(
                    t(
                        f"**{n_state_thin}** sit in a neighbourhood too thin for their "
                        "related-capability score to mean anything",
                        f"**{n_state_thin}** están en un vecindario demasiado delgado para "
                        "que su puntaje de capacidades relacionadas signifique algo",
                    )
                )
            st.warning(
                t(
                    f"⚠ **No *{label}* industry in {scope} is eligible to recommend** in "
                    f"{place}: ",
                    f"⚠ **Ninguna industria *{label}* de {scope} es elegible para "
                    f"recomendar** en {place}: ",
                )
                + "; ".join(reasons)
                + t(
                    ". That is a finding about how thin this margin is, not an empty "
                    "margin — turn the guards off above to see them ranked.",
                    ". Eso es un hallazgo sobre lo delgado que es este margen, no un "
                    "margen vacío: desactive los filtros arriba para verlas ordenadas.",
                )
            )
            continue
        if n_available == 0:
            # An empty margin is a finding, not a state to skip over: Pichincha
            # with the filter on has no scorable absent industry at all, and
            # rendering two colors of dots without saying so reads as a bug.
            meaning = {
                "absent": t(
                    f"there is no tradable industry {place} is missing that this analysis "
                    "can score — its extensive margin is empty, so diversification here "
                    "means **deepening and upgrading what it already has** rather than "
                    "adding something new",
                    f"no hay ninguna industria transable que {place} no tenga y que este "
                    "análisis pueda puntuar: su margen extensivo está vacío, así que "
                    "diversificar aquí significa **profundizar y mejorar lo que ya tiene** "
                    "en lugar de añadir algo nuevo",
                ),
                "present_not_spec": t(
                    f"every tradable industry present in {place} is already specialized "
                    "there — it has no foothold industries left to grow into full "
                    "specializations",
                    f"todas las industrias transables presentes en {place} ya están "
                    "especializadas ahí: no le quedan industrias con pie de apoyo para "
                    "convertir en especializaciones plenas",
                ),
                "specialized": t(
                    f"{place} is not specialized in any tradable industry that can be "
                    "scored, so it has no core to deepen and the whole story is entry",
                    f"{place} no está especializada en ninguna industria transable "
                    "puntuable, así que no tiene núcleo que profundizar y toda la historia "
                    "es de entrada",
                ),
            }[state_key]
            st.warning(
                t(
                    f"⚠ **No {qualifier} *{label}* industry in {scope}** for {place}: "
                    f"{meaning}.{guard_note}",
                    f"⚠ **Ninguna industria *{label}* {qualifier} en {scope}** para "
                    f"{place}: {meaning}.{guard_note}",
                )
            )
            continue
        if n_available > int(top_k):
            continue
        picks = highlighted[highlighted["state"] == state_key]
        best_rank = int((scored["x"] > picks["x"].max()).sum()) + 1 if not picks.empty else 0
        tail = ""
        if best_rank > 0.75 * len(scored):
            tail = t(
                f" The most feasible of them ranks {best_rank:,} of {len(scored):,} on "
                "feasibility — near the bottom of the province's whole opportunity "
                "space, so read these as the residual rather than as candidates.",
                f" La más factible de ellas ocupa el puesto {best_rank:,} de "
                f"{len(scored):,} en factibilidad, cerca del fondo de todo el espacio de "
                "oportunidades de la provincia, así que léalas como el residuo y no como "
                "candidatas.",
            )
        st.warning(
            t(
                f"⚠ {place} has only **{n_available}** {qualifier} *{label}* "
                f"industr{'y' if n_available == 1 else 'ies'} in {scope}, so all of "
                "them are highlighted — this is the whole margin, not a top "
                f"{int(top_k)}.{guard_note}{tail}",
                f"⚠ {place} tiene solo **{n_available}** "
                f"industria{'' if n_available == 1 else 's'} *{label}* {qualifier} en "
                f"{scope}, así que todas están destacadas: este es el margen completo y no "
                f"un top {int(top_k)}.{guard_note}{tail}",
            )
        )

    # What the guard is holding out, counted on the frame on screen — and, when the
    # cut is small enough to name, named, so a reader can check the call rather than
    # take it on trust.
    n_held = int((~support_mask).sum())
    if support_on and n_held:
        would_be = (
            scored[~support_mask]
            .sort_values("score", ascending=False)
            .groupby("state", group_keys=False)
            .head(int(top_k))
        )
        unit = t("est. / ", "estab. / ")
        fte_word = t(" FTE)", " plazas)")
        named = ", ".join(
            f"**{row['display_code']}** ({fmt_number(row['n_estab_prov'], 0)} {unit}"
            f"{fmt_number(row['plazas_equiv_prov'], 1)}{fte_word}"
            for _, row in would_be.head(5).iterrows()
        )
        st.caption(
            t(
                f"🔒 The support guard holds **{n_held:,}** of {len(scored):,} scorable "
                "industries out of the shortlist — they are present here but under "
                f"{MIN_ESTAB} establishments *and* under {MIN_PLAZAS:,.0f} FTE, so their "
                "presence is a ratio rather than an industry. "
                f"**{len(would_be)}** of them would otherwise have been picked",
                f"🔒 El filtro de actividad real deja **{n_held:,}** de {len(scored):,} "
                "industrias puntuables fuera de la lista corta: están presentes aquí pero "
                f"con menos de {MIN_ESTAB} establecimientos *y* menos de "
                f"{MIN_PLAZAS:,.0f} plazas equivalentes, así que su presencia es una razón "
                f"y no una industria. **{len(would_be)}** de ellas habrían sido elegidas",
            )
            + (t(f", led by {named}.", f", encabezadas por {named}.") if named else ".")
            + t(
                " They are still on the plot as grey dots, with their exported presence "
                "state unchanged. Turn the guard off above to rank them.",
                " Siguen en el gráfico como puntos grises, con su estado de presencia "
                "exportado sin cambios. Desactive el filtro arriba para ordenarlas.",
            )
        )
    elif not support_on and n_held:
        # Guard off: the count that matters is how many thin industries actually made
        # the shortlist, not how many were eligible to.
        thin_picks = highlighted.loc[~support_mask.reindex(highlighted.index, fill_value=False)]
        unit = t("est. / ", "estab. / ")
        fte_word = t(" FTE)", " plazas)")
        named = ", ".join(
            f"**{row['display_code']}** ({fmt_number(row['n_estab_prov'], 0)} {unit}"
            f"{fmt_number(row['plazas_equiv_prov'], 1)}{fte_word}"
            for _, row in thin_picks.head(5).iterrows()
        )
        st.caption(
            t(
                f"⚠ The support guard is **off**: {n_held:,} of {len(scored):,} scorable "
                f"industries are present here on both under {MIN_ESTAB} establishments and "
                f"under {MIN_PLAZAS:,.0f} FTE, and **{len(thin_picks)}** of the "
                f"{len(highlighted)} highlighted are such industries — a presence built on "
                "a handful of units",
                f"⚠ El filtro de actividad real está **desactivado**: {n_held:,} de "
                f"{len(scored):,} industrias puntuables están presentes aquí con menos de "
                f"{MIN_ESTAB} establecimientos y menos de {MIN_PLAZAS:,.0f} plazas "
                f"equivalentes, y **{len(thin_picks)}** de las {len(highlighted)} "
                "destacadas son de ese tipo, es decir una presencia construida sobre un "
                "puñado de unidades",
            )
            + (f": {named}." if named else ".")
        )
    # The connectedness guard's own disclosure, below the support guard's so the two
    # read as a pair in the order the toggles appear (evidence, then measurability).
    n_thin_pool = int((~connected_eligible(scored["thin_neighborhood"]) & eligible_mask).sum())
    if connected_on and n_thin_pool:
        # What the filter actually removed from *this* shortlist: replay the same cut
        # with the filter off and diff. Derived from the frame on screen, so the
        # caption cannot drift from the membership it describes — and it is exact,
        # because every row the unfiltered cut adds is thin by construction (a
        # well-connected row that made the wider pool's top-k makes the narrower
        # pool's top-k too).
        unfiltered = (
            scored[eligible_mask]
            .sort_values("score", ascending=False)
            .groupby("state", group_keys=False)
            .head(int(top_k))
        )
        removed = unfiltered[~connected_eligible(unfiltered["thin_neighborhood"])]
        named = ", ".join(
            f"**{row['display_code']}** "
            + t(
                f"(mass {row['phi_mass_noself']:.3f} across "
                f"{fmt_number(row['n_neighbors_pos'], 0)} industries)",
                f"(masa {row['phi_mass_noself']:.3f} en "
                f"{fmt_number(row['n_neighbors_pos'], 0)} industrias)",
            )
            for _, row in removed.head(5).iterrows()
        )
        st.caption(
            t(
                f"🔗 The connectedness guard holds **{n_thin_pool:,}** of "
                f"{int(eligible_mask.sum()):,} otherwise-eligible industries out of the "
                "shortlist — their off-diagonal proximity mass is under "
                f"{THIN_NEIGHBORHOOD_THRESHOLD}, so the related-capability half of their "
                "feasibility is a ratio of two very small numbers. "
                f"**{len(removed)}** of them would otherwise have been picked",
                f"🔗 El filtro de conectividad deja **{n_thin_pool:,}** de "
                f"{int(eligible_mask.sum()):,} industrias por lo demás elegibles fuera de "
                "la lista corta: su masa de proximidad fuera de la diagonal es menor a "
                f"{THIN_NEIGHBORHOOD_THRESHOLD}, así que la mitad de capacidades "
                "relacionadas de su factibilidad es una razón entre dos números muy "
                f"pequeños. **{len(removed)}** de ellas habrían sido elegidas",
            )
            + (t(f", led by {named}.", f", encabezadas por {named}.") if named else ".")
            + t(
                " They are still on the plot as grey dots. Turn the guard off above to "
                "rank them.",
                " Siguen en el gráfico como puntos grises. Desactive el filtro arriba para "
                "ordenarlas.",
            )
        )
    elif not connected_on:
        n_thin_picks = int(highlighted["thin_neighborhood"].sum())
        if n_thin_picks:
            thin_codes = ", ".join(
                highlighted.loc[highlighted["thin_neighborhood"], "display_code"]
            )
            st.caption(
                t(
                    f"⚠ The connectedness guard is **off**: {n_thin_picks} of the "
                    f"{len(highlighted)} highlighted industries sit in a **thin "
                    f"neighbourhood** ({thin_codes}) — they are barely connected to the "
                    "rest of the economy, with total proximity mass under "
                    f"{THIN_NEIGHBORHOOD_THRESHOLD}, so the related-capability half of "
                    "their feasibility is a ratio of two very small numbers. Look them up "
                    "in the search box above to see how little sits behind it.",
                    f"⚠ El filtro de conectividad está **desactivado**: {n_thin_picks} de "
                    f"las {len(highlighted)} industrias destacadas están en un "
                    f"**vecindario delgado** ({thin_codes}), es decir apenas conectadas "
                    "con el resto de la economía, con una masa de proximidad total menor a "
                    f"{THIN_NEIGHBORHOOD_THRESHOLD}, así que la mitad de capacidades "
                    "relacionadas de su factibilidad es una razón entre dos números muy "
                    "pequeños. Búsquelas en el cuadro de arriba para ver lo poco que hay "
                    "detrás.",
                )
            )
    n_robust = int(highlighted["robust"].sum())
    st.caption(
        t(
            f"✓ **Robust**: {n_robust} of the {len(highlighted)} highlighted industries "
            "would also have made the list under the *other* attractiveness metric at "
            "these settings. The rest are picks that depend on whether you are after pay "
            "or headcount — the trade-off this section exists to show, which is why the "
            "two are never averaged into one number.",
            f"✓ **Robustas**: {n_robust} de las {len(highlighted)} industrias destacadas "
            "también habrían entrado en la lista con la *otra* métrica de atractivo en "
            "estos parámetros. El resto son elecciones que dependen de si busca paga o "
            "número de puestos, la disyuntiva que esta sección existe para mostrar y la "
            "razón por la que las dos nunca se promedian en una sola cifra.",
        )
    )

    # The highlighted set as a ranked table — the take-away artifact.
    table = highlighted.sort_values("score", ascending=False).copy()
    # The " — " between code and name is a **data** separator inside a cell, not
    # prose: the tests split on it and the tradable/robust logic never reads it,
    # so it stays identical in both languages.
    col = {
        "industry": t("Industry", "Industria"),
        "sector": t("Sector", "Sector"),
        "state": t("State", "Estado"),
        "score": t("Score", "Puntaje"),
        "feasibility": t("Feasibility", "Factibilidad"),
        "rp": t("Relative presence", "Presencia relativa"),
        "density": t("Density (related only)", "Densidad (solo relacionadas)"),
        "wage": t("Avg wage (USD, national)", "Salario prom. (USD, nacional)"),
        "premium": t("Wage premium", "Prima salarial"),
        "wage_pct": t("Wage pctile (own sector)", "Percentil salarial (su sector)"),
        "jobs": t("Jobs per firm (national)", "Puestos por empresa (nacional)"),
        "jobs_pct": t("Jobs pctile (own sector)", "Percentil de puestos (su sector)"),
        "estab_here": t("Establishments (here)", "Establecimientos (aquí)"),
        "fte_here": t("Employment (FTE, here)", "Empleo (plazas, aquí)"),
        "fte_nat": t("Employment held (FTE, national)", "Empleo sostenido (plazas, nacional)"),
        "firms": t("Employing firms", "Empresas empleadoras"),
        "robust": t("✓ Robust", "✓ Robusta"),
        "both": t("✓ Both bases", "✓ Ambas bases"),
    }
    table[col["industry"]] = (
        table["display_code"] + " — " + table["class_name"].astype(str)
    )
    table[col["state"]] = table["state"].map(state_labels()).fillna("—")
    display = table.rename(
        columns={
            "gsector_label": col["sector"],
            "score": col["score"],
            "x": col["feasibility"],
            "density": col["density"],
            "robust": col["robust"],
            "both_bases": col["both"],
            "n_estab_prov": col["estab_here"],
            "plazas_equiv_prov": col["fte_here"],
            "rca": col["rp"],
            "wage_per_fte": col["wage"],
            "wage_premium": col["premium"],
            "wage_per_fte_pctile_gsector": col["wage_pct"],
            "fte_per_firm": col["jobs"],
            "fte_per_firm_pctile_gsector": col["jobs_pct"],
            "plazas_equiv_estab_nat": col["fte_nat"],
            "n_firms": col["firms"],
        }
    )[[
        col["industry"], col["sector"], col["state"], col["score"],
        col["feasibility"], col["rp"], col["density"],
        col["wage"], col["premium"], col["wage_pct"],
        col["jobs"], col["jobs_pct"],
        # The province's own activity, immediately before the national pair, so
        # here-and-nationally read side by side — and so the support guard above is
        # auditable in the table itself rather than only in its caption.
        col["estab_here"], col["fte_here"],
        col["fte_nat"], col["firms"],
        # The two robustness disclosures last: they qualify a reading rather than
        # being one, and between Score and Feasibility they interrupted the
        # score -> feasibility -> components ordering the caption describes.
        col["robust"], col["both"],
    ]]
    # The economy-wide average wage the premium is measured against, recovered
    # from the export itself (premium = wage / national reference, identical in
    # every row) — never hardcoded, so a data refresh cannot strand the caption.
    wage_reference = float(
        (profile["wage_per_fte"] / profile["wage_premium"]).median()
    )
    with st.expander(
        t(f"The {len(table)} highlighted industries, ranked",
          f"Las {len(table)} industrias destacadas, ordenadas"),
        expanded=False,
    ):
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                col["score"]: st.column_config.NumberColumn(format="%.3f"),
                col["robust"]: st.column_config.CheckboxColumn(
                    help=t(_HELP_ROBUST_EN, _HELP_ROBUST_ES)
                ),
                col["both"]: st.column_config.CheckboxColumn(
                    help=t(_HELP_BOTH_BASES, _HELP_BOTH_BASES_ES)
                ),
                col["estab_here"]: st.column_config.NumberColumn(format="%.0f"),
                col["fte_here"]: st.column_config.NumberColumn(format="%.1f"),
                col["feasibility"]: st.column_config.NumberColumn(format="%.2f"),
                col["density"]: st.column_config.NumberColumn(format="%.3f"),
                col["rp"]: st.column_config.NumberColumn(format="%.2f"),
                col["wage"]: st.column_config.NumberColumn(format="$%.0f"),
                col["premium"]: st.column_config.NumberColumn(format="%.2f"),
                col["wage_pct"]: st.column_config.NumberColumn(format="%.0f"),
                col["jobs"]: st.column_config.NumberColumn(format="%.1f"),
                col["jobs_pct"]: st.column_config.NumberColumn(format="%.0f"),
                col["fte_nat"]: st.column_config.NumberColumn(format="%.0f"),
                col["firms"]: st.column_config.NumberColumn(format="%.0f"),
            },
        )
        st.caption(
            t(
                "**Feasibility** is the percentile mix of the two raw columns beside it — "
                "relative presence and related-only density — at the β you set; **Score** "
                "mixes that with the attractiveness percentile. Both are rank-space "
                "orderings, not measurements, and neither is exported. Wage, jobs per firm "
                "and the firm count are **national** (the registry books each firm to its "
                "headquarters province); density and relative presence are this "
                "province's. The wage premium is the industry's pay against the "
                f"economy-wide average of ${wage_reference:,.2f} per FTE. Employment held "
                "is a stock the industry carries today, not jobs an entrant would create. "
                "To see the capabilities behind any row — what the province already has "
                "that is closest to it, and what is missing — pick it in **Find an "
                "industry in the plot** above.",
                "La **factibilidad** es la mezcla de percentiles de las dos columnas crudas "
                "que están a su lado, la presencia relativa y la densidad de industrias "
                "relacionadas, con la β que usted fija; el **puntaje** mezcla eso con el "
                "percentil de atractivo. Ambos son ordenamientos en espacio de rangos, no "
                "mediciones, y ninguno se exporta. El salario, los puestos por empresa y "
                "el conteo de empresas son **nacionales** (el registro asigna cada empresa "
                "a la provincia de su sede); la densidad y la presencia relativa son de "
                "esta provincia. La prima salarial es la paga de la industria frente al "
                f"promedio de toda la economía de ${wage_reference:,.2f} por plaza "
                "equivalente. El empleo sostenido es un acervo que la industria tiene hoy, "
                "no empleos que crearía un entrante. Para ver las capacidades detrás de "
                "cualquier fila, es decir qué tiene ya la provincia que se le parezca y "
                "qué le falta, elíjala en **Buscar una industria en el gráfico** arriba.",
            )
        )

    mix = table["gsector_label"].value_counts()
    st.caption(
        "Highlighted: " + ", ".join(f"{int(n)} {sector}" for sector, n in mix.items()) + "."
    )

