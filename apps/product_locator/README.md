# Ecuador Diversification Explorer

A Streamlit dashboard about **where economic activity could plausibly grow in Ecuador**, read
through the 2024 business registry (REEM). It answers the question from both ends: from a
**diversification opportunity** (which provinces already have the industries it needs?) and from
a **province** (which industries make sense here?).

## Language

The app **ships in Spanish** and carries an **Español | English** toggle in the sidebar, above
the page list, so the language can be changed from any page. It is shared publicly with
Ecuadorian policymakers; English stays one click away for Growth Lab colleagues.

A `?lang=` query parameter opens the app in a given language, so a link can carry it to whoever
receives it: `…/opportunity?lang=en`. It seeds a session only — the toggle owns the language from
then on.

Only **UI chrome** translates: titles, captions, widget labels, figure titles, axes, legends,
hovers and table headers. **Data values stay in their source language (Spanish) in both modes** —
province names, CIIU class and sector descriptions. So does every downloaded CSV: the files are
the analysis artifact, not UI, and their column names and values are identical whichever language
the page is read in.

The mechanism is an inline `t(en, es)` helper in `lang.py`, not a key catalog — the translation
sits next to the string it translates, so the two languages are on adjacent lines and one cannot
be edited without seeing the other. That matters because these captions carry numbers and
methodology claims. **Every new user-facing string must ship through `t()`.** The Spanish copy
carries **no em dashes** (Spanish uses *la raya* far less than English for parentheticals); a test
enforces that on all six pages. See `notes/reference/product-locator/language_toggle.md`.

## The six pages

The app opens on **Home** (in the sidebar's *Start here* group), a landing page that explains
what the app contains and links to everything else — the five analysis pages all open in the middle of an argument, and this app
is shared publicly. It reads no data and computes nothing.

There is deliberately **no in-app methodology appendix** — the methodology is documented
outside the app, and the app is the interactive way to navigate it. An *About the metrics* page
carried an in-app duplicate until 2026-09-10; do not re-add it.

The sidebar groups the rest into three lenses — **Top-down — from an opportunity**, **Bottom-up —
from a province**, and **Both lenses in action**. Every page is ungated: each renders on a fresh
session with nothing selected. The first two lenses have the same shape: the specific view first,
then the same ranking replayed across everything as a heatmap. The third is the synthesis, listed
last because it reads as the app's conclusion only once you know what each lens contributes.

### Top-down — from an opportunity

An upstream analysis identified **50 concrete diversification opportunities** for Ecuador — 20 on
the **intensive margin** (products it already exports competitively and could deepen) and 30 on
the **extensive margin** (genuinely new products) — all at HS4 in the HS 2022 classification, and
already scored for feasibility and attractiveness. Because that set is fixed and pre-vetted, this
lens does not re-rank *products*: it ranks **provinces**, on presence alone.

1. **Opportunity & provinces** — pick one of the 50 (cards grouped and colored by margin), and the
   page maps it to the CIIU industries that make it, plus — on the extensive margin — the
   industries behind its **anchors** (products Ecuador already exports competitively that sit
   close to it in the product space). It then ranks the provinces by how much of those
   industries is **already observably there**: a choropleth, a per-province breakdown, and ranked
   bars. Controls: presence as **specialization** (location quotient, default) or **scale** (share
   of the industry nationally); counted in **employment** (default) or **establishments**; a
   support guard; a route-weight preset (direct vs anchor industries); and how many provinces to
   assign the product to.
2. **All opportunities × provinces** — the same rule replayed for all 50 opportunities under the
   same settings, in two grids. First **where each opportunity lands** (50 rows × 23 provinces,
   the cell carrying the province's rank among those that qualified), then the same allocation
   **seen through the industries** those products need — the form that compares with the province
   lens. A product is assigned to **every province whose weighted presence reaches parity** (the
   national average in the specialization view; an equal share of the industry in the scale view),
   so how many provinces it reaches is a finding, not a setting — 2 to 9, median 4. The blank
   columns are the point: two provinces no opportunity reaches at all, which is the case for the
   bottom-up lens.

**Galápagos is not assigned industries by this lens** (`config.TD_EXCLUDED_PROVINCES`, applied
once in `metrics.qualifying_provinces`). The parity rule reached ten industries for it, all of
them manufacturing, which is not a practical menu for an island economy; the bottom-up lens finds
its air and sea transport services instead. Page A greys it on the map and drops it from the
ranking, Page B drops its column rather than showing it empty (empty already means *reached by
nothing* there), and its whole candidate list arrives through the top-up. The exclusion is
**assignment only, never measurement** — Galápagos keeps every observation it contributes, so the
national denominators and the scale-view parity bar still span all 24 provinces.

**Feasibility here is observed presence and nothing else** — no relatedness density, no proximity,
no composite score.

### Bottom-up — from a province

3. **Province diversification explorer** — the mirror image, with no product involved: pick a
   province and see (a) what its registered economy is made of (composition treemap, three CIIU
   levels, employment or establishments), (b) what it does differently from a peer economy —
   **relative presence**, the province's share of an industry ÷ the peer's, against Ecuador or the
   other provinces of its region (top-10 / bottom-10 on a log scale around ×1, with a support
   guard that hides an industry only when it has both under 5 establishments and under 50 FTE here —
   the same guard the top-down lens applies), (c) which industries are worth diversifying into — a
   **feasibility × attractiveness** plane, highlighting the best-scoring industries in each presence
   state (absent / present / specialized) — and (d) **how much each could be worth**.

   **The same support guard decides what may be recommended**, not just what is charted: an
   industry the province already has is held out of the shortlist when it fails both floors, so a
   two-establishment "specialization" cannot be a headline pick. It is a toggle (default on) with a
   caption saying what it is holding out, and it changes membership only — every feasibility, score
   and raw value is identical either way. The ranked table shows the establishments and FTE the
   province holds in each row, so the call is auditable there too, plus a `✓ Both bases` column
   marking the rows whose presence state survives switching from establishments to employment.

   A **second eligibility guard** sits beside it (also a toggle, also default on): *only recommend
   well-connected industries*. A handful of industries are barely linked to anything else in the
   economy — their off-diagonal proximity mass is under 0.05 — so the related-capability half of
   their feasibility is a ratio of two very small numbers. Those are held out of the shortlist,
   counted and named in a caption, and left on the plane as context dots. Support is about
   **evidence**, connectedness about **measurability**; both change membership and nothing else.

   **Feasibility is an explicit mix you set** (`β` slider): how much of the industry the province
   already has (relative presence) against how much of what it needs the rest of the local economy
   provides (related-only relatedness density). It is also the plotted x-axis, so the plot and the
   ranking cannot disagree; the raw ingredients travel in every hover and table row. A second slider
   weighs feasibility against attractiveness (pay or jobs per firm, nationally). Both live under
   *Advanced settings* — they order a shortlist and are never shown as measurements. A `✓ Robust`
   column marks the industries that survive **both** attractiveness metrics.

   One page-level **Focus on tradable industries** toggle (default **on** — it matches the
   committed province candidate list, the candidates grid and the builder) restricts what the page may
   recommend to the 252 industries that pass two label tests: the output can plausibly be sold
   outside the province (`tradable_regional`), **and** a private entrant could actually cause the
   industry to appear there (`entry_regime == open_market`). Mining / quarrying / hunting come out as
   geology; the **state-allocated** industries (hospitals, universities, power generation) and the
   **regulated networks** (telecoms, broadcasting, banks, insurers) come out because their provincial
   presence is a ministry decision or a branch of a national charter, not something a province can
   diversify into. It is a **choice set, not an information set**: it changes which industries are
   eligible — and therefore the ranks and scores computed over them — and no raw number anywhere.
4. **Candidate industries × provinces** — the same ranking run for **all 24 provinces at once**,
   as an industry × province heatmap: every industry that reaches any province's shortlist gets a
   row, and each cell is colored by what that province has of it today (absent / present /
   specialized). Read across a row to see where an industry is a candidate, down a column for one
   province's shortlist. It answers what the explorer structurally cannot — whether an opportunity
   is broad or place-specific, and whether its **role changes by province**: about 45% of candidate
   industries are a new entry in one province and something to deepen in another. Columns are
   grouped and ruled by region; rows run from the broadest opportunities down to the
   single-province tail.

   **The page owns two controls and inherits the rest.** The attractiveness metric and a
   *min. provinces* threshold are its own; the tradable filter, both eligibility guards, how many
   industries each province contributes per presence state, and the two rank-space sliders are read
   from whatever you left set on the **explorer** — the grid is that page replayed for 24
   provinces, so it should not have a second set of knobs to keep in sync. The page does not
   restate those values back at you: the intro says where they come from, the grid caption states
   the *per state* count the ranking ran at, each guard's caption fires when it is off, and the
   closing caption names the explorer as their owner.

   The support guard matters most on this page: a two-establishment "specialization" repeated
   across a dozen provinces reads as a broad national opportunity, which is the misreading a grid
   invites — pension funding was exactly that, a 17-province row that collapses to none under the
   guard. Turn it or the tradable filter off (on the explorer) and a derived caption names the
   widest rows it was removing, measured on the grid in front of you.
### Both lenses in action

5. **The candidate industries list** — the deliverable neither lens produces on its own: **at least
   ten industries for every one of the 24 provinces**. The top-down assignment is the base — the
   industries the 50 opportunities put in a province because it already has the presence to make
   them — and where that reaches fewer than ten, the rest are **topped up** from the province's own
   bottom-up ranking. Every row says which lens found it, which is why the page is its own lens
   rather than a sixth page under one of the two: filing it inside either would claim the list
   belongs to that half.

   The first section stacks the two lenses as one bar per province, sorted by total, with the floor
   of ten marked. A short, all-red bar is a province the product analysis never reached — which is
   the case for the bottom-up lens, made visible in one glance. Two provinces get nothing at all
   from the top-down side.

   Picking a province prices its list against **Ecuador's own frontier**: for each industry, the
   largest share of its own economy any province posts in it, applied to this province's total
   employment — its **potential**, or the **gap** still to travel, as a treemap. Every benchmark is
   activity an Ecuadorian province already has, so it is an existence proof rather than a forecast
   — and a **stock**, never jobs an entrant would create. Industries with no province holding at
   least 50 FTE anywhere cannot be priced at all: below that floor a "frontier" is a handful of
   workers.

   **Each view leaves some rows out, and the caption says which.** A treemap has no zero-area tile:
   an industry nothing in Ecuador can benchmark gets none, and one already at the frontier has a
   gap of zero and drops out of the gap view. The count and — where few enough — the names are
   derived from what is on screen, never hardcoded, and the table below always carries the full
   list whether or not a row got a tile.

   Tiles are laid out by **broad sector** and can be coloured by sector (the default, matching the
   layout and the explorer's composition treemap) or by **presence state**. Switching the colouring
   **moves nothing** — same hierarchy, same values, only the fills change.

   Below the treemap the list itself opens as a table, in rank order with a download — the only
   place the **HS4 codes** behind each top-down row are readable, and worth opening for the 17
   provinces whose list is exactly ten rows. Beside those codes sits their **accessible market**
   in billions of current USD, summed over the opportunities the row names: a *world* market size
   from the upstream analysis, not Ecuador's exports and not a projection of what the province
   would capture — the column tooltip says so, and it is blank on top-ups. A second button beside it, outside the collapsed
   table, downloads **all 24 provinces at once**.

   The page has **one control: the attractiveness metric** — and unlike almost every other control
   in the app it changes **who is on the list**, not just the order: only 39% of the top-ups are
   shared between the two settings. The page says so beside the control. Ten is fixed and is not a
   slider: it is enough industries to look for *structure* in a list while staying small enough to
   investigate seriously, which is a judgement about how a list gets used rather than a statistical
   property.

   **The page derives the list live and reads no CSVs**, so the app deploys with no pipeline run.
   The same list is a committed analysis artifact under
   `data/processed/summary_tables/province_candidates/`; a test asserts the two are equal rather
   than having the app load them.

## Run locally

From this folder:

```bash
pip install -r ../../requirements.txt   # or use the project .venv
streamlit run main.py
```

The app reads only precomputed CSV / JSON / GeoJSON files (no microdata crunching, no
geopandas, no parquet), so it starts in a couple of seconds.

## Data it consumes

All inputs are **precomputed** and read from `data/` (paths anchored in `config.py` via
`PROJECT_ROOT`). The two tables behind the top-down lens are:

- `summary_tables/oportunidades/oportunidades_industrias.csv` — the 50 opportunities mapped to
  CIIU classes, by route (`producto` = industries that make the product, `ancla` = industries
  behind its anchors), with the HS→ISIC allocation weights. Built by
  `utils/build_oportunidades_industrias.py` on top of `utils/build_hs_isic_crosswalk.py`.
- `summary_tables/industry_space/province_class_rca_multibase.csv` — province × CIIU class
  presence on three bases (location quotients, national shares, raw employment and
  establishments). 24 provinces, no national row. **Also the bottom-up potential/gap benchmark's
  input**: it is the only *dense* province × class table, so a maximum taken over all provinces is
  well defined.

The bottom-up pages read the `relatedness_density` and `industry_labor_profile` exports, plus
`data/intermediate/classifications/isic_tradability_ai.csv` for the tradable menu (derived at load
time from `tradable_regional == 'tradable'` **and** `entry_regime == 'open_market'` minus
`config.EXTRACTIVE_CLASSES` — never hardcoded, so the count follows the labels; the two
second-opinion review documents sit beside the CSV). `relatedness_density/province_density_plazas.csv` is read for one purpose only —
the explorer's `✓ Both bases` column; every presence state on the page is establishment-based.

**Nothing here is written by the app, and no v3 change touched a dataset**: the support guard is a
UI filter over a `config.py` constant and the benchmark is a maximum of shipped shares times a
shipped total, so `data/DATA_STRUCTURE.md` and every `data_dictionary.md` are unchanged.

Shared inputs live in `data/processed/summary_tables/product_locator/`: `provinces.geojson`,
simplified province geometry + attributes (it replaces the GADM shapefile, so there is no
geopandas at runtime).

**The runtime manifest is 12 files** — everything any page actually reads, all under `data/`:

| File | Read for |
|---|---|
| `processed/summary_tables/product_locator/provinces.geojson` | choropleths, province metadata |
| `intermediate/classifications/ciiu_rev4dot1.csv` | CIIU section / division / class labels |
| `intermediate/classifications/isic_tradability_ai.csv` | tradable menu, `entry_regime` |
| `processed/isic/isic_rev4_detailed.json` | ISIC class descriptions |
| `processed/summary_tables/oportunidades/oportunidades_industrias.csv` | the 50 opportunities |
| `processed/summary_tables/industry_space/province_class_rca_multibase.csv` | presence, frontier benchmark |
| `processed/summary_tables/relatedness_density/industry_proximity_matrix_multi_industry.csv` | proximity |
| `processed/summary_tables/relatedness_density/province_density_estab_count.csv` | density (primary base) |
| `processed/summary_tables/relatedness_density/province_density_plazas.csv` | density (`✓ Both bases` only) |
| `processed/summary_tables/relatedness_density/industry_density_support.csv` | connectedness guard |
| `processed/summary_tables/industry_labor_profile/class_labor_profile.csv` | attractiveness, sector labels |
| `processed/summary_tables/industry_labor_profile/industry_labor_composition.csv` | province × class employment |

`config.py` still declares paths to `reem_empresas_slim.parquet`,
`reem_establecimientos_slim.parquet` and the two `industry_tradability/` tables, and
`data_loader.py` still defines their loaders — but **no view calls them**. They are leftovers
from the app's first design, not inputs: the app runs without those files, and the standalone
deploy repo omits them on purpose.

See `data/DATA_STRUCTURE.md` and the per-folder `data_dictionary.md` files for schemas. To refresh
the app's data, re-run the owning builder or notebook in the research repo.

## Layout

```
main.py          entry point (language toggle, st.navigation over the 4 page groups)
nav.py           page registry, rebuilt per run so titles follow the language
lang.py          t(en, es) + current_lang()   state.py  session state (shared page params)
config.py        paths, colors, constants     data_loader.py  cached data loaders
metrics.py       presence / ranking math      viz.py    plotly figures
views/           one module per page: home.py, opportunity.py, assignment.py, bottom_up.py,
                 candidates.py, candidate_list.py
tests/           pytest suite (pytest.ini): pure-logic tests plus `-m tripwire` pins
                 against the 2024 exports and `-m apptest` page smoke runs
```

Run the tests from the repo root:

```bash
python -m pytest apps/product_locator/tests -q          # everything, 138 tests (~35 s)
python -m pytest apps/product_locator/tests -q -m "not tripwire"   # logic only
```

The suite drives the app in **English**, pinned in `tests/conftest.py` and
`tests/_page_runner.py`: many assertions use UI copy as lookup keys (widget labels, dataframe
column headers), and re-pinning them all to Spanish would churn the file for no extra
verification. The language behaviour lives in three dedicated tests instead — the shipped
default, the toggle, and the no-em-dash check on the Spanish copy.

The `tripwire` pins are **expected to break when the upstream notebooks are re-run** — that is what
they are for. Re-pin them deliberately (and update the as-built) rather than deleting them.

Design decisions and their evidence live in the research repo's
`notes/reference/product-locator/` as-builts.
