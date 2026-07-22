# Ecuador Product Location Explorer

A Streamlit dashboard that answers: **"where in Ecuador could a given product plausibly be
produced?"** — starting from an HS product code, mapping it to the industries that make it, and
showing where those industries (and their *related* industries) are already present.

## The four pages

1. **Product & industries** — pick an HS classification edition and a 6-digit product; the app
   maps it to candidate ISIC/CIIU industries (weighted HS→ISIC concordance) and lets you curate
   the list. Shows each industry's Ecuador export scale for context.
2. **Geographic analysis** — province choropleths and KPIs for the selected industries from the
   2024 business registry (REEM): enterprises, establishments, employment, net exporters, and
   their national shares. Enterprise vs. establishment views expose the *headquarter effect*.
3. **Province profile** — demographics, labor market and economic structure of one province, plus
   a **headquarters-reach** analysis (where multi-unit firms' establishments and employment sit)
   and "what else do these firms do".
4. **Related industries & opportunities** — economic-complexity view built on the
   `relatedness_density` analysis: industry **proximity**, per-province **relatedness density**,
   and intensive/extensive **opportunity** rankings. A self/no-self density toggle switches between
   "where can it be made today" and "where do the *related* capabilities exist even if the industry
   is absent". See `notes/relatedness_density_math_explainer.md` in the research repo for the math.

## Run locally

From this folder:

```bash
pip install -r ../../requirements.txt   # or use the project .venv
streamlit run main.py
```

The app reads only precomputed CSV / Parquet / GeoJSON files (no microdata crunching, no
geopandas), so it starts in a couple of seconds.

## Data it consumes

All inputs are **precomputed** and read from `data/` (paths anchored in `config.py` via
`PROJECT_ROOT`). The lightweight, app-specific inputs live in
`data/processed/summary_tables/product_locator/`:

- `reem_empresas_slim.parquet`, `reem_establecimientos_slim.parquet` — the REEM columns the app
  uses, sliced from the full processed tables.
- `provinces.geojson` — simplified province geometry + attributes (replaces the GADM shapefile;
  no geopandas at runtime).
- `ec_trade_{2022,2012,1992}.csv` — Ecuador export scale per HS edition.

Plus shared reference/summary tables (CIIU + HS↔ISIC crosswalks, ISIC descriptions, provincial
multisource / GVA / census / labor summaries, and the `relatedness_density` + `industry_tradability`
exports). See `data/processed/summary_tables/product_locator/data_dictionary.md`.

These bundles are generated in the **`ecuador-project`** research repo by
`utils/build_product_locator_app_data.py` (slim REEM + geometry) and
`utils/build_product_locator_trade.py` (trade); the relatedness/summary tables come from the
project notebooks. To refresh the app's data, regenerate there and re-copy the bundle.

## Layout

```
main.py          entry point (st.navigation over the 4 pages)
nav.py           page registry            state.py     session state + guards
config.py        paths, colors, constants resolver.py  HS→ISIC match resolution
data_loader.py   cached data loaders      metrics.py   aggregation logic
viz.py           plotly figures           views/       one module per page
```
