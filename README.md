# Ecuador Product Location Explorer

An interactive Streamlit dashboard that helps identify **where in Ecuador a given product could
plausibly be produced**, based on the industries that make it and the related industrial
capabilities already present across the country's provinces.

> **Live app:** https://ecu-hs-locator.streamlit.app/

It walks through four steps: an HS product → the industries (ISIC/CIIU classes) that produce it →
where those industries are active across provinces (2024 business registry) → a full province
profile → and an economic-complexity view of related industries and diversification opportunities.
See [`apps/product_locator/README.md`](apps/product_locator/README.md) for a page-by-page
description.

## Run locally

Requires Python 3.13.

```bash
pip install -r requirements.txt
streamlit run apps/product_locator/main.py
```

The app reads only precomputed CSV / Parquet / GeoJSON files, so it starts in a couple of seconds
(no heavy computation, no geospatial libraries).

## Deploy on Streamlit Community Cloud

1. Sign in at **https://share.streamlit.io** with GitHub.
2. **Create app** → deploy from GitHub, repository = this repo, branch = `main`, main file path =
   `apps/product_locator/main.py`.
3. (Optional) set Python version to **3.13** in the advanced settings.
4. Deploy. The first build installs `requirements.txt` (~2–5 min); afterwards you get a public
   `*.streamlit.app` URL to share. Push to `main` to auto-redeploy.

## What's in this repo

This is a **self-contained deployment mirror** of the `product_locator` app developed in a separate
(private) research repository. It contains only the app code and the **precomputed data it reads**:

- `apps/product_locator/` — the Streamlit app.
- `data/` — the exact input files the app loads (slim REEM tables, province GeoJSON, HS/ISIC
  crosswalks, provincial summaries, and the relatedness/tradability exports). See
  `data/processed/summary_tables/product_locator/data_dictionary.md`.
- `requirements.txt` — pinned runtime dependencies.

## Data provenance

All datasets are **outputs** of the upstream research pipeline (INEC business registry REEM 2024,
national accounts, census, ENEMDU labor survey; HS↔ISIC concordances; and the relatedness-density
economic-complexity analysis). They are generated and documented in the research repo — this repo
is a read-only deployment copy, not the canonical source. To refresh the data, regenerate it
upstream and re-sync the files here. Monetary values are current (nominal) USD; province codes are
zero-padded 2-digit strings.
