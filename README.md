# Ecuador Diversification Explorer

*Explorador de Diversificación del Ecuador*

An interactive Streamlit dashboard about **where economic activity could plausibly grow in
Ecuador**, read through the 2024 business registry (REEM) and a set of 50 identified export
diversification opportunities.

> **Live app:** https://ecu-hs-locator.streamlit.app/

The app answers one question — *which industries could a province plausibly grow into?* — through
three lenses. **Top-down, from an opportunity**: it takes the 50 diversification opportunities
identified for Ecuador (20 on the intensive margin, 30 on the extensive margin, all at HS4 in the
HS 2022 classification), maps each to the CIIU industries that make it, and ranks the provinces on
how much of those industries is already observably there. **Bottom-up, from a province**: it starts
from a province's registered economy — what it is made of, what it does differently from Ecuador
or its regional peers, and which industries score best on a feasibility × attractiveness plane.
**Both lenses in action**: the two are combined into a single candidate industries list of at least
ten industries for every one of the 24 provinces, each row labelled with the lens that found it.

Six pages in all — a Home landing page that explains the app and links to everything, plus five
analysis pages. See [`apps/product_locator/README.md`](apps/product_locator/README.md) for the
page-by-page description and the methodology behind each control.

## Language

The app **ships in Spanish** and carries an **Español | English** toggle in the sidebar, so the
language can be changed from any page. A `?lang=` query parameter opens it in a given language, so
a shared link carries its own language: `…/opportunity?lang=en`.

Only UI chrome translates. Province names, CIIU descriptions and every downloaded CSV stay in
their source language (Spanish) in both modes — the downloads are the analysis artifact, not UI.

## Run locally

Requires Python 3.13.

```bash
pip install -r requirements.txt
cd apps/product_locator
streamlit run main.py
```

The app reads only precomputed CSV / JSON / GeoJSON files (no microdata crunching, no parquet, no
geospatial libraries), so it starts in a couple of seconds.

## Deploy on Streamlit Community Cloud

1. Sign in at **https://share.streamlit.io** with GitHub.
2. **Create app** → deploy from GitHub, repository = this repo, branch = `main`, main file path =
   `apps/product_locator/main.py`.
3. Set the Python version to **3.13** in the advanced settings.
4. Deploy. The first build installs `requirements.txt`; afterwards you get a public
   `*.streamlit.app` URL to share. Push to `main` to auto-redeploy.

## What's in this repo

- `apps/product_locator/` — the Streamlit app (entry point `main.py`).
- `data/` — the twelve precomputed tables the app reads, at the same repo-relative paths the app
  expects. Nothing here is written by the app.
- `requirements.txt` — pinned runtime dependencies.

The layout mirrors the research repo on purpose: every app module is byte-identical to its
upstream copy, and `config.py` resolves `data/` by walking two levels up from
`apps/product_locator/`. Moving the code to the repo root would break that, so don't.

## Data provenance

Every file under `data/` is an **aggregated output** of a larger private analysis repository,
built from official Ecuadorian sources — INEC's business registry (REEM 2024) and its
classifications (CIIU Rev. 4.1 / ISIC Rev. 4), plus an HS-level export diversification analysis.
No microdata, no firm-level records: the tables are province × industry aggregates and industry
reference tables.

**This repo is a deploy snapshot, not the analysis home.** The data is refreshed by re-running the
owning notebooks upstream and re-copying the files here by hand. Monetary values are current
(nominal) USD; province codes are zero-padded 2-digit strings; industry codes are CIIU Rev. 4
classes.
