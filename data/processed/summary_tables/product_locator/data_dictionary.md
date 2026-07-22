# Product Locator app bundle — data dictionary

The **deploy-committed inputs** the `apps/product_locator` dashboard (Ecuador Product Location
Explorer) consumes at runtime. The app is deployed from the GitHub repo (Streamlit Community
Cloud), so the heavy/gitignored sources (full REEM microdata parquets, the raw GADM shapefile) are
not available there — instead this folder holds lightweight, app-column-only versions plus the
compact trade tables. Keeping them here means the deployed app needs **no gitignored microdata, no
`data/raw/`, and no geopandas** at runtime.

Two builders write here:
- `utils/build_product_locator_app_data.py` → the slim REEM tables + `provinces.geojson`.
- `utils/build_product_locator_trade.py` → the `ec_trade_*.csv` export tables (below).

## App data bundle (slim REEM + geometry)

- `reem_empresas_slim.parquet` — enterprise table, only the 13 columns `data_loader.load_reem_data`
  reads (`id_empresa, codigo_provincia, provincia, codigo_seccion, codigo_division, codigo_clase,
  empleo_equiv, plazas_equiv, remuneraciones, ventas_totales, exportaciones, importaciones,
  exportaciones_netas`). ~1.07M rows, zstd (~11 MB). Slimmed from
  `data/processed/reem/empresas_2024_processed.parquet` (30 cols, gitignored).
- `reem_establecimientos_slim.parquet` — establishment table, only the 7 columns
  `load_estab_data` reads (`id_empresa, id_unidad_local, codigo_provincia, provincia, codigo_clase,
  plazas_equiv, is_active`). ~1.34M rows, zstd (~15 MB). Slimmed from the 17-col establishments
  parquet (gitignored).
- `provinces.geojson` — 24 province polygons, simplified (`tolerance 0.005`, EPSG:4326), with
  `id_code, id_name, id_abbr, region, province_label` in each feature's `properties`. Feeds both
  the choropleths (`featureidkey = "properties.id_code"`) and the app's province attribute lookup,
  replacing the runtime shapefile + geopandas. ~0.26 MB.

The slim parquets carry the same cleaning-on-read contract as the originals (the loaders zero-pad
codes and coerce dtypes), so behavior is unchanged. Rebuild after refreshing the processed REEM
data or the province geometry: `python utils/build_product_locator_app_data.py`.

## Trade tables (Page-1 export scaling)

Produced by `utils/build_product_locator_trade.py`, which reads the Atlas country-product-year files
under `data/raw/hs/` (`hs22/hs12/hs92_country_product_year_6.csv`), filters to Ecuador
(`country_iso3_code == 'ECU'`) and the latest available year (2024), zero-pads the HS6 code, and
aggregates export/import value by code. One file per HS edition so the scaling uses the same
classification the user selects on Page 1.

### Files

- `ec_trade_2022.csv` — HS 2022 (H6) classification.
- `ec_trade_2012.csv` — HS 2012 (H4) classification.
- `ec_trade_1992.csv` — HS 1992 (H0) classification.

Each ~4.5k–4.8k rows (one per HS6 code Ecuador reported), 2024, Ecuador only.

### Columns

| Column | Type | Description |
|---|---|---|
| `hs_code` | string | 6-digit zero-padded HS code, in the file's own HS edition. |
| `export_value` | float | Ecuador's exports of the product (USD, nominal). |
| `import_value` | float | Ecuador's imports of the product (USD, nominal). |
| `national_export_share` | float | `export_value` / Ecuador's total exports that year (fraction). |
| `year` | int | Reference year (latest available; 2024). |

### Notes

- Read `hs_code` as a string (leading zeros matter) and join to the matching HS edition's
  `HS{xx}_ISIC4_all.csv` crosswalk to reach industries.
- An "industry export footprint" is computed in the app by summing `export_value` over the HS
  codes that map to an ISIC class — **unweighted**, so a product mapping to several industries is
  counted in each; the UI carries a double-counting disclaimer.
- Regenerate with `python utils/build_product_locator_trade.py` (from the repo root) when the
  Atlas source files are refreshed; the script always takes the latest year present.
