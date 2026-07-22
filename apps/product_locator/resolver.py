"""Resolve a user-selected HS6 product code to candidate ISIC Rev.4 classes.

The user picks an HS edition and a 6-digit code from that edition's
reference list; matching is a direct lookup in the edition's multiple-match
HS->ISIC4 crosswalk (e.g. HS22_ISIC4_all.csv).
"""

from __future__ import annotations

import pandas as pd


def resolve_isic_matches(
    hs_code: str,
    crosswalk: pd.DataFrame,
    isic_descriptions: pd.DataFrame,
    class_name_lookup: dict[str, str],
    reem_classes: set[str],
) -> pd.DataFrame:
    """Return one row per matched ISIC class for ``hs_code``.

    Columns: codigo_clase, isic_name (Spanish, CIIU), isic_name_en /
    description (English, ISIC detailed), match_weight, in_reem.
    """
    code = str(hs_code).strip().zfill(6)
    matched = crosswalk.loc[crosswalk["hs_code"] == code].copy()
    if matched.empty:
        return pd.DataFrame(
            columns=["codigo_clase", "isic_name", "isic_name_en", "description", "match_weight", "in_reem"]
        )

    matched = matched.rename(columns={"isic4_code": "codigo_clase"})
    matched = matched.drop_duplicates(subset=["codigo_clase"]).sort_values("codigo_clase")

    desc_lookup = isic_descriptions.set_index("isic_code")
    matched["isic_name"] = matched["codigo_clase"].map(class_name_lookup)
    matched["isic_name_en"] = matched["codigo_clase"].map(desc_lookup["isic_name"])
    matched["description"] = matched["codigo_clase"].map(desc_lookup["description"])
    matched["isic_name"] = matched["isic_name"].fillna(matched["isic_name_en"]).fillna(matched["codigo_clase"])
    matched["description"] = matched["description"].fillna("No description available.")
    matched["in_reem"] = matched["codigo_clase"].isin(reem_classes)

    return matched[
        ["codigo_clase", "isic_name", "isic_name_en", "description", "match_weight", "in_reem"]
    ].reset_index(drop=True)
