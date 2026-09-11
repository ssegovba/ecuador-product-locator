"""Bilingual UI chrome: one helper, two languages, no catalog.

``t(en, es)`` picks the string for the language the current run renders in. The
translation lives next to the string it translates, which is the point: the two
languages sit on adjacent lines, so one cannot be edited without seeing the
other. That matters here because the app's captions carry numbers and
methodology claims (the parity rule, the guard floors, the row counts) — a
Spanish caption that drifted from its English twin would be a false
methodological statement shown to the audience the app was built for.

This is deliberately **not** internationalization: no locale catalog, no
``gettext``, no number or date localization, no third language. Two languages,
one bilingual author. It matches the convention the project's notebooks already
use (CLAUDE.md's ``SPANISH`` flag and its own ``t()`` helper).

**Only UI chrome translates.** Province names, CIIU class and sector
descriptions and ``gsector_label`` values are *data* and stay in their source
language (Spanish) in both modes.

Two rules for anyone adding strings here:

1. **Every new user-facing string ships through ``t()``.** A literal that
   reaches the screen untranslated is a bug, not a to-do.
2. **Long blocks stay paired module constants** (``_HELP_CUT_EN`` /
   ``_HELP_CUT_ES``) resolved through ``t()`` at the point of use. Wrapping a
   thirteen-line literal in a two-argument call makes the render site
   unreadable.

The Spanish copy carries **no em dashes**: Spanish uses *la raya* far less than
English for parenthetical asides and prefers commas, parentheses, colons, or a
second sentence. Translate by recasting, never by carrying the dash across.
"""

from __future__ import annotations

import streamlit as st

from config import DEFAULT_LANG, LANGS


def current_lang() -> str:
    """The language this run renders in.

    Falls back to :data:`config.DEFAULT_LANG` whenever the session cannot be
    read — no Streamlit runtime at all (plain pytest, a ``metrics.py`` import),
    or a session that has not reached ``main.py``'s language resolution yet. So
    ``t()`` is always safe to call, from anywhere.
    """
    try:
        value = st.session_state.get("lang")
    except Exception:  # no ScriptRunContext, or session state unavailable
        return DEFAULT_LANG
    return value if value in LANGS else DEFAULT_LANG


def t(en: str, es: str) -> str:
    """The string for the current language: Spanish by default."""
    return es if current_lang() == "es" else en


def coerce_choice(key: str, options, *label_tables: dict) -> None:
    """Force a stored option-widget value back into ``options``.

    **Call this before the widget renders.** Streamlit reconciles an option
    widget's stored value through the *formatted* option strings, and a widget
    whose ``label`` or ``format_func`` output changes with the language gets a
    new element id on a flip. Coming back the other way it can return holding the
    other language's **display label** instead of its option code — and then
    hands that label straight to the caller, which passes it on as a dataframe
    column name (``KeyError: 'Salario promedio pagado'``) or trips Streamlit's own
    "not in list" check.

    The stored value is left alone when it is already a valid option. Otherwise it
    is mapped back through any ``{code: label}`` tables supplied, so the reader's
    actual choice survives the flip; failing that the key is dropped, and the
    widget falls back to its own ``default`` / ``index``.

    Cheap and idempotent, so it is safe to call on every run for every keyed
    option widget — which is what the pages do, since a widget only has to have a
    translated label to be exposed.
    """
    try:
        if key not in st.session_state:
            return
        stored = st.session_state[key]
    except Exception:  # no session (bare pytest) — nothing to repair
        return
    try:
        if stored in options:
            return
    except TypeError:  # unhashable / uncomparable stored value
        pass
    for table in label_tables:
        for code, label in table.items():
            if label == stored and code in options:
                st.session_state[key] = code
                return
    del st.session_state[key]
