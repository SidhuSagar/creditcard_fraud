from __future__ import annotations

import re
from typing import Any

TIME_ALIASES = (
    "time", "timestamp", "timestamps", "unixtime", "unix_timestamp",
    "transactiontime", "transactiontimestamp", "timeseconds", "seconds",
    "elapsedtime", "elapsed_seconds",
)
AMOUNT_ALIASES = (
    "amount", "amt", "amountusd", "amount_usd", "transactionamount",
    "transaction_amount", "transactionamountusd", "transactiondollaramount",
    "transaction_dollar_amount", "dollaramount", "value", "transactionvalue",
    "money", "price",
)
LABEL_ALIASES = (
    "class", "label", "target", "y", "outcome", "result", "isfraud",
    "is_fraud", "fraud", "fraudulent", "isfraudulent", "classlabel",
)
V1_28 = [f"V{i}" for i in range(1, 29)]

def normalise(name: Any) -> str:
    text = str(name).lstrip("\ufeff").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


_ALIAS_LOOKUP = {}
def _add(canonical, names):
    for n in [canonical, *names]:
        k = normalise(n)
        if k:
            _ALIAS_LOOKUP.setdefault(k, canonical)
_add("Time", TIME_ALIASES)
_add("Amount", AMOUNT_ALIASES)
for v in V1_28:
    _add(v, [v.lower(), f"v{int(v[1:]):02d}"])

_LABEL_KEYS = {normalise(n) for n in LABEL_ALIASES}


def resolve_columns(columns: Any):
    cols = list(columns)
    rename: dict[str, str] = {}
    used: set[str] = set()
    for canonical in ["Time", *V1_28, "Amount"]:
        ck = normalise(canonical)
        match = next((c for c in cols if c not in used and normalise(c) == ck), None)
        if match is None:
            match = next((c for c in cols if c not in used and _ALIAS_LOOKUP.get(normalise(c)) == canonical), None)
        if match is not None:
            used.add(match)
            rename[match] = canonical
    missing = [c for c in ["Time", *V1_28, "Amount"] if c not in rename.values()]
    label = next((c for c in cols if c not in used and normalise(c) in _LABEL_KEYS), None)
    if label is not None:
        rename[label] = "Class"
    return rename, missing


def find_amount_col(columns: Any):
    cols = list(columns)
    for c in cols:
        if _ALIAS_LOOKUP.get(normalise(c)) == "Amount":
            return c
    return None


def find_datetime_col(columns: Any):
    from . import generic_detector  # avoid circular; generic_detector reuses _find logic
    return generic_detector._find_datetime_col(columns)


def found_columns(columns: Any, limit: int = 40) -> str:
    names = [str(c) for c in columns]
    head = ", ".join(names[:limit])
    if len(names) > limit:
        head += f", … (+{len(names) - limit} more)"
    return head or "(none)"
