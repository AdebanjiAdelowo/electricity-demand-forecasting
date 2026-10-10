"""Numeric grounding check for copilot answers.

Every number in an answer must be traceable to a tool result. A number is
accepted when it equals some tool value after rounding to the precision the
answer used, or when it is plainly not a measurement (a horizon mentioned in
the question, a year, a date component, a list index).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w]*\d)")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?(?:[+-]\d{2}:\d{2})?")


def _numbers_in_json(obj, out: set[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, int | float):
        out.add(float(obj))
    elif isinstance(obj, str):
        for m in NUM.findall(DATE.sub(" ", obj)):
            try:
                out.add(float(m.replace(",", "")))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in_json(v, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _numbers_in_json(v, out)


@dataclass
class Verdict:
    grounded: bool
    checked: int
    unsupported: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"grounded": self.grounded, "numbers_checked": self.checked, "unsupported_numbers": self.unsupported}


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def _supported(token: str, pool: set[float]) -> bool:
    val = float(token.replace(",", ""))
    nd = _decimals(token)
    tol = 0.5 * 10 ** (-nd) + 1e-9  # the answer may round a tool value, never move it
    for p in pool:
        for cand in (p, p * 100, abs(p), abs(p) * 100):  # fractions may be quoted as percentages
            if abs(cand - val) <= tol:
                return True
    return False


def verify(answer: str, tool_results: list, question: str = "", allow: tuple[float, ...] = (50, 80, 90, 95, 100)) -> Verdict:
    """Check that the numbers in `answer` are supported by `tool_results` (or the question itself)."""
    pool: set[float] = set()
    _numbers_in_json(tool_results, pool)
    _numbers_in_json(question, pool)
    pool.update(allow)  # nominal interval levels
    text = DATE.sub(" ", answer)
    bad, n = [], 0
    for tok in NUM.findall(text):
        val = float(tok.replace(",", ""))
        if "." not in tok and "," not in tok and (abs(val) <= 12 or 1990 <= val <= 2100):
            continue  # small counts, ordinals, months, years
        n += 1
        if not _supported(tok, pool):
            bad.append(tok)
    return Verdict(grounded=not bad, checked=n, unsupported=bad)


def dumps(obj) -> str:
    return json.dumps(obj, default=str, separators=(",", ":"))
