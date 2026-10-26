"""Unit expressions such as 'kg·m²/s²', 'N*m', 'km/h' or 'm·s^-1', and SI unit names of dimensions.

Grammar::

    expr    := factor (('*' | '·' | '/') factor)*
    factor  := ( SYMBOL | '1' | '(' expr ')' ) ( '^' INT )?
    SYMBOL  := a registered unit symbol, optionally followed by superscript digits (s², m⁻¹)
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .dimensions import Dimension

if TYPE_CHECKING:  # pragma: no cover
    from .unit_registry import UnitRegistry

_TOKEN = re.compile(r"\s*(?:(?P<sym>[^\s*/^·()]+)|(?P<op>[*/^·()]))")
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_SUP_RE = re.compile(r"^(?P<base>.+?)(?P<exp>⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)$")
_TO_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
# SI base units in the order of dimensions.BASE_SYMBOLS (L, M, T, I, Θ, N, J, A)
SI_BASE = ("m", "kg", "s", "A", "K", "mol", "cd", "rad")


class UnitExpressionError(ValueError):
    pass


def parse_unit_expression(text: str, registry: "UnitRegistry") -> tuple[float, Dimension]:
    """Parse a unit expression into (SI factor, dimension). Offset units (e.g. °C) are rejected."""
    text = text.strip()
    if not text:
        raise UnitExpressionError("Empty unit")
    if registry.has_unit(text):
        return _plain(text, registry)
    toks = [(m.group("sym"), m.group("op")) for m in _TOKEN.finditer(text) if m.group().strip()]
    if "".join((s or o) for s, o in toks) != "".join(text.split()):
        raise UnitExpressionError(f"Invalid characters in unit '{text}'")
    pos = 0

    def symbol(sym: str) -> tuple[float, Dimension]:
        if registry.has_unit(sym):
            return _plain(sym, registry)
        m = _SUP_RE.match(sym)
        if m and registry.has_unit(m.group("base")):
            f, d = _plain(m.group("base"), registry)
            e = int(m.group("exp").translate(_SUPERSCRIPTS))
            return f ** e, d ** e
        raise UnitExpressionError(f"Unknown unit '{sym}' in '{text}'")

    def factor() -> tuple[float, Dimension]:
        nonlocal pos
        if pos >= len(toks):
            raise UnitExpressionError(f"Incomplete unit expression '{text}'")
        sym, op = toks[pos]
        pos += 1
        if op == "(":
            f, d = product()
            if pos >= len(toks) or toks[pos][1] != ")":
                raise UnitExpressionError(f"Missing ')' in unit '{text}'")
            pos += 1
        elif sym == "1":
            f, d = 1.0, Dimension()
        elif sym is not None:
            f, d = symbol(sym)
        else:
            raise UnitExpressionError(f"Unexpected '{op}' in unit '{text}'")
        if pos < len(toks) and toks[pos][1] == "^":
            pos += 1
            if pos >= len(toks) or toks[pos][0] is None:
                raise UnitExpressionError(f"Missing exponent in unit '{text}'")
            try:
                e = int(toks[pos][0].translate(_SUPERSCRIPTS))
            except ValueError:
                raise UnitExpressionError(f"Unit exponents must be integers ('{text}')") from None
            pos += 1
            f, d = f ** e, d ** e
        return f, d

    def product() -> tuple[float, Dimension]:
        nonlocal pos
        f, d = factor()
        while pos < len(toks) and toks[pos][1] in ("*", "·", "/"):
            op = toks[pos][1]
            pos += 1
            f2, d2 = factor()
            f, d = (f * f2, d * d2) if op != "/" else (f / f2, d / d2)
        return f, d

    f, d = product()
    if pos != len(toks):
        raise UnitExpressionError(f"Unexpected text in unit '{text}'")
    return f, d


def _plain(symbol: str, registry: "UnitRegistry") -> tuple[float, Dimension]:
    u = registry.unit(symbol)
    if u.offset:
        raise UnitExpressionError(f"Unit '{symbol}' has an offset and cannot be combined with other units")
    return u.factor, u.dimension


def si_unit_text(dim: Dimension) -> str:
    """SI base-unit expression of a dimension, e.g. kg·m²/s² (angles shown as rad)."""
    num, den = [], []
    for k in (1, 0, 2, 3, 4, 5, 6, 7):  # conventional order: kg, m, s, A, K, mol, cd, rad
        sym, e = SI_BASE[k], dim.exponents[k]
        if e > 0:
            num.append(sym + (str(e).translate(_TO_SUP) if e != 1 else ""))
        elif e < 0:
            den.append(sym + (str(-e).translate(_TO_SUP) if e != -1 else ""))
    top = "·".join(num) or "1"
    if not den:
        return top
    return f"{top}/{den[0]}" if len(den) == 1 else f"{top}/({'·'.join(den)})"
