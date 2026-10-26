"""Unit requirements of formulas: what an expression evaluates to, and what a process expects.

Used by Units ▸ Formula Unit Inspector. Pure functions (no GUI), so they are tested directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..units.dimensions import Dimension
from ..units.unit_expressions import si_unit_text
from ..units.unit_registry import UnitRegistry
from .dimensional_analysis import (NUMERIC, OUTPUT_TARGETS, TARGETS_BY_KIND, CheckEnv, Checker, FormulaTypeError,
                                   FType, formula_dim)
from .formula_parser import FormulaSyntaxError, Node, children, parse, span


@dataclass
class TermRow:
    text: str
    type: str
    dimension: str          # "[L·T^-1]" or "" for non-numeric values
    si_unit: str            # "m/s" or ""
    quantities: str         # matching quantity names
    depth: int


@dataclass
class ExpressionReport:
    ok: bool
    message: str = ""
    error_pos: int = -1
    result: TermRow | None = None
    rows: list[TermRow] = field(default_factory=list)


def quantity_names(units: UnitRegistry, dim: Dimension) -> str:
    """Quantities with this dimension in formulas, where angles are dimensionless (rad/s ≡ 1/s)."""
    return ", ".join(q.label for q in units.quantities() if formula_dim(q.dimension) == dim)


def _row(units: UnitRegistry, text: str, t: FType, depth: int) -> TermRow:
    if t.kind in NUMERIC:
        if t.poly:
            return TermRow(text, t.describe(), "any (literal 0)", "", "", depth)
        dim: Dimension = t.dim
        names = quantity_names(units, dim)
        return TermRow(text, t.describe().split(" [")[0], f"[{dim}]", si_unit_text(dim), names, depth)
    return TermRow(text, t.describe(), "", "", "", depth)


def inspect_expression(source: str, env: CheckEnv, units: UnitRegistry) -> ExpressionReport:
    """Type and dimension of an expression and of each of its sub-expressions (angles are dimensionless)."""
    try:
        node = parse(source)
        checker = Checker(CheckEnv(env.mode, env.kind, env.params, env.globals_, env.custom, dict(env.definitions),
                                   source))
        result = checker.check(node)
    except (FormulaSyntaxError, FormulaTypeError) as exc:
        return ExpressionReport(False, exc.message, exc.pos)
    rows: list[TermRow] = []

    def walk(n: Node, depth: int) -> None:
        t = checker.types.get(id(n))
        if t is not None:
            a, b = span(n)
            rows.append(_row(units, source[a:b], t, depth))
        for c in children(n):
            walk(c, depth + 1)
    walk(node, 0)
    return ExpressionReport(True, result=_row(units, source.strip(), result, 0), rows=rows)


@dataclass
class Requirement:
    kind: str       # parameter | global | custom property | output
    name: str
    type: str
    quantity: str
    si_unit: str
    display_unit: str


def env_for(pdef) -> CheckEnv:
    """Typing environment for expressions written in the context of a process (formula or built-in)."""
    from .formula_process import check_env, param_types
    if pdef.formula_spec is not None:
        return check_env(pdef.formula_spec)
    return CheckEnv(mode=pdef.mode, kind=pdef.kind if pdef.kind in TARGETS_BY_KIND else "contribution",
                    params=param_types(pdef.parameters))


def process_requirements(pdef, units: UnitRegistry) -> list[Requirement]:
    """Inputs and outputs of a process with their quantities and units."""
    out: list[Requirement] = []

    def unit_cols(quantity: str | None) -> tuple[str, str, str]:
        if not quantity:
            return "dimensionless", "1", "1"
        q = units.quantity(quantity)
        return q.label, si_unit_text(q.dimension), units.preferred_unit(quantity)

    for d in pdef.parameters:
        out.append(Requirement("parameter", f"param.{d.key}", d.vtype.value, *unit_cols(d.quantity)))
    spec = pdef.formula_spec
    if spec is not None:
        for g in spec.globals:
            out.append(Requirement("global", f"global.{g['name']}", g.get("type", "float"), *unit_cols(g.get("quantity"))))
        for c in spec.custom:
            out.append(Requirement("custom property", f"custom.{c['key']}", c.get("type", "float"),
                                   *unit_cols(c.get("quantity"))))
        targets = sorted(TARGETS_BY_KIND.get(spec.kind, ()) - {"custom"})
        for name in targets:
            t = OUTPUT_TARGETS[name]
            dim_text = si_unit_text(t.dim) if t.kind in NUMERIC else "—"
            labels = quantity_names(units, t.dim) if t.kind in NUMERIC else ""
            out.append(Requirement("output", name, t.describe().split(" [")[0], labels or t.describe(), dim_text,
                                   ""))
    return out
