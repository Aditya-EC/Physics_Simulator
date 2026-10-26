"""Safe formula language: lexer, Pratt parser and abstract syntax tree.

Formulas are parsed into an AST and later compiled to Python closures
(formula_runtime.py). Python's ``eval``/``exec`` are never used, and the
language has no access to Python objects: only the names, attributes and
functions listed in dimensional_analysis.py exist.

Grammar (lowest to highest precedence)::

    expr        := ternary
    ternary     := or ( '?' expr ':' expr )?
    or          := and ( ('||' | 'or') and )*
    and         := not ( ('&&' | 'and') not )*
    not         := ('!' | 'not') not | comparison
    comparison  := additive ( ('==' | '!=' | '<' | '<=' | '>' | '>=') additive )?
    additive    := term ( ('+' | '-') term )*
    term        := unary ( ('*' | '/' | '%') unary )*
    unary       := ('-' | '+') unary | power
    power       := postfix ( ('^' | '**') unary )?          (right associative)
    postfix     := primary ( '.' NAME | '[' expr ']' | '(' args ')' )*
    primary     := NUMBER ('[' unit ']')? | STRING | NAME | '(' expr ')'

A unit annotation directly follows a number: ``9.81[m/s^2]``, ``30[deg]``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..units.dimensions import Dimension
from ..units.unit_expressions import UnitExpressionError, parse_unit_expression
from ..units.unit_registry import UnitError, UnitRegistry, default_registry


class FormulaSyntaxError(ValueError):
    def __init__(self, message: str, pos: int = -1, source: str = "") -> None:
        self.message, self.pos, self.source = message, pos, source
        super().__init__(self.describe())

    def describe(self) -> str:
        if self.pos < 0 or not self.source:
            return self.message
        line = self.source
        return f"{self.message} at column {self.pos + 1}:\n  {line}\n  {' ' * self.pos}^"


# ---- AST ----------------------------------------------------------------------------------------

@dataclass
class Node:
    pos: int = field(default=0, kw_only=True)
    end: int = field(default=-1, kw_only=True, compare=False)   # exclusive end in the source (leaves, calls…)
    start: int = field(default=-1, kw_only=True, compare=False)  # opening '(' when the node was parenthesised


@dataclass
class Num(Node):
    value: float
    dim: Dimension = field(default_factory=Dimension)
    unit_text: str = ""


@dataclass
class Str(Node):
    value: str


@dataclass
class Name(Node):
    id: str


@dataclass
class Attr(Node):
    obj: Node
    name: str


@dataclass
class Index(Node):
    obj: Node
    index: Node


@dataclass
class Call(Node):
    func: str
    args: list[Node]


@dataclass
class Unary(Node):
    op: str
    operand: Node


@dataclass
class Binary(Node):
    op: str
    left: Node
    right: Node


@dataclass
class Cond(Node):
    test: Node
    then: Node
    other: Node


# ---- lexer --------------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<string>"[^"\\]*")
  | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op>\*\*|==|!=|<=|>=|&&|\|\||[-+*/%^()\[\],.?:<>!])
""", re.VERBOSE)

KEYWORD_OPS = {"and": "&&", "or": "||", "not": "!"}


@dataclass
class Token:
    kind: str   # number | string | name | op | unit | end
    text: str
    pos: int
    end: int = -1   # exclusive end position in the source


def tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    while i < len(src):
        m = _TOKEN_RE.match(src, i)
        if not m:
            raise FormulaSyntaxError(f"Unexpected character '{src[i]}'", i, src)
        kind = m.lastgroup
        text = m.group()
        if kind == "number":
            tokens.append(Token("number", text, i, m.end()))
            # a '[' immediately after a number starts a unit annotation
            j = m.end()
            if j < len(src) and src[j] == "[":
                close = src.find("]", j)
                if close < 0:
                    raise FormulaSyntaxError("Unclosed unit annotation", j, src)
                tokens.append(Token("unit", src[j + 1:close].strip(), j, close + 1))
                i = close + 1
                continue
        elif kind == "name" and text in KEYWORD_OPS:
            tokens.append(Token("op", KEYWORD_OPS[text], i, m.end()))
        elif kind != "ws":
            tokens.append(Token(kind, text, i, m.end()))
        i = m.end()
    tokens.append(Token("end", "", len(src), len(src)))
    return tokens


# ---- units ------------------------------------------------------------------------------------------------

def parse_unit(text: str, registry: UnitRegistry | None = None) -> tuple[float, Dimension]:
    """Parse a unit expression such as 'm/s^2', 'kg*m^2', 'N·m', 'kg·m²/s', '1/s' into (SI factor, dimension)."""
    try:
        return parse_unit_expression(text, registry or default_registry())
    except UnitExpressionError as exc:
        raise UnitError(str(exc)) from None


# ---- parser ----------------------------------------------------------------------------------------------------

class Parser:
    def __init__(self, src: str, registry: UnitRegistry | None = None) -> None:
        self.src = src
        self.tokens = tokenize(src)
        self.i = 0
        self.registry = registry

    @property
    def tok(self) -> Token:
        return self.tokens[self.i]

    def error(self, msg: str, tok: Token | None = None) -> FormulaSyntaxError:
        t = tok or self.tok
        return FormulaSyntaxError(msg, t.pos, self.src)

    def _prev_end(self) -> int:
        return self.tokens[self.i - 1].end

    def accept(self, *ops: str) -> Token | None:
        t = self.tok
        if t.kind == "op" and t.text in ops:
            self.i += 1
            return t
        return None

    def expect(self, op: str) -> Token:
        t = self.accept(op)
        if t is None:
            raise self.error(f"Expected '{op}'" + (f" but found '{self.tok.text}'" if self.tok.text else ""))
        return t

    def parse(self) -> Node:
        if self.tok.kind == "end":
            raise self.error("Empty expression")
        node = self.ternary()
        if self.tok.kind != "end":
            raise self.error(f"Unexpected '{self.tok.text}'")
        return node

    def ternary(self) -> Node:
        node = self.logic_or()
        t = self.accept("?")
        if t:
            a = self.ternary()
            self.expect(":")
            b = self.ternary()
            node = Cond(node, a, b, pos=t.pos)
        return node

    def logic_or(self) -> Node:
        node = self.logic_and()
        while (t := self.accept("||")):
            node = Binary("||", node, self.logic_and(), pos=t.pos)
        return node

    def logic_and(self) -> Node:
        node = self.logic_not()
        while (t := self.accept("&&")):
            node = Binary("&&", node, self.logic_not(), pos=t.pos)
        return node

    def logic_not(self) -> Node:
        t = self.accept("!")
        if t:
            return Unary("!", self.logic_not(), pos=t.pos)
        return self.comparison()

    def comparison(self) -> Node:
        node = self.additive()
        t = self.accept("==", "!=", "<", "<=", ">", ">=")
        if t:
            node = Binary(t.text, node, self.additive(), pos=t.pos)
            if self.tok.kind == "op" and self.tok.text in ("==", "!=", "<", "<=", ">", ">="):
                raise self.error("Chained comparisons are not supported; use 'and'")
        return node

    def additive(self) -> Node:
        node = self.term()
        while (t := self.accept("+", "-")):
            node = Binary(t.text, node, self.term(), pos=t.pos)
        return node

    def term(self) -> Node:
        node = self.unary()
        while (t := self.accept("*", "/", "%")):
            node = Binary(t.text, node, self.unary(), pos=t.pos)
        return node

    def unary(self) -> Node:
        t = self.accept("-", "+")
        if t:
            operand = self.unary()
            if t.text == "+":
                return operand
            if isinstance(operand, Num):  # fold negative literals (keeps exponents constant)
                return Num(-operand.value, operand.dim, operand.unit_text, pos=t.pos, end=operand.end)
            return Unary("-", operand, pos=t.pos)
        return self.power()

    def power(self) -> Node:
        node = self.postfix()
        t = self.accept("^", "**")
        if t:
            node = Binary("^", node, self.unary(), pos=t.pos)
        return node

    def postfix(self) -> Node:
        node = self.primary()
        while True:
            if (t := self.accept(".")):
                if self.tok.kind != "name":
                    raise self.error("Expected an attribute name after '.'")
                node = Attr(node, self.tok.text, pos=t.pos, end=self.tok.end)
                self.i += 1
            elif (t := self.accept("[")):
                idx = self.ternary()
                self.expect("]")
                node = Index(node, idx, pos=t.pos, end=self._prev_end())
            elif self.tok.kind == "op" and self.tok.text == "(":
                if not isinstance(node, Name):
                    raise self.error("Only named functions can be called")
                self.i += 1
                args: list[Node] = []
                if not self.accept(")"):
                    while True:
                        args.append(self.ternary())
                        if self.accept(")"):
                            break
                        self.expect(",")
                node = Call(node.id, args, pos=node.pos, end=self._prev_end())
            else:
                return node

    def primary(self) -> Node:
        t = self.tok
        if t.kind == "number":
            self.i += 1
            value = float(t.text)
            if not math.isfinite(value):
                raise self.error("Number out of range", t)
            if self.tok.kind == "unit":
                ut = self.tok
                self.i += 1
                try:
                    factor, dim = parse_unit(ut.text, self.registry)
                except UnitError as exc:
                    raise self.error(str(exc), ut) from None
                return Num(value * factor, dim, ut.text, pos=t.pos, end=ut.end)
            return Num(value, pos=t.pos, end=t.end)
        if t.kind == "string":
            self.i += 1
            return Str(t.text[1:-1], pos=t.pos, end=t.end)
        if t.kind == "name":
            self.i += 1
            return Name(t.text, pos=t.pos, end=t.end)
        if (open_paren := self.accept("(")):
            node = self.ternary()
            self.expect(")")
            node.start, node.end = open_paren.pos, self._prev_end()
            return node
        if t.kind == "end":
            raise self.error("Unexpected end of expression")
        raise self.error(f"Unexpected '{t.text}'")


def children(node: Node) -> list[Node]:
    if isinstance(node, Attr):
        return [node.obj]
    if isinstance(node, Index):
        return [node.obj, node.index]
    if isinstance(node, Call):
        return list(node.args)
    if isinstance(node, Unary):
        return [node.operand]
    if isinstance(node, Binary):
        return [node.left, node.right]
    if isinstance(node, Cond):
        return [node.test, node.then, node.other]
    return []


def span(node: Node) -> tuple[int, int]:
    """(start, end) of the source text a node was parsed from (including parentheses around it)."""
    start, end = (node.start if node.start >= 0 else node.pos), node.end
    for c in children(node):
        cs, ce = span(c)
        start, end = min(start, cs), max(end, ce)
    return start, max(end, start)


def parse(src: str, registry: UnitRegistry | None = None) -> Node:
    if len(src) > 10_000:
        raise FormulaSyntaxError("Expression is too long (limit 10 000 characters)")
    try:
        return Parser(src, registry).parse()
    except RecursionError:
        raise FormulaSyntaxError("Expression is nested too deeply") from None
