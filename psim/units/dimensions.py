"""Physical dimensions as exponent vectors.

Base dimensions: length L, mass M, time T, electric current I,
temperature Θ, amount N, luminous intensity J, plus a plane-angle
pseudo-dimension A. Treating angle as a tracked dimension (rather than as
dimensionless) keeps rad/s distinct from Hz and lets the formula engine
flag accidental mixing of angles and plain numbers; conversions between
angle units are still exact.
"""
from __future__ import annotations

from dataclasses import dataclass

BASE_SYMBOLS = ("L", "M", "T", "I", "Θ", "N", "J", "A")


@dataclass(frozen=True)
class Dimension:
    exponents: tuple[int, ...] = (0,) * len(BASE_SYMBOLS)

    @staticmethod
    def of(**kw: int) -> "Dimension":
        names = {"L": 0, "M": 1, "T": 2, "I": 3, "Theta": 4, "N": 5, "J": 6, "A": 7}
        e = [0] * len(BASE_SYMBOLS)
        for k, v in kw.items():
            e[names[k]] = v
        return Dimension(tuple(e))

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents)))

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents)))

    def __pow__(self, p: int) -> "Dimension":
        return Dimension(tuple(a * p for a in self.exponents))

    @property
    def is_dimensionless(self) -> bool:
        return not any(self.exponents)

    def __str__(self) -> str:
        if self.is_dimensionless:
            return "1"
        parts = []
        for sym, e in zip(BASE_SYMBOLS, self.exponents):
            if e == 1:
                parts.append(sym)
            elif e:
                parts.append(f"{sym}^{e}")
        return "·".join(parts)


DIMENSIONLESS = Dimension()
LENGTH = Dimension.of(L=1)
MASS = Dimension.of(M=1)
TIME = Dimension.of(T=1)
ANGLE = Dimension.of(A=1)
AREA = LENGTH ** 2
VOLUME = LENGTH ** 3
VELOCITY = LENGTH / TIME
ACCELERATION = VELOCITY / TIME
FORCE = MASS * ACCELERATION
ENERGY = FORCE * LENGTH
POWER = ENERGY / TIME
PRESSURE = FORCE / AREA
ANGULAR_VELOCITY = ANGLE / TIME
ANGULAR_ACCELERATION = ANGULAR_VELOCITY / TIME
# Torque is force x lever arm; it shares the dimension of energy (as in SI).
TORQUE = FORCE * LENGTH
MOMENTUM = MASS * VELOCITY
ANGULAR_MOMENTUM = MASS * AREA * ANGLE / TIME
DENSITY = MASS / VOLUME
AREAL_DENSITY = MASS / AREA
INERTIA = MASS * AREA
FREQUENCY = DIMENSIONLESS / TIME
