// Joint constraint rows (see psim/physics/joint_kernel.py for the row layout and conventions).
#include "psim/joints.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "psim/vecmath.hpp"

namespace psim {
namespace {

constexpr double kInf = 1e30;
constexpr double kLimitMargin = 0.05;
using Row = std::array<double, 12>;

Vec3 vec(const double* p) { return {p[0], p[1], p[2]}; }

Mat3 mat(const double* p) {
    return {{{p[0], p[1], p[2]}, {p[3], p[4], p[5]}, {p[6], p[7], p[8]}}};
}

Mat3 matmul(const Mat3& a, const Mat3& b) {
    Mat3 r{};
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) r[i][k] = a[i][0] * b[0][k] + a[i][1] * b[1][k] + a[i][2] * b[2][k];
    return r;
}

Mat3 transpose(const Mat3& a) {
    Mat3 r{};
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) r[i][k] = a[k][i];
    return r;
}

std::array<Vec3, 2> perp(const Vec3& n) {
    const Vec3 a = std::fabs(n.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 t1 = cross(n, a);
    t1 = t1 / std::sqrt(dot(t1, t1));
    return {t1, cross(n, t1)};
}

Vec3 rotvec(const Mat3& r) {
    const Quat q = mat3_to_quat(r);   // normalised, w >= 0
    const Vec3 u{q.x, q.y, q.z};
    const double s = norm(u);
    if (s < 1e-12) return u * 2.0;
    return u / s * (2.0 * std::atan2(s, q.w));
}

struct Geo {
    Mat3 Ra, Rb;
    Vec3 ra, rb, d, aw, bw;
};

Geo geometry(const JointSet& j, int64_t k, int64_t a, int64_t b, const BodyConfig& c) {
    Geo g;
    g.Ra = mat(c.rot + 9 * a);
    g.Rb = b >= 0 ? mat(c.rot + 9 * b) : mat3_identity();
    const Vec3 pa = vec(c.origin + 3 * a) + mul(g.Ra, vec(j.anchor_a + 3 * k));
    const Vec3 pb = b >= 0 ? vec(c.origin + 3 * b) + mul(g.Rb, vec(j.anchor_b + 3 * k)) : vec(j.anchor_b + 3 * k);
    g.d = pb - pa;
    g.ra = pa - vec(c.com + 3 * a);
    g.rb = b >= 0 ? pb - vec(c.com + 3 * b) : Vec3{};
    g.aw = mul(g.Ra, vec(j.axis_a + 3 * k));
    g.bw = mul(g.Rb, vec(j.axis_b + 3 * k));
    return g;
}

void put(Row& r, int at, const Vec3& v) {
    r[at] = v.x;
    r[at + 1] = v.y;
    r[at + 2] = v.z;
}

// A direction (function of the geometry) for linear or angular rows.
enum class Dir { kFixed, kTransported, kAxis, kNegAxis, kAlong, kNegAlong };

struct Direction {
    Dir type;
    Vec3 e;   // kFixed: the direction; kTransported: the vector carried ⟂ to the axis
    Vec3 of(const Geo& x) const {
        switch (type) {
            case Dir::kFixed: return e;
            case Dir::kTransported: {
                const Vec3 u = e - x.aw * dot(e, x.aw);
                return u / std::max(norm(u), 1e-12);
            }
            case Dir::kAxis: return x.aw;
            case Dir::kNegAxis: return -x.aw;
            case Dir::kAlong:
            case Dir::kNegAlong: {
                const double n = norm(x.d);
                const Vec3 u = n > 1e-9 ? x.d / n : x.aw;
                return type == Dir::kAlong ? u : -u;
            }
        }
        return e;
    }
};

struct Emitter {
    JointOutput& out;
    int64_t a, b;
    const Geo& g;
    const Geo* h;          // midpoint geometry or nullptr
    int64_t open = 0;      // block size to open on the next row
    int64_t inside = 0;    // rows left inside the open block

    Row lin_row(const Geo& x, const Direction& dir, bool extra) const {
        const Vec3 n = dir.of(x);
        const Vec3 arm_a = extra ? x.ra + x.d : x.ra;
        Row r{};
        put(r, 0, -n);
        put(r, 3, -cross(arm_a, n));
        if (b >= 0) {
            put(r, 6, n);
            put(r, 9, cross(x.rb, n));
        }
        return r;
    }

    Row ang_row(const Geo& x, const Direction& dir) const {
        const Vec3 n = dir.of(x);
        Row r{};
        put(r, 3, -n);
        if (b >= 0) put(r, 9, n);
        return r;
    }

    void emit(const Row& jr, const Row& br, double rhs, double lo, double hi, double err, int64_t mode) {
        if (open) {
            out.block.push_back(open);
            inside = open - 1;
            open = 0;
        } else if (inside) {
            out.block.push_back(0);
            --inside;
        } else {
            out.block.push_back(1);
        }
        out.ia.push_back(a);
        out.ib.push_back(b);
        out.J.insert(out.J.end(), jr.begin(), jr.end());
        out.B.insert(out.B.end(), br.begin(), br.end());
        out.rhs.push_back(rhs);
        out.lo.push_back(lo);
        out.hi.push_back(hi);
        out.err.push_back(err);
        out.mode.push_back(mode);
    }

    void lin(const Direction& dir, bool extra, double err, double rhs = 0.0, double lo = -kInf, double hi = kInf,
             int64_t mode = 0) {
        const Row jr = lin_row(g, dir, extra);
        emit(jr, h ? lin_row(*h, dir, extra) : jr, rhs, lo, hi, err, mode);
    }

    void ang(const Direction& dir, double err, double rhs = 0.0, double lo = -kInf, double hi = kInf,
             int64_t mode = 0) {
        const Row jr = ang_row(g, dir);
        emit(jr, h ? ang_row(*h, dir) : jr, rhs, lo, hi, err, mode);
    }
};

}  // namespace

void joint_rows(const JointSet& j, const BodyConfig& end, const BodyConfig* mid, double dt, double slop,
                JointOutput& out) {
    for (int64_t k = 0; k < j.n; ++k) {
        const int64_t a = j.body_a[k], b = j.body_b[k], type = j.kind[k], flags = j.flags[k];
        const Geo g = geometry(j, k, a, b, end);
        Geo hm;
        if (mid) hm = geometry(j, k, a, b, *mid);
        Emitter em{out, a, b, g, mid ? &hm : nullptr};
        const Vec3 d = g.d;
        const Vec3 E[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        static const int64_t kBlock[5] = {6, 3, 5, 5, 0};
        em.open = kBlock[type] >= 2 ? kBlock[type] : 0;

        // limit rows near the limits, then the motor row, for one joint coordinate;
        // `positive` increases the coordinate (its row velocity is d(value)/dt)
        auto limits_and_motor = [&](double value, auto&& positive, auto&& negative) {
            if (flags & kLimit) {
                if (value - j.lower[k] < kLimitMargin) {
                    const double c = value - j.lower[k];
                    positive(-std::max(c, 0.0) / dt, 0.0, kInf, c, 1);
                }
                if (j.upper[k] - value < kLimitMargin) {
                    const double c = j.upper[k] - value;
                    negative(-std::max(c, 0.0) / dt, 0.0, kInf, c, 1);
                }
            }
            if ((flags & kMotor) && j.motor_max[k] > 0) {
                const double impulse = j.motor_max[k] * dt;
                positive(j.motor_speed[k], -impulse, impulse, 0.0, 2);
            }
        };

        if (type == kFixed || type == kBall || type == kRevolute)
            for (int i = 0; i < 3; ++i) em.lin({Dir::kFixed, E[i]}, false, d[i]);
        if (type == kFixed || type == kPrismatic) {
            const Vec3 theta = rotvec(matmul(g.Rb, transpose(matmul(g.Ra, mat(j.ref_rot + 9 * k)))));
            for (int i = 0; i < 3; ++i) em.ang({Dir::kFixed, E[i]}, theta[i]);
        }
        if (type == kRevolute) {
            const Vec3 aw = g.aw;
            for (const Vec3& t : perp(aw)) em.ang({Dir::kTransported, t}, dot(cross(aw, g.bw), t));
            const Vec3 ua_l = perp(vec(j.axis_a + 3 * k))[0];
            const Vec3 ua = mul(g.Ra, ua_l);
            const Vec3 ub = mul(g.Rb, mul(transpose(mat(j.ref_rot + 9 * k)), ua_l));
            const double angle = -std::atan2(dot(cross(ua, ub), aw), dot(ua, ub));
            limits_and_motor(
                angle,
                [&](double rhs, double lo, double hi, double err, int64_t mode) {
                    em.ang({Dir::kNegAxis, {}}, err, rhs, lo, hi, mode);
                },
                [&](double rhs, double lo, double hi, double err, int64_t mode) {
                    em.ang({Dir::kAxis, {}}, err, rhs, lo, hi, mode);
                });
        }
        if (type == kPrismatic) {
            const Vec3 aw = g.aw;
            for (const Vec3& t : perp(aw)) em.lin({Dir::kTransported, t}, true, dot(d, t));
            const double travel = -dot(d, aw);
            limits_and_motor(
                travel,
                [&](double rhs, double lo, double hi, double err, int64_t mode) {
                    em.lin({Dir::kNegAxis, {}}, true, err, rhs, lo, hi, mode);
                },
                [&](double rhs, double lo, double hi, double err, int64_t mode) {
                    em.lin({Dir::kAxis, {}}, true, err, rhs, lo, hi, mode);
                });
        }
        if (type == kDistance) {
            const double dist = norm(d);
            if (flags & kRope) {
                if (dist > j.length[k] - slop) {   // near taut: may extend only up to the length
                    const double slack = j.length[k] - dist;
                    em.lin({Dir::kNegAlong, {}}, false, slack, -std::max(slack, 0.0) / dt, 0.0, kInf, 1);
                }
            } else {
                em.lin({Dir::kAlong, {}}, false, dist - j.length[k]);
            }
        }
    }
}

}  // namespace psim
