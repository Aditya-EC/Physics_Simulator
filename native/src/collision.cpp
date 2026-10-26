// Native collision pipeline. Every narrowphase routine mirrors its Python twin in psim/physics/collision.py
// (same contact points, normals, depths and ordering), so the two backends agree to round-off.
#include "psim/collision.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

#include "psim/vecmath.hpp"

namespace psim {
namespace {

constexpr double kInf = 1e30;

struct Shape {
    int64_t kind;
    Vec3 c;
    double r[9];        // row-major rotation
    double radius;
    Vec3 half;
    double half_len;

    Vec3 col(int k) const { return {r[k], r[3 + k], r[6 + k]}; }
    Vec3 to_local(const Vec3& p) const {        // Rᵀ (p − c)
        const Vec3 d = p - c;
        return {dot(col(0), d), dot(col(1), d), dot(col(2), d)};
    }
    Vec3 to_world_dir(const Vec3& l) const { return col(0) * l.x + col(1) * l.y + col(2) * l.z; }
    void segment(Vec3& a, Vec3& b) const {
        const Vec3 axis = col(2) * half_len;
        a = c - axis;
        b = c + axis;
    }
};

struct Contact {
    Vec3 point, normal;
    double depth;
};
using Contacts = std::vector<Contact>;

inline double comp(const Vec3& v, int i) { return v[i]; }
inline double clamp01(double x) { return std::min(1.0, std::max(0.0, x)); }

Vec3 closest_point_segment(const Vec3& p, const Vec3& a, const Vec3& b) {
    const Vec3 ab = b - a;
    const double denom = dot(ab, ab);
    const double t = denom < 1e-18 ? 0.0 : clamp01(dot(p - a, ab) / denom);
    return a + ab * t;
}

void closest_points_segments(const Vec3& p1, const Vec3& q1, const Vec3& p2, const Vec3& q2, Vec3& c1, Vec3& c2) {
    const Vec3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    const double a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    double s = 0.0, t = 0.0;
    if (a < 1e-18 && e < 1e-18) {
        c1 = p1;
        c2 = p2;
        return;
    }
    if (a < 1e-18) {
        s = 0.0;
        t = clamp01(f / e);
    } else {
        const double c = dot(d1, r);
        if (e < 1e-18) {
            s = clamp01(-c / a);
            t = 0.0;
        } else {
            const double b = dot(d1, d2);
            const double denom = a * e - b * b;
            s = denom > 1e-18 ? clamp01((b * f - c * e) / denom) : 0.0;
            t = (b * s + f) / e;
            if (t < 0) {
                t = 0.0;
                s = clamp01(-c / a);
            } else if (t > 1) {
                t = 1.0;
                s = clamp01((b - c) / a);
            }
        }
    }
    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
}

Contacts sphere_sphere(const Vec3& c1, double r1, const Vec3& c2, double r2) {
    const Vec3 d = c2 - c1;
    const double dist = norm(d);
    if (dist >= r1 + r2) return {};
    const Vec3 n = dist > 1e-12 ? d / dist : Vec3{0, 0, 1};
    return {{c1 + n * (r1 - (r1 + r2 - dist) / 2), n, r1 + r2 - dist}};
}

Contacts sphere_plane(const Vec3& c, double r, const Shape& plane, bool sphere_is_a) {
    const Vec3 n = plane.col(2);
    const double dist = dot(c - plane.c, n);
    if (dist >= r) return {};
    return {{c - n * dist, sphere_is_a ? -n : n, r - dist}};
}

Contacts sphere_box(const Vec3& c, double r, const Shape& box) {
    const Vec3 local = box.to_local(c);
    bool inside = true;
    double cl[3];
    for (int k = 0; k < 3; ++k) {
        const double h = comp(box.half, k), x = comp(local, k);
        cl[k] = std::min(h, std::max(-h, x));
        if (std::fabs(x) > h) inside = false;
    }
    if (!inside) {
        const Vec3 q = box.c + box.to_world_dir({cl[0], cl[1], cl[2]});
        const Vec3 d = c - q;
        const double dist = norm(d);
        if (dist >= r) return {};
        return {{q, d / dist, r - dist}};
    }
    int k = 0;
    double gaps[3];
    for (int i = 0; i < 3; ++i) gaps[i] = comp(box.half, i) - std::fabs(comp(local, i));
    for (int i = 1; i < 3; ++i)
        if (gaps[i] < gaps[k]) k = i;
    double nl[3] = {0, 0, 0};
    nl[k] = comp(local, k) >= 0 ? 1.0 : -1.0;
    const Vec3 n = box.to_world_dir({nl[0], nl[1], nl[2]});
    return {{c - n * gaps[k], n, r + gaps[k]}};
}

void box_vertices(const Shape& box, Vec3 out[8]) {
    int i = 0;
    for (int x : {-1, 1})
        for (int y : {-1, 1})
            for (int z : {-1, 1})
                out[i++] = box.c + box.to_world_dir({x * box.half.x, y * box.half.y, z * box.half.z});
}

void sort_by_depth(Contacts& v, size_t keep) {
    std::stable_sort(v.begin(), v.end(), [](const Contact& a, const Contact& b) { return a.depth > b.depth; });
    if (v.size() > keep) v.resize(keep);
}

Contacts box_plane(const Shape& box, const Shape& plane, bool box_is_a) {
    const Vec3 n = plane.col(2);
    Vec3 verts[8];
    box_vertices(box, verts);
    Contacts out;
    for (const Vec3& p : verts) {
        const double d = dot(p - plane.c, n);
        if (d < 0) out.push_back({p - n * d, box_is_a ? -n : n, -d});
    }
    sort_by_depth(out, 4);
    return out;
}

Contacts capsule_plane(const Shape& cap, const Shape& plane, bool cap_is_a) {
    Vec3 a, b;
    cap.segment(a, b);
    Contacts out = sphere_plane(a, cap.radius, plane, cap_is_a);
    const Contacts more = sphere_plane(b, cap.radius, plane, cap_is_a);
    out.insert(out.end(), more.begin(), more.end());
    return out;
}

Contacts capsule_sphere(const Shape& cap, const Vec3& c, double r) {
    Vec3 a, b;
    cap.segment(a, b);
    return sphere_sphere(closest_point_segment(c, a, b), cap.radius, c, r);
}

Contacts capsule_capsule(const Shape& s1, const Shape& s2) {
    Vec3 a1, b1, a2, b2;
    s1.segment(a1, b1);
    s2.segment(a2, b2);
    const Vec3 d1 = b1 - a1, d2 = b2 - a2;
    const bool parallel = norm(cross(d1, d2)) < 1e-6 * std::max(1e-12, norm(d1) * norm(d2));
    if (parallel && norm(d1) > 1e-9) {
        Contacts out;
        for (const Vec3& p : {a2, b2}) {
            const Contacts c = sphere_sphere(closest_point_segment(p, a1, b1), s1.radius, p, s2.radius);
            out.insert(out.end(), c.begin(), c.end());
        }
        for (const Vec3& p : {a1, b1}) {
            for (const Contact& c : sphere_sphere(closest_point_segment(p, a2, b2), s2.radius, p, s1.radius))
                out.push_back({c.point, -c.normal, c.depth});
        }
        if (out.size() > 2) out.resize(2);
        return out;
    }
    Vec3 p, q;
    closest_points_segments(a1, b1, a2, b2, p, q);
    return sphere_sphere(p, s1.radius, q, s2.radius);
}

Contacts capsule_box(const Shape& cap, const Shape& box) {
    Vec3 a, b;
    cap.segment(a, b);
    const Vec3 samples[3] = {a, b, closest_point_segment(box.c, a, b)};
    Contacts out;
    for (const Vec3& p : samples) {
        for (const Contact& c : sphere_box(p, cap.radius, box)) {
            bool fresh = true;
            for (const Contact& o : out)
                if (norm(c.point - o.point) <= 1e-6) fresh = false;
            if (fresh) out.push_back(c);
        }
    }
    return out;
}

std::vector<Vec3> clip(const std::vector<Vec3>& poly, const Vec3& n, double d) {
    std::vector<Vec3> out;
    const size_t m = poly.size();
    for (size_t i = 0; i < m; ++i) {
        const Vec3& p = poly[i];
        const Vec3& q = poly[(i + 1) % m];
        const double dp = dot(n, p) - d, dq = dot(n, q) - d;
        if (dp <= 0) out.push_back(p);
        if (dp * dq < 0) out.push_back(p + (q - p) * (dp / (dp - dq)));
    }
    return out;
}

std::vector<Vec3> face(const Shape& box, int axis, double sign) {
    int i = axis == 0 ? 1 : 0, j = axis == 2 ? 1 : 2;
    const Vec3 c = box.c + box.col(axis) * (comp(box.half, axis) * sign);
    const Vec3 u = box.col(i) * comp(box.half, i), v = box.col(j) * comp(box.half, j);
    return {c + u + v, c - u + v, c - u - v, c + u - v};
}

double sum_abs_proj(const Shape& box, const Vec3& ax) {
    return box.half.x * std::fabs(dot(box.col(0), ax)) + box.half.y * std::fabs(dot(box.col(1), ax)) +
           box.half.z * std::fabs(dot(box.col(2), ax));
}

void support_edge(const Shape& box, int k, const Vec3& dir, Vec3& p, Vec3& q) {
    double s[3];
    for (int i = 0; i < 3; ++i) {
        const double x = dot(box.col(i), dir);
        s[i] = x > 0 ? 1.0 : (x < 0 ? -1.0 : 1.0);
    }
    s[k] = 0.0;
    const Vec3 mid = box.c + box.to_world_dir({s[0] * box.half.x, s[1] * box.half.y, s[2] * box.half.z});
    const Vec3 e = box.col(k) * comp(box.half, k);
    p = mid - e;
    q = mid + e;
}

Contacts box_box(const Shape& b1, const Shape& b2) {
    const Vec3 d = b2.c - b1.c;
    // candidate axes in the Python order: faces of b1, faces of b2, then edge pairs (i, j)
    struct Axis {
        Vec3 v;
        int kind;   // 0 face1, 1 face2, 2 edge
        int i, j;
    };
    Axis axes[15];
    int na = 0;
    for (int k = 0; k < 3; ++k) axes[na++] = {b1.col(k), 0, k, 0};
    for (int k = 0; k < 3; ++k) axes[na++] = {b2.col(k), 1, k, 0};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const Vec3 ax = cross(b1.col(i), b2.col(j));
            const double n = norm(ax);
            if (n > 1e-6) axes[na++] = {ax / n, 2, i, j};
        }
    double best = std::numeric_limits<double>::infinity(), best_overlap = 0.0;
    Vec3 n{0, 0, 1};
    int best_idx = -1;
    for (int k = 0; k < na; ++k) {
        const Vec3& ax = axes[k].v;
        const double overlap = sum_abs_proj(b1, ax) + sum_abs_proj(b2, ax) - std::fabs(dot(d, ax));
        if (overlap <= 0) return {};
        const double bias = axes[k].kind != 2 ? 1.0 : 1.05;
        if (overlap * bias < best) {
            best = overlap * bias;
            n = dot(d, ax) >= 0 ? ax : -ax;
            best_idx = k;
            best_overlap = overlap;
        }
    }
    const Axis& ba = axes[best_idx];
    if (ba.kind == 2) {
        Vec3 p1, q1, p2, q2, c1, c2;
        support_edge(b1, ba.i, n, p1, q1);
        support_edge(b2, ba.j, -n, p2, q2);
        closest_points_segments(p1, q1, p2, q2, c1, c2);
        return {{(c1 + c2) * 0.5, n, best_overlap}};
    }
    const Shape& ref = ba.kind == 0 ? b1 : b2;
    const Shape& inc = ba.kind == 0 ? b2 : b1;
    const bool flip = ba.kind != 0;
    const int ref_axis = ba.i;
    const Vec3 ref_n = flip ? -n : n;
    const double sign = dot(ref.col(ref_axis), ref_n) >= 0 ? 1.0 : -1.0;
    const Vec3 ref_face_n = ref.col(ref_axis) * sign;
    double dots[3];
    int inc_axis = 0;
    for (int k = 0; k < 3; ++k) dots[k] = dot(inc.col(k), ref_face_n);
    for (int k = 1; k < 3; ++k)
        if (std::fabs(dots[k]) > std::fabs(dots[inc_axis])) inc_axis = k;
    const double s = dots[inc_axis] > 0 ? -1.0 : (dots[inc_axis] < 0 ? 1.0 : -1.0);
    std::vector<Vec3> poly = face(inc, inc_axis, s);
    for (int k = 0; k < 3; ++k) {
        if (k == ref_axis) continue;
        const Vec3 side = ref.col(k);
        for (double sg : {1.0, -1.0}) {
            const Vec3 nn = side * sg;
            poly = clip(poly, nn, dot(nn, ref.c) + comp(ref.half, k));
            if (poly.empty()) return {};
        }
    }
    const double plane_d = dot(ref_face_n, ref.c) + comp(ref.half, ref_axis);
    Contacts out;
    for (const Vec3& p : poly) {
        const double depth = plane_d - dot(ref_face_n, p);
        if (depth > 0) out.push_back({p + ref_face_n * (depth / 2), flip ? -ref_face_n : ref_face_n, depth});
    }
    sort_by_depth(out, 8);
    return out;
}

Contacts flipped(Contacts v) {
    for (Contact& c : v) c.normal = -c.normal;
    return v;
}

Contacts collide(const Shape& a, const Shape& b) {
    const int64_t ka = a.kind, kb = b.kind;
    if (ka == kPlane && kb == kPlane) return {};
    if (kb == kPlane) {
        if (ka == kSphere) return sphere_plane(a.c, a.radius, b, true);
        if (ka == kBox) return box_plane(a, b, true);
        return capsule_plane(a, b, true);
    }
    if (ka == kPlane) return flipped(collide(b, a));
    if (ka == kSphere && kb == kSphere) return sphere_sphere(a.c, a.radius, b.c, b.radius);
    if (ka == kBox && kb == kSphere) return sphere_box(b.c, b.radius, a);
    if (ka == kCapsule && kb == kSphere) return capsule_sphere(a, b.c, b.radius);
    if (ka == kCapsule && kb == kCapsule) return capsule_capsule(a, b);
    if (ka == kBox && kb == kCapsule) return capsule_box(b, a);
    if (ka == kBox && kb == kBox) return box_box(a, b);
    return flipped(collide(b, a));
}

double bounding_radius(const Shape& s) {
    if (s.kind == kSphere) return s.radius;
    if (s.kind == kBox) return norm(s.half);
    if (s.kind == kCapsule) return s.radius + s.half_len;
    return std::numeric_limits<double>::infinity();
}

Shape shape_at(const ShapeSet& s, int64_t i) {
    Shape sh;
    sh.kind = s.kind[i];
    sh.c = {s.center[3 * i], s.center[3 * i + 1], s.center[3 * i + 2]};
    for (int k = 0; k < 9; ++k) sh.r[k] = s.rot[9 * i + k];
    sh.radius = s.radius[i];
    sh.half = {s.half[3 * i], s.half[3 * i + 1], s.half[3 * i + 2]};
    sh.half_len = s.half_len[i];
    return sh;
}

void friction_basis(const Vec3& n, Vec3& t1, Vec3& t2) {
    const Vec3 a = std::fabs(n.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    t1 = cross(n, a);
    t1 = t1 / std::sqrt(dot(t1, t1));
    t2 = cross(n, t1);
}

}  // namespace

std::vector<std::pair<int64_t, int64_t>> candidate_pairs(const ShapeSet& s, double margin) {
    std::vector<Shape> shapes;
    shapes.reserve(static_cast<size_t>(s.n));
    for (int64_t i = 0; i < s.n; ++i) shapes.push_back(shape_at(s, i));
    std::vector<std::pair<int64_t, int64_t>> out;
    std::vector<int64_t> finite, planes;
    std::vector<double> rad(static_cast<size_t>(s.n), 0.0);
    for (int64_t i = 0; i < s.n; ++i) {
        if (shapes[i].kind == kPlane) {
            planes.push_back(i);
        } else {
            finite.push_back(i);
            rad[i] = bounding_radius(shapes[i]);
        }
    }
    for (int64_t p : planes)
        for (int64_t i : finite) out.emplace_back(std::min(p, i), std::max(p, i));
    // sort and sweep along x: intervals [cx − r, cx + r] (extended by the margin)
    std::sort(finite.begin(), finite.end(), [&](int64_t a, int64_t b) {
        return shapes[a].c.x - rad[a] < shapes[b].c.x - rad[b];
    });
    for (size_t u = 0; u < finite.size(); ++u) {
        const int64_t i = finite[u];
        const double hi = shapes[i].c.x + rad[i] + margin;
        for (size_t t = u + 1; t < finite.size(); ++t) {
            const int64_t j = finite[t];
            if (shapes[j].c.x - rad[j] > hi) break;
            if (norm(shapes[j].c - shapes[i].c) <= rad[i] + rad[j] + margin)
                out.emplace_back(std::min(i, j), std::max(i, j));
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

void contact_rows(const ShapeSet& s, const BodySet& bodies, const int64_t* excluded, int64_t n_excluded,
                  double margin, double restitution_threshold, ContactOutput& out) {
    std::set<std::pair<int64_t, int64_t>> excl;
    for (int64_t k = 0; k < n_excluded; ++k) {
        excl.insert({excluded[2 * k], excluded[2 * k + 1]});
        excl.insert({excluded[2 * k + 1], excluded[2 * k]});
    }
    auto vec = [](const double* p, int64_t i) { return Vec3{p[3 * i], p[3 * i + 1], p[3 * i + 2]}; };
    auto add_row = [&](int64_t a, int64_t b, const Vec3& la, const Vec3& aa, const Vec3& lb, const Vec3& ab,
                       double rhs, double lo, double hi, int64_t parent, double mu, double err, int64_t mode) {
        out.ia.push_back(a);
        out.ib.push_back(b);
        for (const Vec3* x : {&la, &aa, &lb, &ab}) {
            out.J.push_back(x->x);
            out.J.push_back(x->y);
            out.J.push_back(x->z);
        }
        out.rhs.push_back(rhs);
        out.lo.push_back(lo);
        out.hi.push_back(hi);
        out.parent.push_back(parent);
        out.mu.push_back(mu);
        out.err.push_back(err);
        out.mode.push_back(mode);
        return static_cast<int64_t>(out.ia.size()) - 1;
    };
    for (const auto& pq : candidate_pairs(s, margin)) {
        const int64_t p = pq.first, q = pq.second;
        const int64_t a = s.body[p], b = s.body[q];
        if (a == b || !(bodies.dynamic[a] || bodies.dynamic[b]) || excl.count({a, b})) continue;
        const Contacts found = collide(shape_at(s, p), shape_at(s, q));
        if (found.empty()) continue;
        out.pairs.push_back(p);
        out.pairs.push_back(q);
        const double e = std::max(s.restitution[p], s.restitution[q]);
        const double mu = std::sqrt(std::max(s.friction[p], 0.0) * std::max(s.friction[q], 0.0));
        const Vec3 ca = vec(bodies.com, a), cb = vec(bodies.com, b);
        const Vec3 va = vec(bodies.v, a), vb = vec(bodies.v, b), wa = vec(bodies.w, a), wb = vec(bodies.w, b);
        for (const Contact& c : found) {
            const Vec3& n = c.normal;
            const Vec3 ra = c.point - ca, rb = c.point - cb;
            const double vn0 = dot((vb + cross(wb, rb)) - (va + cross(wa, ra)), n);
            const double target = vn0 < -restitution_threshold ? -e * vn0 : 0.0;
            const int64_t k = add_row(a, b, -n, -cross(ra, n), n, cross(rb, n), target, 0.0, kInf, -1, 0.0,
                                      -c.depth, 1);
            if (mu > 0) {
                Vec3 t1, t2;
                friction_basis(n, t1, t2);
                for (const Vec3* t : {&t1, &t2})
                    add_row(a, b, -*t, -cross(ra, *t), *t, cross(rb, *t), 0.0, -kInf, kInf, k, mu, 0.0, 2);
            }
            out.shape_a.push_back(p);
            out.shape_b.push_back(q);
            for (const Vec3* x : {&c.point, &c.normal}) {
                auto& dst = x == &c.point ? out.point : out.normal;
                dst.push_back(x->x);
                dst.push_back(x->y);
                dst.push_back(x->z);
            }
            out.depth.push_back(c.depth);
            out.row.push_back(k);
        }
    }
}

}  // namespace psim
