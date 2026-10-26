// Native collision pipeline. Every narrowphase routine mirrors its Python twin in psim/physics/collision.py
// (same contact points, normals, depths and ordering), so the two backends agree to round-off.
#include "psim/collision.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
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
    const HullSet* hs = nullptr;    // hull data (kHull, or a box paired with a hull)
    int64_t hid = -1;

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

// ---- convex hulls (mirror psim/physics/collision.py) -------------------------------------------------------

struct WorldHull {
    const HullSet* hs;
    int64_t v0, nv, f0, nf, e0, ne;
    std::vector<Vec3> v, n;
    std::vector<double> d;

    explicit WorldHull(const Shape& s) : hs(s.hs) {
        const int64_t h = s.hid;
        v0 = hs->vert_off[h];
        nv = hs->vert_off[h + 1] - v0;
        f0 = hs->face_off[h];
        nf = hs->face_off[h + 1] - f0;
        e0 = hs->edge_off[h];
        ne = hs->edge_off[h + 1] - e0;
        v.reserve(nv);
        for (int64_t i = 0; i < nv; ++i) {
            const double* p = hs->verts + 3 * (v0 + i);
            v.push_back(s.c + s.to_world_dir({p[0], p[1], p[2]}));
        }
        for (int64_t f = 0; f < nf; ++f) {
            const double* p = hs->normals + 3 * (f0 + f);
            const Vec3 nw = s.to_world_dir({p[0], p[1], p[2]});
            n.push_back(nw);
            d.push_back(hs->offsets[f0 + f] + dot(nw, s.c));
        }
    }
    int64_t face_size(int64_t f) const { return hs->face_vert_off[f0 + f + 1] - hs->face_vert_off[f0 + f]; }
    const Vec3& face_vert(int64_t f, int64_t k) const { return v[hs->face_verts[hs->face_vert_off[f0 + f] + k]]; }
    int64_t edge(int64_t e, int k) const { return hs->edges[2 * (e0 + e) + k]; }
    int64_t edge_face(int64_t e, int k) const { return hs->edge_faces[2 * (e0 + e) + k]; }
    Vec3 mean() const {
        Vec3 m{0, 0, 0};
        for (const Vec3& p : v) m += p;
        return m / static_cast<double>(v.size());
    }
};

double norm2(const Vec3& a) { return dot(a, a); }

// At most 4 well-spread points (deepest, farthest from it, largest triangle, largest enlargement).
std::vector<size_t> reduce_manifold(const std::vector<Vec3>& p, const std::vector<double>& depth) {
    const size_t n = p.size();
    std::vector<size_t> out;
    if (n <= 4) {
        for (size_t i = 0; i < n; ++i) out.push_back(i);
        return out;
    }
    size_t i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    for (size_t i = 1; i < n; ++i)
        if (depth[i] > depth[i0]) i0 = i;
    double best = -1;
    for (size_t i = 0; i < n; ++i) {
        const double d = norm2(p[i] - p[i0]);
        if (d > best) best = d, i1 = i;
    }
    best = -1;
    for (size_t i = 0; i < n; ++i) {
        const double a = norm(cross(p[i1] - p[i0], p[i] - p[i0]));
        if (a > best) best = a, i2 = i;
    }
    const double tri = best;
    best = -1;
    for (size_t i = 0; i < n; ++i) {
        const double sc = norm(cross(p[i0] - p[i], p[i1] - p[i])) + norm(cross(p[i1] - p[i], p[i2] - p[i])) +
                          norm(cross(p[i2] - p[i], p[i0] - p[i]));
        if (sc > best) best = sc, i3 = i;
    }
    out = {i0, i1, i2};
    if (best > tri * (1 + 1e-9) && i3 != i0 && i3 != i1 && i3 != i2) out.push_back(i3);
    std::sort(out.begin(), out.end());  // in clipping order, independent of near-ties in the picks
    return out;
}

Contacts reduced_contacts(const std::vector<Vec3>& pts, const std::vector<double>& depths, const Vec3& normal) {
    Contacts out;
    for (size_t k : reduce_manifold(pts, depths)) out.push_back({pts[k], normal, depths[k]});
    return out;
}

Contacts hull_hull(const Shape& a, const Shape& b) {
    const WorldHull A(a), B(b);
    double sep_a = -std::numeric_limits<double>::infinity(), sep_b = sep_a;
    int64_t fa = 0, fb = 0;
    for (int64_t f = 0; f < A.nf; ++f) {
        double mn = std::numeric_limits<double>::infinity();
        for (const Vec3& p : B.v) mn = std::min(mn, dot(p, A.n[f]));
        if (mn - A.d[f] > sep_a) sep_a = mn - A.d[f], fa = f;
    }
    if (sep_a > 0) return {};
    for (int64_t f = 0; f < B.nf; ++f) {
        double mn = std::numeric_limits<double>::infinity();
        for (const Vec3& p : A.v) mn = std::min(mn, dot(p, B.n[f]));
        if (mn - B.d[f] > sep_b) sep_b = mn - B.d[f], fb = f;
    }
    if (sep_b > 0) return {};
    double best_e = -std::numeric_limits<double>::infinity();
    int64_t ie = -1, je = -1;
    Vec3 n_e{0, 0, 1};
    bool any = false;
    const Vec3 cen = A.mean();
    for (int64_t i = 0; i < A.ne; ++i) {
        const Vec3 an = A.n[A.edge_face(i, 0)], bn = A.n[A.edge_face(i, 1)];
        const Vec3 bxa = cross(bn, an);
        const Vec3 p0 = A.v[A.edge(i, 0)], da = A.v[A.edge(i, 1)] - p0;
        for (int64_t j = 0; j < B.ne; ++j) {
            const Vec3 cn = -B.n[B.edge_face(j, 0)], dn = -B.n[B.edge_face(j, 1)];
            const Vec3 dxc = cross(dn, cn);
            const double cba = dot(cn, bxa), dba = dot(dn, bxa), adc = dot(an, dxc), bdc = dot(bn, dxc);
            if (!(cba * dba < 0 && adc * bdc < 0 && cba * bdc > 0)) continue;
            const Vec3 q0 = B.v[B.edge(j, 0)], db = B.v[B.edge(j, 1)] - q0;
            Vec3 n = cross(da, db);
            const double ln = norm(n);
            if (!(ln > 1e-6 * norm(da) * norm(db))) continue;
            n = n / ln;
            if (dot(n, p0 - cen) < 0) n = -n;
            const double sep = dot(n, q0 - p0);
            if (!any || sep > best_e) {
                any = true;
                best_e = sep, ie = i, je = j, n_e = n;
            }
        }
    }
    if (any && best_e > 0) return {};
    if (ie >= 0 && best_e > 0.95 * std::max(sep_a, sep_b) + 1e-4) {
        Vec3 c1, c2;
        closest_points_segments(A.v[A.edge(ie, 0)], A.v[A.edge(ie, 1)], B.v[B.edge(je, 0)], B.v[B.edge(je, 1)],
                                c1, c2);
        return {{(c1 + c2) * 0.5, n_e, -best_e}};
    }
    const bool flip = sep_b > 0.95 * sep_a + 1e-4;
    const WorldHull& R = flip ? B : A;
    const WorldHull& I = flip ? A : B;
    const int64_t rf = flip ? fb : fa;
    const Vec3 ref_n = R.n[rf];
    const double ref_d = R.d[rf];
    int64_t inc = 0;
    for (int64_t f = 1; f < I.nf; ++f)
        if (dot(I.n[f], ref_n) < dot(I.n[inc], ref_n)) inc = f;
    std::vector<Vec3> poly;
    for (int64_t k = 0; k < I.face_size(inc); ++k) poly.push_back(I.face_vert(inc, k));
    const int64_t m = R.face_size(rf);
    for (int64_t k = 0; k < m; ++k) {
        const Vec3 e = R.face_vert(rf, (k + 1) % m) - R.face_vert(rf, k);
        const Vec3 side = cross(e, ref_n);
        poly = clip(poly, side, dot(side, R.face_vert(rf, k)));
        if (poly.empty()) return {};
    }
    std::vector<Vec3> pts;
    std::vector<double> depths;
    for (const Vec3& p : poly) {
        const double depth = ref_d - dot(ref_n, p);
        if (depth > 0) {
            pts.push_back(p + ref_n * (depth / 2));
            depths.push_back(depth);
        }
    }
    return reduced_contacts(pts, depths, flip ? -ref_n : ref_n);
}

Contacts hull_plane(const Shape& h, const Shape& plane, bool hull_is_a) {
    const WorldHull H(h);
    const Vec3 n = plane.col(2);
    std::vector<Vec3> pts;
    std::vector<double> depths;
    for (const Vec3& p : H.v) {
        const double d = dot(p - plane.c, n);
        if (d < 0) {
            pts.push_back(p - n * d);
            depths.push_back(-d);
        }
    }
    return reduced_contacts(pts, depths, hull_is_a ? -n : n);
}

Contacts sphere_hull(const Vec3& c, double r, const Shape& h) {
    const WorldHull H(h);
    int64_t k = 0;
    std::vector<double> s(static_cast<size_t>(H.nf));
    for (int64_t f = 0; f < H.nf; ++f) {
        s[f] = dot(H.n[f], c) - H.d[f];
        if (s[f] > s[k]) k = f;
    }
    if (s[k] <= 0) return {{c - H.n[k] * s[k], H.n[k], r - s[k]}};
    double best = std::numeric_limits<double>::infinity();
    Vec3 q{0, 0, 0};
    for (int64_t f = 0; f < H.nf; ++f) {
        if (s[f] <= 0) continue;
        const Vec3 p = c - H.n[f] * s[f];
        const int64_t m = H.face_size(f);
        bool inside = true;
        for (int64_t i = 0; i < m && inside; ++i) {
            const Vec3& a = H.face_vert(f, i);
            if (dot(cross(H.face_vert(f, (i + 1) % m) - a, H.n[f]), p - a) > 0) inside = false;
        }
        if (inside && s[f] < best) best = s[f], q = p;
    }
    for (int64_t e = 0; e < H.ne; ++e) {
        const Vec3 p = closest_point_segment(c, H.v[H.edge(e, 0)], H.v[H.edge(e, 1)]);
        const double dist = norm(c - p);
        if (dist < best) best = dist, q = p;
    }
    if (best >= r) return {};
    return {{q, (c - q) / best, r - best}};
}

Contacts capsule_hull(const Shape& cap, const Shape& h) {
    Vec3 a, b;
    cap.segment(a, b);
    const Vec3 samples[3] = {a, b, closest_point_segment(WorldHull(h).mean(), a, b)};
    Contacts out;
    for (const Vec3& p : samples) {
        for (const Contact& c : sphere_hull(p, cap.radius, h)) {
            bool fresh = true;
            for (const Contact& o : out)
                if (norm(c.point - o.point) <= 1e-6) fresh = false;
            if (fresh) out.push_back(c);
        }
    }
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
        if (ka == kHull) return hull_plane(a, b, true);
        return capsule_plane(a, b, true);
    }
    if (ka == kPlane) return flipped(collide(b, a));
    if (ka == kHull) {
        if (kb == kHull || kb == kBox) {
            if (!a.hs || !b.hs) throw std::invalid_argument("hull pair without hull data");
            return hull_hull(a, b);
        }
        if (kb == kSphere) return sphere_hull(b.c, b.radius, a);
        return capsule_hull(b, a);
    }
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
    if (s.kind == kBox || s.kind == kHull) return norm(s.half);
    if (s.kind == kCapsule) return s.radius + s.half_len;
    return std::numeric_limits<double>::infinity();
}

Shape shape_at(const ShapeSet& s, int64_t i, bool start = false) {
    Shape sh;
    const double* center = start && s.center0 ? s.center0 : s.center;
    const double* rot = start && s.rot0 ? s.rot0 : s.rot;
    sh.kind = s.kind[i];
    sh.c = {center[3 * i], center[3 * i + 1], center[3 * i + 2]};
    for (int k = 0; k < 9; ++k) sh.r[k] = rot[9 * i + k];
    sh.radius = s.radius[i];
    sh.half = {s.half[3 * i], s.half[3 * i + 1], s.half[3 * i + 2]};
    sh.half_len = s.half_len[i];
    if (s.hull_id && s.hulls && s.hull_id[i] >= 0) {
        sh.hs = s.hulls;
        sh.hid = s.hull_id[i];
    }
    return sh;
}

double thickness(const Shape& s) {
    if (s.kind == kSphere || s.kind == kCapsule) return 2.0 * s.radius;
    if (s.kind == kBox || s.kind == kHull) return 2.0 * std::min(s.half.x, std::min(s.half.y, s.half.z));
    return std::numeric_limits<double>::infinity();
}

Shape inflated(Shape s, double m) {
    if (s.kind == kSphere || s.kind == kCapsule) s.radius += m;
    if (s.kind == kBox || s.kind == kHull) {   // a fast hull is tested speculatively through its bounding box
        s.kind = kBox;
        s.hs = nullptr;
        s.half = s.half + Vec3{m, m, m};
    }
    return s;
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
            rad[i] = bounding_radius(shapes[i]) + (s.sweep ? s.sweep[i] : 0.0);   // swept bounds (CCD)
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
                  double margin, double restitution_threshold, ContactOutput& out, double dt) {
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
        const Shape sp = shape_at(s, p), sq = shape_at(s, q);
        const double swp = s.sweep ? s.sweep[p] : 0.0, swq = s.sweep ? s.sweep[q] : 0.0;
        const double m = swp + swq;
        const bool speculative = dt > 0 && m > 0.5 * std::min(thickness(sp), thickness(sq));
        const Contacts found = speculative
            ? collide(inflated(shape_at(s, p, true), swp), inflated(shape_at(s, q, true), swq))
            : collide(sp, sq);
        if (found.empty()) continue;
        out.pairs.push_back(p);
        out.pairs.push_back(q);
        const double e = std::max(s.restitution[p], s.restitution[q]);
        const double mu = std::sqrt(std::max(s.friction[p], 0.0) * std::max(s.friction[q], 0.0));
        const double* com = speculative && bodies.com0 ? bodies.com0 : bodies.com;
        const Vec3 ca = vec(com, a), cb = vec(com, b);
        const Vec3 va = vec(bodies.v, a), vb = vec(bodies.v, b), wa = vec(bodies.w, a), wb = vec(bodies.w, b);
        for (const Contact& c : found) {
            const Vec3& n = c.normal;
            const Vec3 ra = c.point - ca, rb = c.point - cb;
            const double vn0 = dot((vb + cross(wb, rb)) - (va + cross(wa, ra)), n);
            const double depth = speculative ? c.depth - m : c.depth;   // < 0: still separated by a gap
            const double gap = std::max(-depth, 0.0);
            double target;
            if (vn0 < -restitution_threshold && gap + vn0 * dt <= 0.0)
                target = -e * vn0;                                         // they meet within this step: bounce
            else
                target = gap > 0.0 ? -gap / dt : 0.0;                      // may close the gap, not cross it
            // position error at the end of the step (speculative pairs were measured at its start)
            const double err = speculative ? -depth + vn0 * dt : -depth;
            const int64_t k = add_row(a, b, -n, -cross(ra, n), n, cross(rb, n), target, 0.0, kInf, -1, 0.0, err, 1);
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
            out.depth.push_back(depth);
            out.row.push_back(k);
        }
    }
}

}  // namespace psim
