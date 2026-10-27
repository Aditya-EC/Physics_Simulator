// Approximate convex decomposition (see psim/physics/decompose.py for the algorithm).
#include "psim/decompose.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "psim/vecmath.hpp"

namespace psim {
namespace {

constexpr double kPositions[3] = {0.25, 0.5, 0.75};
constexpr double kLookaheadPositions[1] = {0.5};
constexpr double kImprovement = 0.9;
constexpr size_t kMaxVertexCuts = 8;

struct Tri {
    Vec3 a, b, c;
};
using Chain = std::vector<Tri>;

Vec3 vec(const Point3& p) { return {p[0], p[1], p[2]}; }

double chain_volume(const Chain& t) {
    double s = 0.0;
    for (const Tri& x : t) s += dot(x.a, cross(x.b, x.c));
    return s / 6.0;
}

Vec3 lerp(const Vec3& a, const Vec3& b, double sa, double sb) {
    const double t = sa / (sa - sb);
    return a + (b - a) * t;
}

// The part of a closed chain with n·x <= d, closed by a cap fanned over the oriented cut segments.
Chain clip(const Chain& tris, const Vec3& n, double d) {
    Chain full, ones, twos_a, twos_b;
    std::vector<std::pair<Vec3, Vec3>> seg_one, seg_two;
    for (const Tri& t : tris) {
        const Vec3 v[3] = {t.a, t.b, t.c};
        const double s[3] = {dot(t.a, n) - d, dot(t.b, n) - d, dot(t.c, n) - d};
        const bool in[3] = {s[0] <= 0.0, s[1] <= 0.0, s[2] <= 0.0};
        const int count = int(in[0]) + int(in[1]) + int(in[2]);
        if (count == 3) {
            full.push_back(t);
        } else if (count == 1) {
            const int i = in[0] ? 0 : (in[1] ? 1 : 2), j = (i + 1) % 3, k = (i + 2) % 3;
            const Vec3 pij = lerp(v[i], v[j], s[i], s[j]), pik = lerp(v[i], v[k], s[i], s[k]);
            ones.push_back({v[i], pij, pik});
            seg_one.emplace_back(pik, pij);
        } else if (count == 2) {
            const int k = !in[0] ? 0 : (!in[1] ? 1 : 2), i = (k + 1) % 3, j = (k + 2) % 3;
            const Vec3 pjk = lerp(v[j], v[k], s[j], s[k]), pki = lerp(v[k], v[i], s[k], s[i]);
            twos_a.push_back({v[i], v[j], pjk});
            twos_b.push_back({v[i], pjk, pki});
            seg_two.emplace_back(pki, pjk);
        }
    }
    Chain out;
    out.reserve(full.size() + ones.size() + 2 * twos_a.size() + seg_one.size() + seg_two.size());
    auto keep = [&](const Tri& t) {
        if (norm(cross(t.b - t.a, t.c - t.a)) > 0.0) out.push_back(t);
    };
    for (const Tri& t : full) keep(t);
    for (const Tri& t : ones) keep(t);
    for (const Tri& t : twos_a) keep(t);
    for (const Tri& t : twos_b) keep(t);
    if (!seg_one.empty() || !seg_two.empty()) {
        const Vec3 o = !seg_one.empty() ? seg_one[0].first : seg_two[0].first;
        for (const auto& s : seg_one) keep({o, s.first, s.second});
        for (const auto& s : seg_two) keep({o, s.first, s.second});
    }
    return out;
}

std::vector<Point3> points_of(const Chain& t) {
    std::vector<Point3> p;
    p.reserve(3 * t.size());
    for (const Tri& x : t)
        for (const Vec3& v : {x.a, x.b, x.c}) p.push_back({v.x, v.y, v.z});
    std::sort(p.begin(), p.end());
    p.erase(std::unique(p.begin(), p.end()), p.end());
    return p;
}

// Quickhull (same construction as hull._triangle_hull, see there); returns the hull volume, 0 for degenerate sets.
double exact_hull_volume(const std::vector<Vec3>& p, double size) {
    const int64_t n = static_cast<int64_t>(p.size());
    if (n < 4) return 0.0;
    const double eps = 1e-9 * size;   // as hull.HULL_EPS
    int64_t i0 = 0;
    for (int64_t i = 1; i < n; ++i)
        if (p[i].x < p[i0].x) i0 = i;
    int64_t i1 = 0;
    double best = -1.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = norm(p[i] - p[i0]);
        if (d > best) best = d, i1 = i;
    }
    const Vec3 line = p[i1] - p[i0];
    int64_t i2 = 0;
    best = -1.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = norm(cross(p[i] - p[i0], line));
        if (d > best) best = d, i2 = i;
    }
    const double d2 = best;
    const Vec3 nrm = cross(p[i1] - p[i0], p[i2] - p[i0]);
    int64_t i3 = 0;
    best = -1.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = std::fabs(dot(p[i] - p[i0], nrm));
        if (d > best) best = d, i3 = i;
    }
    if (d2 <= eps * norm(line) || best <= eps * norm(nrm)) return 0.0;   // coplanar
    const Vec3 centre = (p[i0] + p[i1] + p[i2] + p[i3]) / 4.0;

    struct Face {
        int64_t v[3];
        Vec3 n;
        double d;
        bool alive;
    };
    std::vector<Face> faces;
    std::map<int64_t, std::vector<int64_t>> outside;   // live face → points above it (ordered: first face first)
    std::unordered_map<uint64_t, int64_t> owner;        // directed edge (a, b) → live face
    auto key = [n](int64_t a, int64_t b) { return static_cast<uint64_t>(a) * static_cast<uint64_t>(n) + b; };
    auto add = [&](int64_t a, int64_t b, int64_t c) {
        Vec3 nn = cross(p[b] - p[a], p[c] - p[a]);
        const double l = norm(nn);
        if (l > 0) nn = nn / l;
        const int64_t k = static_cast<int64_t>(faces.size());
        faces.push_back({{a, b, c}, nn, dot(nn, p[a]), true});
        owner[key(a, b)] = owner[key(b, c)] = owner[key(c, a)] = k;
        return k;
    };
    auto assign = [&](const std::vector<int64_t>& points, const std::vector<int64_t>& fs) {
        if (fs.empty()) return;
        for (int64_t q : points) {
            size_t bj = 0;
            double bd = -std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < fs.size(); ++j) {
                const double d = dot(p[q], faces[fs[j]].n) - faces[fs[j]].d;
                if (d > bd) bd = d, bj = j;
            }
            if (bd > eps) outside[fs[bj]].push_back(q);
        }
    };
    auto edges_of = [&](int64_t k, int e) {
        return std::make_pair(faces[k].v[e], faces[k].v[(e + 1) % 3]);
    };
    // boundary of a set of faces as one simple loop, or false (pinch or hole)
    auto horizon_of = [&](const std::vector<char>& in, const std::vector<int64_t>& region,
                          std::vector<std::pair<int64_t, int64_t>>& out) {
        out.clear();
        for (int64_t k : region)
            for (int e = 0; e < 3; ++e) {
                const auto ed = edges_of(k, e);
                if (!in[owner.at(key(ed.second, ed.first))]) out.push_back(ed);
            }
        if (out.empty()) return false;
        std::unordered_map<int64_t, int64_t> nxt;
        for (const auto& e : out)
            if (!nxt.emplace(e.first, e.second).second) return false;   // pinched
        int64_t v = out[0].first;
        size_t steps = 0;
        for (;;) {
            const auto it = nxt.find(v);
            ++steps;
            if (it == nxt.end() || steps > out.size()) return false;
            v = it->second;
            if (v == out[0].first) return steps == out.size();          // several loops: a hole
        }
    };
    std::vector<int64_t> tet;
    const int64_t tv[4][3] = {{i0, i1, i2}, {i0, i1, i3}, {i0, i2, i3}, {i1, i2, i3}};
    for (const auto& t : tv) {
        int64_t a = t[0], b = t[1], c = t[2];
        if (dot(cross(p[b] - p[a], p[c] - p[a]), p[a] - centre) < 0) std::swap(b, c);
        tet.push_back(add(a, b, c));
    }
    {
        std::vector<int64_t> rest;
        for (int64_t i = 0; i < n; ++i)
            if (i != i0 && i != i1 && i != i2 && i != i3) rest.push_back(i);
        assign(rest, tet);
    }
    std::vector<char> in;
    std::vector<int64_t> region, stack, freed, sorted_region, created;
    std::vector<std::pair<int64_t, int64_t>> horizon;
    while (!outside.empty()) {
        const int64_t f0 = outside.begin()->first;
        const std::vector<int64_t>& pts = outside.begin()->second;
        int64_t i = pts[0];
        double bd = -std::numeric_limits<double>::infinity();
        for (int64_t q : pts) {
            const double d = dot(p[q], faces[f0].n) - faces[f0].d;
            if (d > bd) bd = d, i = q;
        }
        auto dist = [&](int64_t k) { return dot(faces[k].n, p[i]) - faces[k].d; };
        in.assign(faces.size(), 0);
        region.assign(1, f0);
        in[f0] = 1;
        stack.assign(1, f0);
        while (!stack.empty()) {                 // the faces the point sees, connected to f0
            const int64_t k = stack.back();
            stack.pop_back();
            for (int e = 0; e < 3; ++e) {
                const auto ed = edges_of(k, e);
                const int64_t g = owner.at(key(ed.second, ed.first));
                if (!in[g] && dist(g) > eps) {
                    in[g] = 1;
                    region.push_back(g);
                    stack.push_back(g);
                }
            }
        }
        std::sort(region.begin(), region.end());
        bool ok = horizon_of(in, region, horizon);
        for (int attempt = 0; attempt < 2 && !ok; ++attempt) {   // nearly coplanar neighbours, then all
            std::vector<int64_t> near;
            for (int64_t k : region)
                for (int e = 0; e < 3; ++e) {
                    const auto ed = edges_of(k, e);
                    near.push_back(owner.at(key(ed.second, ed.first)));
                }
            for (int64_t g : near)
                if (!in[g] && (attempt == 1 || dist(g) > -eps)) {
                    in[g] = 1;
                    region.push_back(g);
                }
            std::sort(region.begin(), region.end());
            ok = horizon_of(in, region, horizon);
        }
        freed.clear();
        for (int64_t k : region) {
            const auto it = outside.find(k);
            if (it == outside.end()) continue;
            for (int64_t q : it->second)
                if (q != i) freed.push_back(q);
            outside.erase(it);
        }
        if (!ok) {                               // could not be repaired: leave the point out
            assign(freed, region);
            continue;
        }
        for (int64_t k : region) {
            faces[k].alive = false;
            for (int e = 0; e < 3; ++e) {
                const auto ed = edges_of(k, e);
                const auto it = owner.find(key(ed.first, ed.second));
                if (it != owner.end() && it->second == k) owner.erase(it);
            }
        }
        created.clear();
        for (const auto& h : horizon) created.push_back(add(h.first, h.second, i));
        assign(freed, created);
    }
    Vec3 c{0, 0, 0};
    for (const Vec3& q : p) c += q;
    c = c / static_cast<double>(n);
    double v = 0.0;
    for (const Face& f : faces)
        if (f.alive) v += dot(p[f.v[0]] - c, cross(p[f.v[1]] - c, p[f.v[2]] - c));
    return v / 6.0;
}

std::vector<Vec3> fibonacci_directions(int64_t n) {
    std::vector<Vec3> d;
    d.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const double k = static_cast<double>(i) + 0.5;
        const double phi = std::acos(1.0 - 2.0 * k / static_cast<double>(n));
        const double theta = kPi * (1.0 + std::sqrt(5.0)) * k;
        d.push_back({std::cos(theta) * std::sin(phi), std::sin(theta) * std::sin(phi), std::cos(phi)});
    }
    return d;
}

struct Cut {
    double cost;
    Chain left, right;
    double cl, cr;
};

std::pair<double, double> concavity(const Chain& t, int64_t max_points) {
    const double vol = chain_volume(t);
    if (vol <= 0.0) return {0.0, 0.0};
    return {std::max(hull_volume(points_of(t), max_points) - vol, 0.0), vol};
}

template <size_t N>
std::vector<Cut> cuts(const Chain& piece, double total, int64_t max_points, const double (&positions)[N],
                      bool vertex_cuts) {
    Vec3 lo{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
    Vec3 hi = -lo;
    for (const Tri& t : piece)
        for (const Vec3& v : {t.a, t.b, t.c}) {
            lo = {std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
            hi = {std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
        }
    const double ext = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
    std::vector<Cut> out;
    for (int axis = 0; axis < 3; ++axis) {
        if (hi[axis] - lo[axis] <= 1e-9 * std::max(ext, 1e-12)) continue;
        Vec3 n{0, 0, 0};
        (axis == 0 ? n.x : axis == 1 ? n.y : n.z) = 1.0;
        std::vector<double> planes;
        for (double f : positions) planes.push_back(lo[axis] + f * (hi[axis] - lo[axis]));
        if (vertex_cuts) {                     // few distinct inner coordinates: also cut beside them
            std::vector<double> inner;
            for (const Tri& t : piece)
                for (const Vec3& v : {t.a, t.b, t.c})
                    if (v[axis] > lo[axis] && v[axis] < hi[axis]) inner.push_back(v[axis]);
            std::sort(inner.begin(), inner.end());
            inner.erase(std::unique(inner.begin(), inner.end()), inner.end());
            if (inner.size() <= kMaxVertexCuts) {
                // just beside each coordinate, never through it (faces in a cut plane would belong to both sides)
                const double delta = 1e-6 * (hi[axis] - lo[axis]);
                for (double x : inner) {
                    planes.push_back(x + -1.0 * delta);
                    planes.push_back(x + 1.0 * delta);
                }
            }
        }
        for (double d : planes) {
            if (std::min(d - lo[axis], hi[axis] - d) < 1e-4 * (hi[axis] - lo[axis])) continue;   // no slivers
            Chain left = clip(piece, n, d), right = clip(piece, -n, -d);
            const auto [cl, vl] = concavity(left, max_points);
            const auto [cr, vr] = concavity(right, max_points);
            if (vl > 1e-12 * total && vr > 1e-12 * total)
                out.push_back({cl + cr, std::move(left), std::move(right), cl, cr});
        }
    }
    return out;
}

double lookahead(const Cut& c, double total, int64_t max_points) {
    double cost = 0.0;
    for (int side = 0; side < 2; ++side) {
        const Chain& piece = side == 0 ? c.left : c.right;
        double best = side == 0 ? c.cl : c.cr;
        if (best > 0.0)
            for (const Cut& x : cuts(piece, total, max_points, kLookaheadPositions, false))
                best = std::min(best, x.cost);
        cost += best;
    }
    return cost;
}

size_t first_min(const std::vector<double>& costs, double tie) {
    const double m = *std::min_element(costs.begin(), costs.end());
    for (size_t i = 0; i < costs.size(); ++i)
        if (costs[i] <= m + tie) return i;
    return 0;
}

}  // namespace

double hull_volume(std::vector<Point3> pts, int64_t max_points) {
    std::vector<Vec3> p;
    if (max_points > 0 && static_cast<int64_t>(pts.size()) > max_points) {
        std::vector<int64_t> idx;
        for (const Vec3& d : fibonacci_directions(max_points)) {
            int64_t best = 0;
            double bv = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < pts.size(); ++i) {
                const double s = pts[i][0] * d.x + pts[i][1] * d.y + pts[i][2] * d.z;
                if (s > bv) bv = s, best = static_cast<int64_t>(i);
            }
            idx.push_back(best);
        }
        std::sort(idx.begin(), idx.end());
        idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
        for (int64_t i : idx) p.push_back(vec(pts[i]));
    } else {
        for (const Point3& q : pts) p.push_back(vec(q));
    }
    if (p.size() < 4) return 0.0;
    double size = 0.0;
    for (int k = 0; k < 3; ++k) {
        double lo = p[0][k], hi = p[0][k];
        for (const Vec3& q : p) lo = std::min(lo, q[k]), hi = std::max(hi, q[k]);
        size = std::max(size, hi - lo);
    }
    if (size == 0.0) size = 1.0;
    // merge points closer than 1e-7 of the size (as hull._dedupe: first point of each rounded cell, in order)
    const double tol = 1e-7 * size;
    std::vector<Vec3> q;
    {
        std::vector<std::pair<std::array<int64_t, 3>, size_t>> keys;
        for (size_t i = 0; i < p.size(); ++i)
            keys.push_back({{static_cast<int64_t>(std::nearbyint(p[i].x / tol)),
                             static_cast<int64_t>(std::nearbyint(p[i].y / tol)),
                             static_cast<int64_t>(std::nearbyint(p[i].z / tol))}, i});
        std::stable_sort(keys.begin(), keys.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        std::vector<size_t> first;
        for (size_t k = 0; k < keys.size(); ++k)
            if (k == 0 || keys[k].first != keys[k - 1].first) first.push_back(keys[k].second);
        std::sort(first.begin(), first.end());
        for (size_t i : first) q.push_back(p[i]);
    }
    return exact_hull_volume(q, size);
}

std::vector<std::vector<Point3>> convex_decomposition(const double* vertices, int64_t n_vertices,
                                                      const int64_t* triangles, int64_t n_triangles, int64_t max_parts,
                                                      double tolerance, int64_t max_hull_points) {
    Chain tris;
    tris.reserve(static_cast<size_t>(n_triangles));
    for (int64_t t = 0; t < n_triangles; ++t) {
        Vec3 v[3];
        for (int k = 0; k < 3; ++k) {
            const int64_t i = triangles[3 * t + k];
            if (i < 0 || i >= n_vertices) throw std::invalid_argument("triangle index out of range");
            v[k] = {vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2]};
        }
        tris.push_back({v[0], v[1], v[2]});
    }
    const double total = chain_volume(tris);
    if (!(total > 0.0)) throw std::invalid_argument("the mesh is not a closed solid with outward normals");
    const double tie = 1e-9 * total;
    std::vector<Chain> pieces{tris};
    std::vector<double> conc{concavity(tris, max_hull_points).first};
    auto sum = [&] {
        double s = 0.0;
        for (double c : conc) s += c;
        return s;
    };
    while (static_cast<int64_t>(pieces.size()) < max_parts && sum() > tolerance * total) {
        const double mx = *std::max_element(conc.begin(), conc.end());
        size_t k = 0;
        while (conc[k] < mx - tie) ++k;
        std::vector<Cut> cs = cuts(pieces[k], total, max_hull_points, kPositions, true);
        if (cs.empty()) {
            conc[k] = 0.0;
            continue;
        }
        std::vector<double> costs;
        for (const Cut& c : cs) costs.push_back(c.cost);
        size_t b = first_min(costs, tie);
        if (cs[b].cost > kImprovement * conc[k]) {
            costs.clear();
            for (const Cut& c : cs) costs.push_back(lookahead(c, total, max_hull_points));
            b = first_min(costs, tie);
        }
        Cut& best = cs[b];
        pieces[k] = std::move(best.left);
        pieces.insert(pieces.begin() + static_cast<std::ptrdiff_t>(k) + 1, std::move(best.right));
        conc[k] = best.cl;
        conc.insert(conc.begin() + static_cast<std::ptrdiff_t>(k) + 1, best.cr);
    }
    std::vector<std::vector<Point3>> out;
    for (const Chain& p : pieces) out.push_back(points_of(p));
    return out;
}

}  // namespace psim
