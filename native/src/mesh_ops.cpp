#include "psim/mesh_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace psim {
namespace {

inline Vec3 vtx(const float* v, uint32_t i) { return {v[3 * i], v[3 * i + 1], v[3 * i + 2]}; }

struct EdgeKey {
    int64_t a, b;
    bool operator==(const EdgeKey& o) const { return a == o.a && b == o.b; }
};
struct EdgeKeyHash {
    size_t operator()(const EdgeKey& k) const {
        return std::hash<int64_t>()(k.a * 73856093LL) ^ std::hash<int64_t>()(k.b * 19349663LL);
    }
};

struct CellKey {
    int64_t x, y, z;
    bool operator==(const CellKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct CellKeyHash {
    size_t operator()(const CellKey& k) const {
        return static_cast<size_t>(k.x * 73856093LL ^ k.y * 19349663LL ^ k.z * 83492791LL);
    }
};

}  // namespace

RayHit ray_mesh_intersect(const Vec3& origin, const Vec3& direction, const float* vertices,
                          int64_t /*vertex_count*/, const uint32_t* indices, int64_t triangle_count) {
    constexpr double eps = 1e-12;
    RayHit best;
    best.t = std::numeric_limits<double>::infinity();
    for (int64_t t = 0; t < triangle_count; ++t) {
        const Vec3 p0 = vtx(vertices, indices[3 * t]);
        const Vec3 p1 = vtx(vertices, indices[3 * t + 1]);
        const Vec3 p2 = vtx(vertices, indices[3 * t + 2]);
        const Vec3 e1 = p1 - p0;
        const Vec3 e2 = p2 - p0;
        const Vec3 p = cross(direction, e2);
        const double det = dot(e1, p);
        if (std::fabs(det) < eps) continue;
        const double inv = 1.0 / det;
        const Vec3 s = origin - p0;
        const double u = dot(s, p) * inv;
        if (u < 0.0 || u > 1.0) continue;
        const Vec3 q = cross(s, e1);
        const double v = dot(direction, q) * inv;
        if (v < 0.0 || u + v > 1.0) continue;
        const double dist = dot(e2, q) * inv;
        if (dist > 1e-9 && dist < best.t) {
            best.hit = true;
            best.t = dist;
            best.triangle = t;
        }
    }
    if (!best.hit) best.t = -1.0;
    return best;
}

void compute_vertex_normals(const float* vertices, int64_t vertex_count, const uint32_t* indices,
                            int64_t triangle_count, float* out_normals) {
    std::vector<Vec3> acc(static_cast<size_t>(vertex_count));
    for (int64_t t = 0; t < triangle_count; ++t) {
        const uint32_t i0 = indices[3 * t], i1 = indices[3 * t + 1], i2 = indices[3 * t + 2];
        const Vec3 n = cross(vtx(vertices, i1) - vtx(vertices, i0), vtx(vertices, i2) - vtx(vertices, i0));
        acc[i0] += n;
        acc[i1] += n;
        acc[i2] += n;
    }
    for (int64_t i = 0; i < vertex_count; ++i) {
        const Vec3 n = normalized(acc[static_cast<size_t>(i)]);
        out_normals[3 * i] = static_cast<float>(n.x);
        out_normals[3 * i + 1] = static_cast<float>(n.y);
        out_normals[3 * i + 2] = static_cast<float>(n.z);
    }
}

std::vector<float> feature_edges(const float* vertices, int64_t vertex_count, const uint32_t* indices,
                                 int64_t triangle_count, double angle_deg, double weld_tolerance) {
    // Weld coincident vertices with a spatial hash.
    const double tol = weld_tolerance > 0.0 ? weld_tolerance : 1e-6;
    std::unordered_map<CellKey, int64_t, CellKeyHash> cells;
    std::vector<int64_t> canonical(static_cast<size_t>(vertex_count));
    for (int64_t i = 0; i < vertex_count; ++i) {
        const CellKey key{static_cast<int64_t>(std::llround(vertices[3 * i] / tol)),
                          static_cast<int64_t>(std::llround(vertices[3 * i + 1] / tol)),
                          static_cast<int64_t>(std::llround(vertices[3 * i + 2] / tol))};
        auto it = cells.find(key);
        if (it == cells.end()) {
            cells.emplace(key, i);
            canonical[static_cast<size_t>(i)] = i;
        } else {
            canonical[static_cast<size_t>(i)] = it->second;
        }
    }

    std::vector<Vec3> face_normals(static_cast<size_t>(triangle_count));
    std::unordered_map<EdgeKey, std::vector<int64_t>, EdgeKeyHash> edges;
    for (int64_t t = 0; t < triangle_count; ++t) {
        int64_t c[3];
        for (int k = 0; k < 3; ++k) c[k] = canonical[indices[3 * t + k]];
        const Vec3 p0 = vtx(vertices, static_cast<uint32_t>(c[0]));
        const Vec3 p1 = vtx(vertices, static_cast<uint32_t>(c[1]));
        const Vec3 p2 = vtx(vertices, static_cast<uint32_t>(c[2]));
        const Vec3 n = cross(p1 - p0, p2 - p0);
        if (norm(n) < 1e-14) continue;  // degenerate (e.g. collapsed pole triangles)
        face_normals[static_cast<size_t>(t)] = normalized(n);
        for (int k = 0; k < 3; ++k) {
            int64_t a = c[k], b = c[(k + 1) % 3];
            if (a == b) continue;
            if (a > b) std::swap(a, b);
            edges[{a, b}].push_back(t);
        }
    }

    const double cos_limit = std::cos(angle_deg * kPi / 180.0);
    std::vector<float> out;
    for (const auto& kv : edges) {
        const auto& faces = kv.second;
        bool keep = false;
        if (faces.size() == 1) {
            keep = true;
        } else if (faces.size() == 2) {
            keep = dot(face_normals[static_cast<size_t>(faces[0])], face_normals[static_cast<size_t>(faces[1])]) <
                   cos_limit;
        }
        if (!keep) continue;
        for (int64_t idx : {kv.first.a, kv.first.b}) {
            out.push_back(vertices[3 * idx]);
            out.push_back(vertices[3 * idx + 1]);
            out.push_back(vertices[3 * idx + 2]);
        }
    }
    return out;
}

MassProperties mass_properties_solid(const float* vertices, const uint32_t* indices, int64_t triangle_count,
                                     double density) {
    const double mult[10] = {1.0 / 6, 1.0 / 24, 1.0 / 24, 1.0 / 24, 1.0 / 60,
                             1.0 / 60, 1.0 / 60, 1.0 / 120, 1.0 / 120, 1.0 / 120};
    double integral[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    auto subexpressions = [](double w0, double w1, double w2, double& f1, double& f2, double& f3, double& g0,
                             double& g1, double& g2) {
        const double temp0 = w0 + w1;
        f1 = temp0 + w2;
        const double temp1 = w0 * w0;
        const double temp2 = temp1 + w1 * temp0;
        f2 = temp2 + w2 * f1;
        f3 = w0 * temp1 + w1 * temp2 + w2 * f2;
        g0 = f2 + w0 * (f1 + w0);
        g1 = f2 + w1 * (f1 + w1);
        g2 = f2 + w2 * (f1 + w2);
    };

    for (int64_t t = 0; t < triangle_count; ++t) {
        const Vec3 p0 = vtx(vertices, indices[3 * t]);
        const Vec3 p1 = vtx(vertices, indices[3 * t + 1]);
        const Vec3 p2 = vtx(vertices, indices[3 * t + 2]);
        const Vec3 d = cross(p1 - p0, p2 - p0);

        double f1x, f2x, f3x, g0x, g1x, g2x;
        double f1y, f2y, f3y, g0y, g1y, g2y;
        double f1z, f2z, f3z, g0z, g1z, g2z;
        subexpressions(p0.x, p1.x, p2.x, f1x, f2x, f3x, g0x, g1x, g2x);
        subexpressions(p0.y, p1.y, p2.y, f1y, f2y, f3y, g0y, g1y, g2y);
        subexpressions(p0.z, p1.z, p2.z, f1z, f2z, f3z, g0z, g1z, g2z);

        integral[0] += d.x * f1x;
        integral[1] += d.x * f2x;
        integral[2] += d.y * f2y;
        integral[3] += d.z * f2z;
        integral[4] += d.x * f3x;
        integral[5] += d.y * f3y;
        integral[6] += d.z * f3z;
        integral[7] += d.x * (p0.y * g0x + p1.y * g1x + p2.y * g2x);
        integral[8] += d.y * (p0.z * g0y + p1.z * g1y + p2.z * g2y);
        integral[9] += d.z * (p0.x * g0z + p1.x * g1z + p2.x * g2z);
    }
    for (int i = 0; i < 10; ++i) integral[i] *= mult[i];

    MassProperties mp;
    const double volume = integral[0];
    mp.measure = volume;
    if (std::fabs(volume) < 1e-300) return mp;

    const Vec3 cm{integral[1] / volume, integral[2] / volume, integral[3] / volume};
    // Unit-density inertia about the centroid, then scale by density.
    const double ixx = integral[5] + integral[6] - volume * (cm.y * cm.y + cm.z * cm.z);
    const double iyy = integral[4] + integral[6] - volume * (cm.z * cm.z + cm.x * cm.x);
    const double izz = integral[4] + integral[5] - volume * (cm.x * cm.x + cm.y * cm.y);
    const double ixy = -(integral[7] - volume * cm.x * cm.y);
    const double iyz = -(integral[8] - volume * cm.y * cm.z);
    const double ixz = -(integral[9] - volume * cm.z * cm.x);

    mp.mass = volume * density;
    mp.centroid = cm;
    mp.inertia = {{{ixx * density, ixy * density, ixz * density},
                   {ixy * density, iyy * density, iyz * density},
                   {ixz * density, iyz * density, izz * density}}};
    return mp;
}

MassProperties mass_properties_shell(const float* vertices, const uint32_t* indices, int64_t triangle_count,
                                     double areal_density) {
    double area = 0.0;
    Vec3 first;
    double second[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int64_t t = 0; t < triangle_count; ++t) {
        const Vec3 p[3] = {vtx(vertices, indices[3 * t]), vtx(vertices, indices[3 * t + 1]),
                           vtx(vertices, indices[3 * t + 2])};
        const double a = 0.5 * norm(cross(p[1] - p[0], p[2] - p[0]));
        if (a <= 0.0) continue;
        area += a;
        const Vec3 s = p[0] + p[1] + p[2];
        first += s * (a / 3.0);
        // Integral of r r^T over a triangle = A/12 * (sum p_i p_i^T + s s^T)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double acc = s[i] * s[j];
                for (int k = 0; k < 3; ++k) acc += p[k][i] * p[k][j];
                second[i][j] += a / 12.0 * acc;
            }
    }
    MassProperties mp;
    mp.measure = area;
    if (area <= 0.0) return mp;
    const Vec3 cm = first / area;
    // Shift second moment to the centroid: C_c = C - A * cm cm^T
    double c[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) c[i][j] = second[i][j] - area * cm[i] * cm[j];
    const double tr = c[0][0] + c[1][1] + c[2][2];
    mp.mass = area * areal_density;
    mp.centroid = cm;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) mp.inertia[i][j] = areal_density * ((i == j ? tr : 0.0) - c[i][j]);
    return mp;
}

}  // namespace psim
