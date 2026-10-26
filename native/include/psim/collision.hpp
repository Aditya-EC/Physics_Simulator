// Collision detection and contact constraint rows for one simulation step.
// Native counterpart of psim/physics/contact_kernel.py + collision.py (same pair order, contacts and rows).
#pragma once

#include <cstdint>
#include <vector>

namespace psim {

enum ShapeKind : int64_t { kSphere = 0, kBox = 1, kCapsule = 2, kPlane = 3 };

struct ShapeSet {
    int64_t n = 0;
    const int64_t* kind = nullptr;
    const int64_t* body = nullptr;        // solver body index of each shape
    const double* center = nullptr;       // (n, 3) world
    const double* rot = nullptr;          // (n, 3, 3) row-major world rotation
    const double* radius = nullptr;
    const double* half = nullptr;         // (n, 3) box half extents
    const double* half_len = nullptr;     // capsule core half length (local Z)
    const double* restitution = nullptr;
    const double* friction = nullptr;
};

struct BodySet {
    int64_t n = 0;
    const uint8_t* dynamic = nullptr;
    const double* com = nullptr;          // (n, 3)
    const double* v = nullptr;
    const double* w = nullptr;
};

struct ContactOutput {
    std::vector<int64_t> pairs;           // 2 per touching shape pair
    std::vector<int64_t> shape_a, shape_b, row;
    std::vector<double> point, normal, depth;
    std::vector<int64_t> ia, ib, parent, mode;
    std::vector<double> J, rhs, lo, hi, mu, err;
};

// Candidate pairs (i < j, lexicographic order) whose bounding spheres overlap; planes pair with all non-planes.
// Uses a sort-and-sweep on the x extent, then sorts the result, so the order equals the all-pairs loop.
std::vector<std::pair<int64_t, int64_t>> candidate_pairs(const ShapeSet& s, double margin);

// Broadphase + narrowphase + rows (see contact_kernel.py for the layout). `excluded` holds body pairs.
void contact_rows(const ShapeSet& s, const BodySet& b, const int64_t* excluded, int64_t n_excluded, double margin,
                  double restitution_threshold, ContactOutput& out);

}  // namespace psim
