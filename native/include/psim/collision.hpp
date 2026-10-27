// Collision detection and contact constraint rows for one simulation step.
// Native counterpart of psim/physics/contact_kernel.py + collision.py (same pair order, contacts and rows).
#pragma once

#include <cstdint>
#include <vector>

namespace psim {

enum ShapeKind : int64_t { kSphere = 0, kBox = 1, kCapsule = 2, kPlane = 3, kHull = 4 };

// Convex hulls, flattened (see psim/physics/hull.py pack_hulls). Offsets have one more entry than items.
struct HullSet {
    int64_t n = 0;
    const double* verts = nullptr;        // (V, 3) relative to the collider centre
    const int64_t* vert_off = nullptr;
    const double* normals = nullptr;      // (F, 3) outward
    const double* offsets = nullptr;      // (F,)  n·x <= d inside
    const int64_t* face_off = nullptr;
    const int64_t* face_verts = nullptr;  // vertex indices (local to the hull), counter-clockwise
    const int64_t* face_vert_off = nullptr;   // per global face index
    const int64_t* edges = nullptr;       // (E, 2) local vertex indices
    const int64_t* edge_faces = nullptr;  // (E, 2) local face indices (face with a→b, face with b→a)
    const int64_t* edge_off = nullptr;
};

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
    const double* static_friction = nullptr;   // optional: static coefficients (values below `friction` are raised)
    const double* rolling = nullptr;           // optional: rolling resistance lengths (m); a pair uses the larger
    const double* sweep = nullptr;        // optional: how far each shape's surface can move this step (CCD)
    const double* center0 = nullptr;      // optional: start-of-step centres and rotations (speculative pairs)
    const double* rot0 = nullptr;
    const int64_t* hull_id = nullptr;     // index into hulls for kind kHull (and boxes paired with hulls), or -1
    const HullSet* hulls = nullptr;
};

struct BodySet {
    int64_t n = 0;
    const uint8_t* dynamic = nullptr;
    const double* com = nullptr;          // (n, 3)
    const double* v = nullptr;
    const double* w = nullptr;
    const double* com0 = nullptr;         // optional: start-of-step centres of mass (speculative pairs)
    const double* v0 = nullptr;           // optional: start-of-step velocities (sticking or sliding contacts)
    const double* w0 = nullptr;
};

// Contacts whose tangential speed at the start of the step is below this stick (static friction); faster ones slide.
constexpr double kStaticSpeed = 0.01;

struct ContactOutput {
    std::vector<int64_t> pairs;           // 2 per touching shape pair
    std::vector<int64_t> shape_a, shape_b, row;
    std::vector<double> point, normal, depth;
    std::vector<int64_t> ia, ib, parent, mode;
    std::vector<double> J, rhs, lo, hi, mu, err;
};

// Candidate pairs (i < j, lexicographic order) whose bounding spheres overlap; planes pair with all non-planes.
// Uses a sort-and-sweep along the axis of largest spread, then sorts the result, so the order equals the all-pairs loop.
std::vector<std::pair<int64_t, int64_t>> candidate_pairs(const ShapeSet& s, double margin);

// Broadphase + narrowphase + rows (see contact_kernel.py for the layout). `excluded` holds body pairs.
// With s.sweep and dt > 0, pairs that could tunnel get speculative contacts (see contact_kernel.py).
void contact_rows(const ShapeSet& s, const BodySet& b, const int64_t* excluded, int64_t n_excluded, double margin,
                  double restitution_threshold, ContactOutput& out, double dt = 0.0);

}  // namespace psim
