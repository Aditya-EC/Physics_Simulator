// Mesh-level numerical routines: picking, normals, feature edges, mass properties.
#pragma once

#include <cstdint>
#include <vector>

#include "psim/vecmath.hpp"

namespace psim {

struct RayHit {
    bool hit = false;
    double t = 0.0;
    int64_t triangle = -1;
};

// Closest intersection of a ray with an indexed triangle mesh (both faces count).
RayHit ray_mesh_intersect(const Vec3& origin, const Vec3& direction, const float* vertices,
                          int64_t vertex_count, const uint32_t* indices, int64_t triangle_count);

// Area-weighted per-vertex normals.
void compute_vertex_normals(const float* vertices, int64_t vertex_count, const uint32_t* indices,
                            int64_t triangle_count, float* out_normals);

// Returns line segments (pairs of points, flattened xyz) of boundary edges and of
// edges whose adjacent face normals differ by more than `angle_deg`.
// Coincident vertices are welded (within `weld_tolerance`) before analysis so
// meshes split for flat shading are handled correctly.
std::vector<float> feature_edges(const float* vertices, int64_t vertex_count, const uint32_t* indices,
                                 int64_t triangle_count, double angle_deg, double weld_tolerance);

struct MassProperties {
    double measure = 0.0;  // volume (solid) or area (shell)
    double mass = 0.0;
    Vec3 centroid;
    Mat3 inertia{};  // about the centroid, in the mesh frame
};

// Solid of uniform density bounded by a closed, outward-oriented triangle mesh
// (Eberly, "Polyhedral Mass Properties (Revisited)").
MassProperties mass_properties_solid(const float* vertices, const uint32_t* indices, int64_t triangle_count,
                                     double density);

// Thin shell of uniform areal density over the triangles of the mesh.
MassProperties mass_properties_shell(const float* vertices, const uint32_t* indices, int64_t triangle_count,
                                     double areal_density);

}  // namespace psim
