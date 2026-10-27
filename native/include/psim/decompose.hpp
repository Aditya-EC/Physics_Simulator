// Approximate convex decomposition of a closed triangle mesh by exact plane cuts.
// Native counterpart of psim/physics/decompose.py (same algorithm, same tie rules).
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace psim {

using Point3 = std::array<double, 3>;

// Volume of the convex hull of `pts` (0 for fewer than 4 or coplanar points). With max_points > 0, larger point
// sets are first reduced to their support points in max_points Fibonacci directions.
double hull_volume(std::vector<Point3> pts, int64_t max_points = 0);

// Point sets of the convex parts (each sorted lexicographically, without duplicates). Throws
// std::invalid_argument when the mesh does not enclose a positive volume.
std::vector<std::vector<Point3>> convex_decomposition(const double* vertices, int64_t n_vertices,
                                                      const int64_t* triangles, int64_t n_triangles, int64_t max_parts,
                                                      double tolerance, int64_t max_hull_points);

}  // namespace psim
