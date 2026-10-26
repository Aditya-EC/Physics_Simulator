// Projected Gauss–Seidel (sequential impulse) solver for generic velocity constraint rows.
#pragma once

#include <cstdint>

namespace psim {

// Solves rows  J_r · [v_a, w_a, v_b, w_b] = rhs_r  for accumulated impulses lambda_r with
// lo_r <= lambda_r <= hi_r, or |lambda_r| <= mu_r * lambda_parent for friction rows (parent_r >= 0).
//   ia, ib   (m)      body indices (-1 = the fixed world)
//   J        (m, 12)  [lin_a, ang_a, lin_b, ang_b]
//   inv_m    (n)      inverse masses, inv_i (n, 3, 3) world inverse inertia tensors
//   v, w     (n, 3)   velocities, updated in place; lambda (m) receives the accumulated impulses
//   block    (m)      1 for a scalar row; k = 2…6 at the first row of k consecutive *unbounded* rows on the
//                     same bodies that are solved together with their k×k effective-mass matrix (0 for the
//                     other rows of the block). Blocks converge fast even for very different masses.
//   B        (m, 12)  optional impulse directions (nullptr = J): impulses are applied as M^-1 B^T λ while the
//                     constraint J v = rhs is enforced, with effective mass J M^-1 B^T. Joints pass the Jacobian
//                     of the midpoint configuration, which turns velocities instead of projecting them and so
//                     avoids the O(ω² Δt) energy loss of a plain projection.
// Rows are processed in order, `iterations` times. Rows whose effective mass is zero are skipped.
void pgs_solve(int64_t m, const int64_t* ia, const int64_t* ib, const double* J, const double* rhs,
               const double* lo, const double* hi, const int64_t* parent, const double* mu, const int64_t* block,
               int64_t n, const double* inv_m, const double* inv_i, double* v, double* w, int iterations,
               double* lambda, const double* B = nullptr);

}  // namespace psim
