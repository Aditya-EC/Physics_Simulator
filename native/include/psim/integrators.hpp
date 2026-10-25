// Batched time-integration kernels operating in place on contiguous (N, 3) / (N, 4) arrays.
#pragma once

#include <cstdint>

namespace psim {

// x += v dt (using v BEFORE the update), then v += a dt
void integrate_explicit_euler(double* x, double* v, const double* a, int64_t n, double dt);

// v += a dt, then x += v dt (symplectic / semi-implicit Euler)
void integrate_semi_implicit_euler(double* x, double* v, const double* a, int64_t n, double dt);

// q <- exp(0.5 * w * dt) (x) q for world-frame angular velocities w; result normalised.
void quat_integrate(double* q, const double* w, int64_t n, double dt);

// out[i] = R(q[i]) v[i]
void quat_rotate_batch(const double* q, const double* v, int64_t n, double* out);

// out[i] = R(q[i]) M[i] R(q[i])^T  (e.g. inertia or inverse inertia to the world frame)
void rotate_tensors(const double* q, const double* m, int64_t n, double* out);

}  // namespace psim
