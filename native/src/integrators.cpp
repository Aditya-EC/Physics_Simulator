#include "psim/integrators.hpp"

#include <cmath>

#include "psim/vecmath.hpp"

namespace psim {

void integrate_explicit_euler(double* x, double* v, const double* a, int64_t n, double dt) {
    for (int64_t i = 0; i < 3 * n; ++i) {
        x[i] += v[i] * dt;
        v[i] += a[i] * dt;
    }
}

void integrate_semi_implicit_euler(double* x, double* v, const double* a, int64_t n, double dt) {
    for (int64_t i = 0; i < 3 * n; ++i) {
        v[i] += a[i] * dt;
        x[i] += v[i] * dt;
    }
}

void quat_integrate(double* q, const double* w, int64_t n, double dt) {
    for (int64_t i = 0; i < n; ++i) {
        const Vec3 omega{w[3 * i], w[3 * i + 1], w[3 * i + 2]};
        const double speed = norm(omega);
        Quat cur{q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]};
        if (speed * dt > 0.0) {
            // exact exponential map for constant angular velocity over the step
            const Quat dq = quat_from_axis_angle(omega / speed, speed * dt);
            cur = dq * cur;
        }
        const double len = std::sqrt(cur.w * cur.w + cur.x * cur.x + cur.y * cur.y + cur.z * cur.z);
        q[4 * i] = cur.w / len;
        q[4 * i + 1] = cur.x / len;
        q[4 * i + 2] = cur.y / len;
        q[4 * i + 3] = cur.z / len;
    }
}

void quat_rotate_batch(const double* q, const double* v, int64_t n, double* out) {
    for (int64_t i = 0; i < n; ++i) {
        const Quat qi = normalized(Quat{q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]});
        const Vec3 r = rotate(qi, {v[3 * i], v[3 * i + 1], v[3 * i + 2]});
        out[3 * i] = r.x;
        out[3 * i + 1] = r.y;
        out[3 * i + 2] = r.z;
    }
}

void rotate_tensors(const double* q, const double* m, int64_t n, double* out) {
    for (int64_t i = 0; i < n; ++i) {
        const Mat3 r = quat_to_mat3(normalized(Quat{q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]}));
        const double* mi = m + 9 * i;
        double tmp[3][3];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                double acc = 0.0;
                for (int k = 0; k < 3; ++k) acc += r[a][k] * mi[3 * k + b];
                tmp[a][b] = acc;
            }
        double* oi = out + 9 * i;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                double acc = 0.0;
                for (int k = 0; k < 3; ++k) acc += tmp[a][k] * r[b][k];
                oi[3 * a + b] = acc;
            }
    }
}

}  // namespace psim
