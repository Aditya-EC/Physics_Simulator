// Standalone C++ tests for the numerical core (run through CTest).
#include <cmath>
#include <cstdio>
#include <vector>

#include "psim/constraints.hpp"
#include "psim/integrators.hpp"
#include "psim/mesh_ops.hpp"
#include "psim/transforms.hpp"

using namespace psim;

static int g_failures = 0;

#define CHECK_NEAR(a, b, tol)                                                                          \
    do {                                                                                               \
        const double _a = (a), _b = (b);                                                               \
        if (std::fabs(_a - _b) > (tol)) {                                                              \
            std::printf("FAIL %s:%d  %s = %.12g, expected %.12g\n", __FILE__, __LINE__, #a, _a, _b); \
            ++g_failures;                                                                              \
        }                                                                                              \
    } while (0)

// Unit cube [0,1]^3 with outward-facing triangles.
static void unit_cube(std::vector<float>& v, std::vector<uint32_t>& f) {
    v = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    f = {0, 2, 1, 0, 3, 2,  // bottom (-z)
         4, 5, 6, 4, 6, 7,  // top (+z)
         0, 1, 5, 0, 5, 4,  // front (-y)
         2, 3, 7, 2, 7, 6,  // back (+y)
         1, 2, 6, 1, 6, 5,  // right (+x)
         3, 0, 4, 3, 4, 7}; // left (-x)
}

static void test_euler_roundtrip() {
    const double angles[][3] = {{0.1, -0.4, 2.0}, {-1.2, 0.3, -2.9}, {0.0, 0.0, 0.0}, {3.0, 1.2, 0.5}};
    for (const auto& a : angles) {
        const Quat q = quat_from_euler_xyz(a[0], a[1], a[2]);
        const Vec3 e = quat_to_euler_xyz(q);
        const Quat q2 = quat_from_euler_xyz(e.x, e.y, e.z);
        CHECK_NEAR(std::fabs(q.w * q2.w + q.x * q2.x + q.y * q2.y + q.z * q2.z), 1.0, 1e-12);
    }
}

static void test_pose_inverse() {
    const Pose p{{1, 2, 3}, quat_from_euler_xyz(0.3, -0.2, 1.1)};
    const Pose id = compose(p, inverse(p));
    CHECK_NEAR(norm(id.position), 0.0, 1e-12);
    CHECK_NEAR(id.orientation.w, 1.0, 1e-12);
}

static void test_rotation() {
    const Quat q = quat_from_axis_angle({0, 0, 1}, kPi / 2);
    const Vec3 r = rotate(q, {1, 0, 0});
    CHECK_NEAR(r.x, 0.0, 1e-12);
    CHECK_NEAR(r.y, 1.0, 1e-12);
}

static void test_mass_cube() {
    std::vector<float> v;
    std::vector<uint32_t> f;
    unit_cube(v, f);
    const MassProperties mp = mass_properties_solid(v.data(), f.data(), 12, 2.0);
    CHECK_NEAR(mp.measure, 1.0, 1e-9);
    CHECK_NEAR(mp.mass, 2.0, 1e-9);
    CHECK_NEAR(mp.centroid.x, 0.5, 1e-9);
    CHECK_NEAR(mp.centroid.z, 0.5, 1e-9);
    CHECK_NEAR(mp.inertia[0][0], 2.0 * 2.0 / 12.0, 1e-9);  // m (a^2 + b^2) / 12
    CHECK_NEAR(mp.inertia[0][1], 0.0, 1e-9);
}

static void test_ray() {
    std::vector<float> v;
    std::vector<uint32_t> f;
    unit_cube(v, f);
    const RayHit h = ray_mesh_intersect({0.5, 0.5, 5.0}, {0, 0, -1}, v.data(), 8, f.data(), 12);
    CHECK_NEAR(h.hit ? 1.0 : 0.0, 1.0, 0.0);
    CHECK_NEAR(h.t, 4.0, 1e-9);
    const RayHit miss = ray_mesh_intersect({5, 5, 5}, {0, 0, -1}, v.data(), 8, f.data(), 12);
    CHECK_NEAR(miss.hit ? 1.0 : 0.0, 0.0, 0.0);
}

static void test_feature_edges() {
    std::vector<float> v;
    std::vector<uint32_t> f;
    unit_cube(v, f);
    const std::vector<float> e = feature_edges(v.data(), 8, f.data(), 12, 30.0, 1e-6);
    CHECK_NEAR(static_cast<double>(e.size()) / 6.0, 12.0, 0.0);  // 12 cube edges, diagonals excluded
}

static void test_integrators() {
    // free fall from rest, semi-implicit Euler: v_n = n g dt, x_n = g dt^2 n(n+1)/2
    double x[3] = {0, 0, 0}, v[3] = {0, 0, 0};
    const double a[3] = {0, 0, -9.81};
    for (int i = 0; i < 100; ++i) integrate_semi_implicit_euler(x, v, a, 1, 0.01);
    CHECK_NEAR(v[2], -9.81, 1e-12);
    CHECK_NEAR(x[2], -9.81 * 1e-4 * 100 * 101 / 2.0, 1e-12);
    double x2[3] = {0, 0, 0}, v2[3] = {0, 0, 0};
    for (int i = 0; i < 100; ++i) integrate_explicit_euler(x2, v2, a, 1, 0.01);
    CHECK_NEAR(x2[2], -9.81 * 1e-4 * 100 * 99 / 2.0, 1e-12);
    // constant spin about Z for a quarter turn
    double q[4] = {1, 0, 0, 0};
    const double w[3] = {0, 0, kPi / 2};
    for (int i = 0; i < 1000; ++i) quat_integrate(q, w, 1, 0.001);
    CHECK_NEAR(q[0], std::cos(kPi / 4), 1e-12);
    CHECK_NEAR(q[3], std::sin(kPi / 4), 1e-12);
}

static void test_pgs_contact() {
    // a 1 kg body falling at 2 m/s onto the fixed world (row: normal +z, lambda >= 0, target velocity 0)
    const int64_t ia[1] = {-1}, ib[1] = {0}, parent[1] = {-1};
    const double J[12] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
    const double rhs[1] = {0.0}, lo[1] = {0.0}, hi[1] = {1e30}, mu[1] = {0.0};
    const int64_t block[1] = {1};
    const double inv_m[1] = {1.0}, inv_i[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double v[3] = {0.5, 0.0, -2.0}, w[3] = {0, 0, 0}, lambda[1];
    pgs_solve(1, ia, ib, J, rhs, lo, hi, parent, mu, block, 1, inv_m, inv_i, v, w, 10, lambda);
    CHECK_NEAR(v[2], 0.0, 1e-12);
    CHECK_NEAR(v[0], 0.5, 1e-12);
    CHECK_NEAR(lambda[0], 2.0, 1e-12);
    // separating bodies are not pulled back
    double v2[3] = {0.0, 0.0, 1.0};
    pgs_solve(1, ia, ib, J, rhs, lo, hi, parent, mu, block, 1, inv_m, inv_i, v2, w, 10, lambda);
    CHECK_NEAR(v2[2], 1.0, 1e-12);
    CHECK_NEAR(lambda[0], 0.0, 1e-12);
}

int main() {
    test_pgs_contact();
    test_integrators();
    test_euler_roundtrip();
    test_pose_inverse();
    test_rotation();
    test_mass_cube();
    test_ray();
    test_feature_edges();
    if (g_failures == 0) std::printf("All native tests passed.\n");
    return g_failures == 0 ? 0 : 1;
}
