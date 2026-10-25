// Core vector / quaternion / matrix types for the PSim numerical core.
//
// Conventions (shared with the Python layer):
//   * Right-handed coordinate system, Z axis up.
//   * Quaternions are stored as (w, x, y, z) and are unit length.
//   * Matrices are row-major and act on column vectors: p' = M * p.
//   * Euler angles are extrinsic X, then Y, then Z (R = Rz * Ry * Rx), radians.
#pragma once

#include <array>
#include <cmath>

namespace psim {

constexpr double kPi = 3.14159265358979323846;

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }

    double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalized(const Vec3& a) {
    const double n = norm(a);
    return n > 0.0 ? a / n : Vec3{0.0, 0.0, 0.0};
}

using Mat3 = std::array<std::array<double, 3>, 3>;
using Mat4 = std::array<std::array<double, 4>, 4>;

inline Mat3 mat3_identity() { return {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}; }
inline Mat4 mat4_identity() { return {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}}; }

inline Vec3 mul(const Mat3& m, const Vec3& v) {
    return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
}

struct Quat {
    double w = 1.0, x = 0.0, y = 0.0, z = 0.0;

    Quat() = default;
    Quat(double w_, double x_, double y_, double z_) : w(w_), x(x_), y(y_), z(z_) {}
};

inline Quat operator*(const Quat& a, const Quat& b) {
    return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}

inline Quat conjugate(const Quat& q) { return {q.w, -q.x, -q.y, -q.z}; }

inline Quat normalized(const Quat& q) {
    const double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (n <= 0.0) return {1.0, 0.0, 0.0, 0.0};
    Quat r{q.w / n, q.x / n, q.y / n, q.z / n};
    // Canonical hemisphere keeps serialized values stable.
    if (r.w < 0.0) r = {-r.w, -r.x, -r.y, -r.z};
    return r;
}

inline Quat quat_from_axis_angle(const Vec3& axis, double angle) {
    const Vec3 a = normalized(axis);
    const double h = 0.5 * angle;
    const double s = std::sin(h);
    return {std::cos(h), a.x * s, a.y * s, a.z * s};
}

inline Vec3 rotate(const Quat& q, const Vec3& v) {
    // v' = v + 2w(u x v) + 2 u x (u x v)
    const Vec3 u{q.x, q.y, q.z};
    const Vec3 t = cross(u, v) * 2.0;
    return v + t * q.w + cross(u, t);
}

inline Mat3 quat_to_mat3(const Quat& q) {
    const double w = q.w, x = q.x, y = q.y, z = q.z;
    return {{{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
             {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
             {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}}};
}

inline Quat mat3_to_quat(const Mat3& m) {
    const double tr = m[0][0] + m[1][1] + m[2][2];
    Quat q;
    if (tr > 0.0) {
        const double s = std::sqrt(tr + 1.0) * 2.0;
        q = {0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s};
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        const double s = std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
        q = {(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s};
    } else if (m[1][1] > m[2][2]) {
        const double s = std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
        q = {(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s};
    } else {
        const double s = std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
        q = {(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s};
    }
    return normalized(q);
}

// Extrinsic XYZ (R = Rz * Ry * Rx).
inline Quat quat_from_euler_xyz(double rx, double ry, double rz) {
    const Quat qx = quat_from_axis_angle({1, 0, 0}, rx);
    const Quat qy = quat_from_axis_angle({0, 1, 0}, ry);
    const Quat qz = quat_from_axis_angle({0, 0, 1}, rz);
    return normalized(qz * (qy * qx));
}

inline Vec3 quat_to_euler_xyz(const Quat& q) {
    const Mat3 m = quat_to_mat3(normalized(q));
    // m = Rz * Ry * Rx  =>  m[2][0] = -sin(ry)
    const double sy = -m[2][0];
    double rx, ry, rz;
    if (std::fabs(sy) < 0.9999999) {
        ry = std::asin(sy);
        rx = std::atan2(m[2][1], m[2][2]);
        rz = std::atan2(m[1][0], m[0][0]);
    } else {
        // Gimbal lock: choose rx = 0 and fold the remaining rotation into rz.
        ry = sy > 0 ? kPi / 2 : -kPi / 2;
        rx = 0.0;
        rz = std::atan2(-m[0][1], m[1][1]);
    }
    return {rx, ry, rz};
}

}  // namespace psim
