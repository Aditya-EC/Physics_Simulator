// Rigid poses, TRS matrices and camera projection matrices.
#pragma once

#include "psim/vecmath.hpp"

namespace psim {

struct Pose {
    Vec3 position;
    Quat orientation;
};

// child expressed in parent frame -> child expressed in parent's parent frame
inline Pose compose(const Pose& parent, const Pose& child) {
    return {parent.position + rotate(parent.orientation, child.position),
            normalized(parent.orientation * child.orientation)};
}

inline Pose inverse(const Pose& p) {
    const Quat qi = conjugate(p.orientation);
    return {-rotate(qi, p.position), qi};
}

inline Mat4 trs_matrix(const Vec3& t, const Quat& q, const Vec3& s) {
    const Mat3 r = quat_to_mat3(normalized(q));
    Mat4 m = mat4_identity();
    for (int i = 0; i < 3; ++i) {
        m[i][0] = r[i][0] * s.x;
        m[i][1] = r[i][1] * s.y;
        m[i][2] = r[i][2] * s.z;
    }
    m[0][3] = t.x;
    m[1][3] = t.y;
    m[2][3] = t.z;
    return m;
}

inline Mat4 look_at(const Vec3& eye, const Vec3& target, const Vec3& up) {
    const Vec3 f = normalized(target - eye);
    Vec3 s = cross(f, up);
    if (norm(s) < 1e-12) s = cross(f, Vec3{0, 1, 0});
    s = normalized(s);
    const Vec3 u = cross(s, f);
    Mat4 m = mat4_identity();
    m[0] = {s.x, s.y, s.z, -dot(s, eye)};
    m[1] = {u.x, u.y, u.z, -dot(u, eye)};
    m[2] = {-f.x, -f.y, -f.z, dot(f, eye)};
    return m;
}

inline Mat4 perspective(double fovy, double aspect, double znear, double zfar) {
    const double f = 1.0 / std::tan(0.5 * fovy);
    Mat4 m{};
    m[0][0] = f / aspect;
    m[1][1] = f;
    m[2][2] = (zfar + znear) / (znear - zfar);
    m[2][3] = 2.0 * zfar * znear / (znear - zfar);
    m[3][2] = -1.0;
    return m;
}

inline Mat4 orthographic(double l, double r, double b, double t, double n, double f) {
    Mat4 m = mat4_identity();
    m[0][0] = 2.0 / (r - l);
    m[1][1] = 2.0 / (t - b);
    m[2][2] = -2.0 / (f - n);
    m[0][3] = -(r + l) / (r - l);
    m[1][3] = -(t + b) / (t - b);
    m[2][3] = -(f + n) / (f - n);
    return m;
}

}  // namespace psim
