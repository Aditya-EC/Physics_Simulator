// Python bindings for the PSim numerical core.
//
// Mesh routines take float32 vertex and uint32 index arrays (the formats used
// by the renderer) so that no conversion copies are made when called from Python.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

#include "psim/mesh_ops.hpp"
#include "psim/transforms.hpp"

namespace py = pybind11;
using namespace psim;

namespace {

using DArr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using FArr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using UArr = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

Vec3 to_vec3(const DArr& a) {
    if (a.size() != 3) throw std::invalid_argument("expected a 3-vector");
    const double* p = a.data();
    return {p[0], p[1], p[2]};
}

Quat to_quat(const DArr& a) {
    if (a.size() != 4) throw std::invalid_argument("expected a quaternion (w, x, y, z)");
    const double* p = a.data();
    return {p[0], p[1], p[2], p[3]};
}

py::array_t<double> from_vec3(const Vec3& v) {
    py::array_t<double> out(3);
    auto r = out.mutable_unchecked<1>();
    r(0) = v.x;
    r(1) = v.y;
    r(2) = v.z;
    return out;
}

py::array_t<double> from_quat(const Quat& q) {
    py::array_t<double> out(4);
    auto r = out.mutable_unchecked<1>();
    r(0) = q.w;
    r(1) = q.x;
    r(2) = q.y;
    r(3) = q.z;
    return out;
}

py::array_t<double> from_mat3(const Mat3& m) {
    py::array_t<double> out({3, 3});
    auto r = out.mutable_unchecked<2>();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r(i, j) = m[i][j];
    return out;
}

py::array_t<double> from_mat4(const Mat4& m) {
    py::array_t<double> out({4, 4});
    auto r = out.mutable_unchecked<2>();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) r(i, j) = m[i][j];
    return out;
}

Mat3 to_mat3(const DArr& a) {
    if (a.ndim() != 2 || a.shape(0) != 3 || a.shape(1) != 3) throw std::invalid_argument("expected a 3x3 matrix");
    auto r = a.unchecked<2>();
    Mat3 m;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) m[i][j] = r(i, j);
    return m;
}

void check_mesh(const FArr& v, const UArr& f) {
    if (v.ndim() != 2 || v.shape(1) != 3) throw std::invalid_argument("vertices must have shape (N, 3)");
    if (f.ndim() != 2 || f.shape(1) != 3) throw std::invalid_argument("indices must have shape (M, 3)");
    const uint32_t* idx = f.data();
    const auto n = static_cast<uint32_t>(v.shape(0));
    for (py::ssize_t i = 0; i < f.size(); ++i)
        if (idx[i] >= n) throw std::out_of_range("triangle index out of range");
}

py::dict mass_dict(const MassProperties& mp) {
    py::dict d;
    d["measure"] = mp.measure;
    d["mass"] = mp.mass;
    d["centroid"] = from_vec3(mp.centroid);
    d["inertia"] = from_mat3(mp.inertia);
    return d;
}

}  // namespace

PYBIND11_MODULE(_psim_native, m) {
    m.doc() = "PSim native numerical core (C++)";
    m.attr("__version__") = "0.1.0";

    // ---- quaternions -------------------------------------------------------
    m.def("quat_normalize", [](const DArr& q) { return from_quat(normalized(to_quat(q))); });
    m.def("quat_multiply", [](const DArr& a, const DArr& b) { return from_quat(to_quat(a) * to_quat(b)); });
    m.def("quat_conjugate", [](const DArr& q) { return from_quat(conjugate(to_quat(q))); });
    m.def("quat_from_axis_angle",
          [](const DArr& axis, double angle) { return from_quat(normalized(quat_from_axis_angle(to_vec3(axis), angle))); });
    m.def("quat_from_euler", [](const DArr& e) {
        const Vec3 v = to_vec3(e);
        return from_quat(quat_from_euler_xyz(v.x, v.y, v.z));
    });
    m.def("quat_to_euler", [](const DArr& q) { return from_vec3(quat_to_euler_xyz(to_quat(q))); });
    m.def("quat_to_matrix", [](const DArr& q) { return from_mat3(quat_to_mat3(normalized(to_quat(q)))); });
    m.def("matrix_to_quat", [](const DArr& r) { return from_quat(mat3_to_quat(to_mat3(r))); });
    m.def("quat_rotate", [](const DArr& q, const DArr& v) { return from_vec3(rotate(normalized(to_quat(q)), to_vec3(v))); });

    // ---- poses and matrices -----------------------------------------------
    m.def("pose_compose", [](const DArr& p1, const DArr& q1, const DArr& p2, const DArr& q2) {
        const Pose r = compose({to_vec3(p1), normalized(to_quat(q1))}, {to_vec3(p2), normalized(to_quat(q2))});
        return py::make_tuple(from_vec3(r.position), from_quat(r.orientation));
    });
    m.def("pose_inverse", [](const DArr& p, const DArr& q) {
        const Pose r = inverse({to_vec3(p), normalized(to_quat(q))});
        return py::make_tuple(from_vec3(r.position), from_quat(r.orientation));
    });
    m.def("trs_matrix", [](const DArr& t, const DArr& q, const DArr& s) {
        return from_mat4(trs_matrix(to_vec3(t), to_quat(q), to_vec3(s)));
    });
    m.def("look_at", [](const DArr& eye, const DArr& target, const DArr& up) {
        return from_mat4(look_at(to_vec3(eye), to_vec3(target), to_vec3(up)));
    });
    m.def("perspective", [](double fovy, double aspect, double n, double f) {
        return from_mat4(perspective(fovy, aspect, n, f));
    });
    m.def("orthographic", [](double l, double r, double b, double t, double n, double f) {
        return from_mat4(orthographic(l, r, b, t, n, f));
    });
    m.def("transform_points", [](const DArr& mat, const DArr& pts) {
        if (mat.ndim() != 2 || mat.shape(0) != 4 || mat.shape(1) != 4) throw std::invalid_argument("expected 4x4 matrix");
        if (pts.ndim() != 2 || pts.shape(1) != 3) throw std::invalid_argument("points must have shape (N, 3)");
        auto M = mat.unchecked<2>();
        auto P = pts.unchecked<2>();
        const py::ssize_t n = pts.shape(0);
        py::array_t<double> out({n, static_cast<py::ssize_t>(3)});
        auto O = out.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < n; ++i) {
            const double x = P(i, 0), y = P(i, 1), z = P(i, 2);
            double w = M(3, 0) * x + M(3, 1) * y + M(3, 2) * z + M(3, 3);
            if (w == 0.0) w = 1.0;
            for (int r = 0; r < 3; ++r) O(i, r) = (M(r, 0) * x + M(r, 1) * y + M(r, 2) * z + M(r, 3)) / w;
        }
        return out;
    });

    // ---- mesh routines ----------------------------------------------------
    m.def("ray_mesh_intersect", [](const DArr& origin, const DArr& direction, const FArr& v, const UArr& f) {
        check_mesh(v, f);
        RayHit h;
        {
            py::gil_scoped_release release;
            h = ray_mesh_intersect(to_vec3(origin), to_vec3(direction), v.data(), v.shape(0), f.data(), f.shape(0));
        }
        return py::make_tuple(h.hit, h.t, h.triangle);
    }, "Returns (hit, t, triangle_index); t = -1 when there is no hit.");

    m.def("compute_vertex_normals", [](const FArr& v, const UArr& f) {
        check_mesh(v, f);
        py::array_t<float> out({v.shape(0), static_cast<py::ssize_t>(3)});
        compute_vertex_normals(v.data(), v.shape(0), f.data(), f.shape(0), out.mutable_data());
        return out;
    });

    m.def("feature_edges", [](const FArr& v, const UArr& f, double angle_deg, double weld_tolerance) {
        check_mesh(v, f);
        std::vector<float> e = feature_edges(v.data(), v.shape(0), f.data(), f.shape(0), angle_deg, weld_tolerance);
        const auto n = static_cast<py::ssize_t>(e.size() / 3);
        py::array_t<float> out({n, static_cast<py::ssize_t>(3)});
        std::copy(e.begin(), e.end(), out.mutable_data());
        return out;
    }, py::arg("vertices"), py::arg("indices"), py::arg("angle_deg") = 30.0, py::arg("weld_tolerance") = 1e-5);

    m.def("mass_properties_solid", [](const FArr& v, const UArr& f, double density) {
        check_mesh(v, f);
        return mass_dict(mass_properties_solid(v.data(), f.data(), f.shape(0), density));
    });
    m.def("mass_properties_shell", [](const FArr& v, const UArr& f, double areal_density) {
        check_mesh(v, f);
        return mass_dict(mass_properties_shell(v.data(), f.data(), f.shape(0), areal_density));
    });
}
