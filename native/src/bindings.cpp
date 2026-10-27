// Python bindings for the PSim numerical core.
//
// Mesh routines take float32 vertex and uint32 index arrays (the formats used
// by the renderer) so that no conversion copies are made when called from Python.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

#include "psim/collision.hpp"
#include "psim/constraints.hpp"
#include "psim/decompose.hpp"
#include "psim/integrators.hpp"
#include "psim/joints.hpp"
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

// In-place kernels: require writable float64 C-contiguous arrays (no silent copies).
using InPlace = py::array_t<double, py::array::c_style>;

double* writable(InPlace& a, py::ssize_t cols, const char* name) {
    if (a.ndim() != 2 || a.shape(1) != cols)
        throw std::invalid_argument(std::string(name) + " must have shape (N, " + std::to_string(cols) + ")");
    if (!a.writeable()) throw std::invalid_argument(std::string(name) + " must be writable");
    return a.mutable_data();
}

template <typename T>
py::array_t<T> to_array(const std::vector<T>& v, py::ssize_t cols = 0) {
    py::array_t<T> out = cols > 0 ? py::array_t<T>({static_cast<py::ssize_t>(v.size()) / cols, cols})
                                  : py::array_t<T>(static_cast<py::ssize_t>(v.size()));
    if (!v.empty()) std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
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
    // ---- time integration (in place) -----------------------------------------------
    m.def("integrate_linear", [](const std::string& method, InPlace x, InPlace v, const DArr& a, double dt) {
        double* xp = writable(x, 3, "x");
        double* vp = writable(v, 3, "v");
        if (a.ndim() != 2 || a.shape(0) != x.shape(0) || a.shape(1) != 3 || v.shape(0) != x.shape(0))
            throw std::invalid_argument("x, v and a must all have shape (N, 3)");
        if (method == "explicit_euler")
            integrate_explicit_euler(xp, vp, a.data(), x.shape(0), dt);
        else if (method == "semi_implicit_euler")
            integrate_semi_implicit_euler(xp, vp, a.data(), x.shape(0), dt);
        else
            throw std::invalid_argument("unknown integration method '" + method + "'");
    }, py::arg("method"), py::arg("x").noconvert(), py::arg("v").noconvert(), py::arg("a"), py::arg("dt"));
    m.def("quat_integrate", [](InPlace q, const DArr& w, double dt) {
        double* qp = writable(q, 4, "q");
        if (w.ndim() != 2 || w.shape(0) != q.shape(0) || w.shape(1) != 3)
            throw std::invalid_argument("w must have shape (N, 3)");
        quat_integrate(qp, w.data(), q.shape(0), dt);
    }, py::arg("q").noconvert(), py::arg("w"), py::arg("dt"));

    m.def("quat_rotate_batch", [](const DArr& q, const DArr& v) {
        if (q.ndim() != 2 || q.shape(1) != 4 || v.ndim() != 2 || v.shape(1) != 3 || v.shape(0) != q.shape(0))
            throw std::invalid_argument("expected q (N, 4) and v (N, 3)");
        py::array_t<double> out({v.shape(0), static_cast<py::ssize_t>(3)});
        quat_rotate_batch(q.data(), v.data(), q.shape(0), out.mutable_data());
        return out;
    });
    m.def("rotate_tensors", [](const DArr& q, const DArr& t) {
        if (q.ndim() != 2 || q.shape(1) != 4 || t.ndim() != 3 || t.shape(0) != q.shape(0) || t.shape(1) != 3 ||
            t.shape(2) != 3)
            throw std::invalid_argument("expected q (N, 4) and tensors (N, 3, 3)");
        py::array_t<double> out({t.shape(0), static_cast<py::ssize_t>(3), static_cast<py::ssize_t>(3)});
        rotate_tensors(q.data(), t.data(), q.shape(0), out.mutable_data());
        return out;
    });

    m.def("pgs_solve", [](const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ia,
                          const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ib, const DArr& J,
                          const DArr& rhs, const DArr& lo, const DArr& hi,
                          const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& parent,
                          const DArr& mu, const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& block,
                          const DArr& inv_m, const DArr& inv_i, InPlace v, InPlace w, int iterations,
                          const py::object& B) {
        const int64_t m = ia.size();
        DArr b_arr;
        const double* b_ptr = nullptr;
        if (!B.is_none()) {
            b_arr = DArr::ensure(B);
            if (!b_arr || b_arr.size() != 12 * m)
                throw std::invalid_argument("B must have the same shape as J");
            b_ptr = b_arr.data();
        }
        const int64_t n = inv_m.size();
        if (ib.size() != m || J.size() != 12 * m || rhs.size() != m || lo.size() != m || hi.size() != m ||
            parent.size() != m || mu.size() != m || block.size() != m)
            throw std::invalid_argument("constraint arrays must all describe the same number of rows");
        if (inv_i.size() != 9 * n || v.ndim() != 2 || v.shape(0) != n || w.ndim() != 2 || w.shape(0) != n)
            throw std::invalid_argument("body arrays must describe the same number of bodies");
        double* vp = writable(v, 3, "v");
        double* wp = writable(w, 3, "w");
        for (int64_t r = 0; r < m; ++r) {
            if (ia.data()[r] >= n || ib.data()[r] >= n || parent.data()[r] >= m)
                throw std::invalid_argument("constraint row refers to a missing body or parent row");
        }
        py::array_t<double> lambda(m);
        for (int64_t r = 0; r < m; ++r) {
            const int64_t bs = block.data()[r];
            if (bs < 0 || bs > 6 || r + bs > m)
                throw std::invalid_argument("block sizes must be 0..6 and fit in the row list");
            for (int64_t i = 1; i < bs; ++i)
                if (ia.data()[r + i] != ia.data()[r] || ib.data()[r + i] != ib.data()[r])
                    throw std::invalid_argument("rows of a block must connect the same bodies");
        }
        pgs_solve(m, ia.data(), ib.data(), J.data(), rhs.data(), lo.data(), hi.data(), parent.data(), mu.data(),
                  block.data(), n, inv_m.data(), inv_i.data(), vp, wp, iterations, lambda.mutable_data(), b_ptr);
        return lambda;
    }, py::arg("ia"), py::arg("ib"), py::arg("J"), py::arg("rhs"), py::arg("lo"), py::arg("hi"), py::arg("parent"),
       py::arg("mu"), py::arg("block"), py::arg("inv_m"), py::arg("inv_i"), py::arg("v").noconvert(),
       py::arg("w").noconvert(),
       py::arg("iterations"), py::arg("B") = py::none());

    using IArr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
    using BArr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
    m.def("contact_rows", [](const IArr& kind, const IArr& body, const DArr& center, const DArr& rot,
                             const DArr& radius, const DArr& half, const DArr& half_len, const DArr& restitution,
                             const DArr& friction, const BArr& dynamic, const IArr& excluded, const DArr& com,
                             const DArr& v, const DArr& w, double margin, double restitution_threshold,
                             const py::object& sweep, double dt, const py::object& center0,
                             const py::object& rot0, const py::object& com0, const py::object& hull_id,
                             const py::object& hulls, const py::object& static_friction, const py::object& v0,
                             const py::object& w0, const py::object& rolling) {
        const int64_t n = kind.size(), nb = dynamic.size();
        if (body.size() != n || center.size() != 3 * n || rot.size() != 9 * n || radius.size() != n ||
            half.size() != 3 * n || half_len.size() != n || restitution.size() != n || friction.size() != n)
            throw std::invalid_argument("shape arrays must all describe the same number of shapes");
        if (com.size() != 3 * nb || v.size() != 3 * nb || w.size() != 3 * nb || excluded.size() % 2 != 0)
            throw std::invalid_argument("body arrays must describe the same number of bodies");
        IArr hid_arr;
        HullSet hs;
        std::vector<py::object> keep;   // arrays referenced by `hs` stay alive for the call
        const int64_t* hid_ptr = nullptr;
        if (!hulls.is_none()) {
            if (hull_id.is_none()) throw std::invalid_argument("hulls need hull_id");
            hid_arr = IArr::ensure(hull_id);
            if (!hid_arr || hid_arr.size() != n) throw std::invalid_argument("hull_id needs one value per shape");
            hid_ptr = hid_arr.data();
            py::dict hd = hulls.cast<py::dict>();
            auto darr = [&](const char* k, py::ssize_t width) {
                DArr a = DArr::ensure(hd[k]);
                if (!a || a.size() % width != 0) throw std::invalid_argument(std::string("bad hull array ") + k);
                keep.push_back(a);
                return a.data();
            };
            auto iarr = [&](const char* k, py::ssize_t width, py::ssize_t& count) {
                IArr a = IArr::ensure(hd[k]);
                if (!a || a.size() % width != 0) throw std::invalid_argument(std::string("bad hull array ") + k);
                count = a.size() / width;
                keep.push_back(a);
                return a.data();
            };
            py::ssize_t nvo, nfo, nfv, nfvo, ne, nef, neo;
            hs.verts = darr("verts", 3);
            hs.normals = darr("normals", 3);
            hs.offsets = darr("offsets", 1);
            hs.vert_off = iarr("vert_off", 1, nvo);
            hs.face_off = iarr("face_off", 1, nfo);
            hs.face_verts = iarr("face_verts", 1, nfv);
            hs.face_vert_off = iarr("face_vert_off", 1, nfvo);
            hs.edges = iarr("edges", 2, ne);
            hs.edge_faces = iarr("edge_faces", 2, nef);
            hs.edge_off = iarr("edge_off", 1, neo);
            hs.n = nvo - 1;
            if (hs.n < 0 || nfo != nvo || neo != nvo || ne != nef ||
                nfvo != hs.face_off[hs.n] + 1 || hs.face_vert_off[nfvo - 1] != nfv || hs.edge_off[hs.n] != ne)
                throw std::invalid_argument("inconsistent hull arrays");
        }
        for (int64_t i = 0; i < n; ++i) {
            if (kind.data()[i] < 0 || kind.data()[i] > 4) throw std::invalid_argument("unknown shape kind");
            if (kind.data()[i] == 4 && (!hid_ptr || hid_ptr[i] < 0 || hid_ptr[i] >= hs.n))
                throw std::invalid_argument("a hull shape needs valid hull data");
            if (hid_ptr && (hid_ptr[i] < -1 || hid_ptr[i] >= hs.n)) throw std::invalid_argument("bad hull_id");
            if (body.data()[i] < 0 || body.data()[i] >= nb) throw std::invalid_argument("shape refers to a missing body");
        }
        DArr sweep_arr;
        const double* sweep_ptr = nullptr;
        if (!sweep.is_none()) {
            sweep_arr = DArr::ensure(sweep);
            if (!sweep_arr || sweep_arr.size() != n) throw std::invalid_argument("sweep needs one value per shape");
            sweep_ptr = sweep_arr.data();
        }
        DArr c0, r0, m0;
        const bool start = !center0.is_none();
        if (start) {
            c0 = DArr::ensure(center0);
            r0 = DArr::ensure(rot0);
            m0 = DArr::ensure(com0);
            if (!c0 || !r0 || !m0 || c0.size() != 3 * n || r0.size() != 9 * n || m0.size() != 3 * nb)
                throw std::invalid_argument("start-of-step arrays must match the shapes and bodies");
        }
        DArr sf_arr, v0_arr, w0_arr, roll_arr;   // (a default array_t is an empty array, not null: use the flags)
        const bool has_sf = !static_friction.is_none(), has_v0 = !v0.is_none() || !w0.is_none();
        const bool has_roll = !rolling.is_none();
        if (has_roll) {
            roll_arr = DArr::ensure(rolling);
            if (!roll_arr || roll_arr.size() != n) throw std::invalid_argument("rolling needs one value per shape");
        }
        if (has_sf) {
            sf_arr = DArr::ensure(static_friction);
            if (!sf_arr || sf_arr.size() != n) throw std::invalid_argument("static_friction needs one value per shape");
        }
        if (has_v0) {
            v0_arr = DArr::ensure(v0);
            w0_arr = DArr::ensure(w0);
            if (!v0_arr || !w0_arr || v0_arr.size() != 3 * nb || w0_arr.size() != 3 * nb)
                throw std::invalid_argument("v0 and w0 must both be given, one vector per body");
        }
        ShapeSet s{n, kind.data(), body.data(), center.data(), rot.data(), radius.data(), half.data(),
                   half_len.data(), restitution.data(), friction.data(), has_sf ? sf_arr.data() : nullptr,
                   has_roll ? roll_arr.data() : nullptr, sweep_ptr,
                   start ? c0.data() : nullptr, start ? r0.data() : nullptr, hid_ptr, hid_ptr ? &hs : nullptr};
        BodySet b{nb, dynamic.data(), com.data(), v.data(), w.data(), start ? m0.data() : nullptr,
                  has_v0 ? v0_arr.data() : nullptr, has_v0 ? w0_arr.data() : nullptr};
        ContactOutput out;
        {
            py::gil_scoped_release release;   // pure C++ on copies of the inputs' pointers
            contact_rows(s, b, excluded.data(), excluded.size() / 2, margin, restitution_threshold, out, dt);
        }
        py::dict d;
        d["pairs"] = to_array(out.pairs, 2);
        d["shape_a"] = to_array(out.shape_a);
        d["shape_b"] = to_array(out.shape_b);
        d["point"] = to_array(out.point, 3);
        d["normal"] = to_array(out.normal, 3);
        d["depth"] = to_array(out.depth);
        d["row"] = to_array(out.row);
        d["ia"] = to_array(out.ia);
        d["ib"] = to_array(out.ib);
        d["J"] = to_array(out.J, 12);
        d["rhs"] = to_array(out.rhs);
        d["lo"] = to_array(out.lo);
        d["hi"] = to_array(out.hi);
        d["parent"] = to_array(out.parent);
        d["mu"] = to_array(out.mu);
        d["err"] = to_array(out.err);
        d["mode"] = to_array(out.mode);
        return d;
    }, py::arg("kind"), py::arg("body"), py::arg("center"), py::arg("rot"), py::arg("radius"), py::arg("half"),
       py::arg("half_len"), py::arg("restitution"), py::arg("friction"), py::arg("dynamic"), py::arg("excluded"),
       py::arg("com"), py::arg("v"), py::arg("w"), py::arg("margin"), py::arg("restitution_threshold"),
       py::arg("sweep") = py::none(), py::arg("dt") = 0.0, py::arg("center0") = py::none(),
       py::arg("rot0") = py::none(), py::arg("com0") = py::none(), py::arg("hull_id") = py::none(),
       py::arg("hulls") = py::none(), py::arg("static_friction") = py::none(), py::arg("v0") = py::none(),
       py::arg("w0") = py::none(), py::arg("rolling") = py::none());

    m.def("joint_rows", [](const IArr& kind, const IArr& body_a, const IArr& body_b, const DArr& anchor_a,
                           const DArr& anchor_b, const DArr& axis_a, const DArr& axis_b, const DArr& ref_rot,
                           const IArr& flags, const DArr& lower, const DArr& upper, const DArr& motor_speed,
                           const DArr& motor_max, const DArr& length, const DArr& origin, const DArr& com,
                           const DArr& rot, double dt, double slop, const py::object& mid_origin,
                           const py::object& mid_com, const py::object& mid_rot) {
        const int64_t n = kind.size(), nb = origin.size() / 3;
        if (body_a.size() != n || body_b.size() != n || anchor_a.size() != 3 * n || anchor_b.size() != 3 * n ||
            axis_a.size() != 3 * n || axis_b.size() != 3 * n || ref_rot.size() != 9 * n || flags.size() != n ||
            lower.size() != n || upper.size() != n || motor_speed.size() != n || motor_max.size() != n ||
            length.size() != n)
            throw std::invalid_argument("joint arrays must all describe the same number of joints");
        if (origin.size() != 3 * nb || com.size() != 3 * nb || rot.size() != 9 * nb)
            throw std::invalid_argument("body arrays must describe the same number of bodies");
        if (!(dt > 0)) throw std::invalid_argument("dt must be positive");
        for (int64_t k = 0; k < n; ++k) {
            if (kind.data()[k] < 0 || kind.data()[k] > 4) throw std::invalid_argument("unknown joint kind");
            if (body_a.data()[k] < 0 || body_a.data()[k] >= nb || body_b.data()[k] < -1 || body_b.data()[k] >= nb)
                throw std::invalid_argument("joint refers to a missing body");
        }
        DArr mo, mc, mr;
        BodyConfig mid;
        const bool has_mid = !mid_origin.is_none();
        if (has_mid) {
            mo = DArr::ensure(mid_origin);
            mc = DArr::ensure(mid_com);
            mr = DArr::ensure(mid_rot);
            if (!mo || !mc || !mr || mo.size() != 3 * nb || mc.size() != 3 * nb || mr.size() != 9 * nb)
                throw std::invalid_argument("midpoint arrays must match the bodies");
            mid = {mo.data(), mc.data(), mr.data()};
        }
        JointSet js{n, kind.data(), body_a.data(), body_b.data(), anchor_a.data(), anchor_b.data(), axis_a.data(),
                    axis_b.data(), ref_rot.data(), flags.data(), lower.data(), upper.data(), motor_speed.data(),
                    motor_max.data(), length.data()};
        const BodyConfig end{origin.data(), com.data(), rot.data()};
        JointOutput out;
        {
            py::gil_scoped_release release;
            joint_rows(js, end, has_mid ? &mid : nullptr, dt, slop, out);
        }
        py::dict d;
        d["ia"] = to_array(out.ia);
        d["ib"] = to_array(out.ib);
        d["J"] = to_array(out.J, 12);
        d["B"] = to_array(out.B, 12);
        d["rhs"] = to_array(out.rhs);
        d["lo"] = to_array(out.lo);
        d["hi"] = to_array(out.hi);
        d["err"] = to_array(out.err);
        d["mode"] = to_array(out.mode);
        d["block"] = to_array(out.block);
        d["joint"] = to_array(out.joint);
        return d;
    }, py::arg("kind"), py::arg("body_a"), py::arg("body_b"), py::arg("anchor_a"), py::arg("anchor_b"),
       py::arg("axis_a"), py::arg("axis_b"), py::arg("ref_rot"), py::arg("flags"), py::arg("lower"), py::arg("upper"),
       py::arg("motor_speed"), py::arg("motor_max"), py::arg("length"), py::arg("origin"), py::arg("com"),
       py::arg("rot"), py::arg("dt"), py::arg("slop"), py::arg("mid_origin") = py::none(),
       py::arg("mid_com") = py::none(), py::arg("mid_rot") = py::none());

    m.def("convex_decomposition", [](const DArr& vertices, const py::array_t<int64_t, py::array::c_style |
                                     py::array::forcecast>& triangles, int64_t max_parts, double tolerance,
                                     int64_t max_hull_points) {
        if (vertices.ndim() != 2 || vertices.shape(1) != 3) throw std::invalid_argument("vertices must be (V, 3)");
        if (triangles.ndim() != 2 || triangles.shape(1) != 3) throw std::invalid_argument("triangles must be (T, 3)");
        if (max_parts < 1) throw std::invalid_argument("max_parts must be at least 1");
        std::vector<std::vector<Point3>> parts;
        {
            py::gil_scoped_release release;
            parts = convex_decomposition(vertices.data(), vertices.shape(0), triangles.data(), triangles.shape(0),
                                         max_parts, tolerance, max_hull_points);
        }
        py::list out;
        for (const auto& p : parts) {
            py::array_t<double> a({static_cast<py::ssize_t>(p.size()), py::ssize_t(3)});
            auto r = a.mutable_unchecked<2>();
            for (size_t i = 0; i < p.size(); ++i)
                for (int k = 0; k < 3; ++k) r(static_cast<py::ssize_t>(i), k) = p[i][k];
            out.append(a);
        }
        return out;
    }, py::arg("vertices"), py::arg("triangles"), py::arg("max_parts") = 16, py::arg("tolerance") = 0.03,
       py::arg("max_hull_points") = 0);
    m.def("hull_volume", [](const DArr& points, int64_t max_points) {
        if (points.ndim() != 2 || points.shape(1) != 3) throw std::invalid_argument("points must be (N, 3)");
        std::vector<Point3> p(static_cast<size_t>(points.shape(0)));
        for (py::ssize_t i = 0; i < points.shape(0); ++i) p[i] = {points.at(i, 0), points.at(i, 1), points.at(i, 2)};
        return hull_volume(std::move(p), max_points);
    }, py::arg("points"), py::arg("max_points") = 0);

    m.def("mass_properties_shell", [](const FArr& v, const UArr& f, double areal_density) {
        check_mesh(v, f);
        return mass_dict(mass_properties_shell(v.data(), f.data(), f.shape(0), areal_density));
    });
}
