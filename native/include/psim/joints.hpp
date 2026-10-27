// Joint constraint rows for one simulation step.
// Native counterpart of psim/physics/joint_kernel.py (same rows, same order, same values).
#pragma once

#include <cstdint>
#include <vector>

namespace psim {

enum JointKind : int64_t { kFixed = 0, kBall = 1, kRevolute = 2, kPrismatic = 3, kDistance = 4 };
enum JointFlag : int64_t { kLimit = 1, kMotor = 2, kRope = 4, kSpring = 8 };

struct JointSet {
    int64_t n = 0;
    const int64_t* kind = nullptr;
    const int64_t* body_a = nullptr;      // solver body indices; body_b = -1 for the fixed world
    const int64_t* body_b = nullptr;
    const double* anchor_a = nullptr;     // (n, 3) in body A's frame
    const double* anchor_b = nullptr;     // (n, 3) in body B's frame, or world
    const double* axis_a = nullptr;
    const double* axis_b = nullptr;
    const double* ref_rot = nullptr;      // (n, 3, 3) R_A^-1 R_B at creation
    const int64_t* flags = nullptr;       // JointFlag bits
    const double* lower = nullptr;
    const double* upper = nullptr;
    const double* motor_speed = nullptr;
    const double* motor_max = nullptr;
    const double* length = nullptr;
};

// Body configuration: origins and centres of mass (nb, 3), rotations (nb, 3, 3) row-major.
struct BodyConfig {
    const double* origin = nullptr;
    const double* com = nullptr;
    const double* rot = nullptr;
};

struct JointOutput {
    std::vector<int64_t> ia, ib, mode, block, joint;   // mode: 0 equality, 1 inequality, 2 no position correction
    std::vector<double> J, B, rhs, lo, hi, err;
};

// Rows of all joints, enforced at `end`; impulse directions B from `mid` (nullptr: B = J).
void joint_rows(const JointSet& j, const BodyConfig& end, const BodyConfig* mid, double dt, double slop,
                JointOutput& out);

}  // namespace psim
