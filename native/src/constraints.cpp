#include "psim/constraints.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace psim {

namespace {

inline double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

inline void mat_vec(const double* m, const double* x, double* out) {
    for (int i = 0; i < 3; ++i) out[i] = m[3 * i] * x[0] + m[3 * i + 1] * x[1] + m[3 * i + 2] * x[2];
}

}  // namespace

constexpr int kMaxBlock = 6;

// Inverse of a k×k (k ≤ 6) matrix by Gauss–Jordan elimination with partial pivoting; false if singular.
bool invert_small(int k, const double* a, double* out) {
    double m[kMaxBlock][2 * kMaxBlock] = {};
    double scale = 0.0;
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) {
            m[i][j] = a[i * k + j];
            scale = std::max(scale, std::fabs(a[i * k + j]));
        }
    if (scale <= 0.0) return false;
    for (int i = 0; i < k; ++i) m[i][k + i] = 1.0;
    for (int c = 0; c < k; ++c) {
        int piv = c;
        for (int r = c + 1; r < k; ++r)
            if (std::fabs(m[r][c]) > std::fabs(m[piv][c])) piv = r;
        if (std::fabs(m[piv][c]) < 1e-12 * scale) return false;
        if (piv != c)
            for (int j = 0; j < 2 * k; ++j) std::swap(m[c][j], m[piv][j]);
        const double d = m[c][c];
        for (int j = 0; j < 2 * k; ++j) m[c][j] /= d;
        for (int r = 0; r < k; ++r) {
            if (r == c || m[r][c] == 0.0) continue;
            const double f = m[r][c];
            for (int j = 0; j < 2 * k; ++j) m[r][j] -= f * m[c][j];
        }
    }
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) out[i * k + j] = m[i][k + j];
    return true;
}

void pgs_solve(int64_t m, const int64_t* ia, const int64_t* ib, const double* J, const double* rhs,
               const double* lo, const double* hi, const int64_t* parent, const double* mu, const int64_t* block,
               int64_t n, const double* inv_m, const double* inv_i, double* v, double* w, int iterations,
               double* lambda, const double* B) {
    if (B == nullptr) B = J;
    // M^-1 B^T per row (12 values) and effective masses J M^-1 B^T
    std::vector<double> mj(static_cast<size_t>(m) * 12, 0.0);
    std::vector<double> k(static_cast<size_t>(m), 0.0);
    for (int64_t r = 0; r < m; ++r) {
        const double* j = J + 12 * r;
        const double* bj = B + 12 * r;
        double* out = mj.data() + 12 * r;
        double kr = 0.0;
        if (ia[r] >= 0 && ia[r] < n) {
            const int64_t a = ia[r];
            for (int c = 0; c < 3; ++c) out[c] = inv_m[a] * bj[c];
            mat_vec(inv_i + 9 * a, bj + 3, out + 3);
            kr += dot3(j, out) + dot3(j + 3, out + 3);
        }
        if (ib[r] >= 0 && ib[r] < n) {
            const int64_t b = ib[r];
            for (int c = 0; c < 3; ++c) out[6 + c] = inv_m[b] * bj[6 + c];
            mat_vec(inv_i + 9 * b, bj + 9, out + 9);
            kr += dot3(j + 6, out + 6) + dot3(j + 9, out + 9);
        }
        k[r] = kr;
        lambda[r] = 0.0;
    }
    // inverse effective-mass matrices of the blocks (K_ij = J_i · M^-1 B_j^T)
    std::vector<double> kinv(static_cast<size_t>(m) * kMaxBlock * kMaxBlock, 0.0);
    std::vector<char> block_ok(static_cast<size_t>(m), 0);
    std::vector<int64_t> eff(block, block + m);   // singular blocks fall back to scalar rows
    for (int64_t r = 0; r < m; ++r) {
        const int64_t bs = block[r];
        if (bs < 2 || bs > kMaxBlock || r + bs > m) continue;
        double K[kMaxBlock * kMaxBlock] = {0};
        for (int64_t i = 0; i < bs; ++i)
            for (int64_t j = 0; j < bs; ++j) {
                const double* ji = J + 12 * (r + i);
                const double* mjj = mj.data() + 12 * (r + j);
                double s = 0.0;
                if (ia[r] >= 0) s += dot3(ji, mjj) + dot3(ji + 3, mjj + 3);
                if (ib[r] >= 0) s += dot3(ji + 6, mjj + 6) + dot3(ji + 9, mjj + 9);
                K[i * bs + j] = s;
            }
        block_ok[r] = invert_small(static_cast<int>(bs), K, kinv.data() + kMaxBlock * kMaxBlock * r) ? 1 : 0;
        if (!block_ok[r])
            for (int64_t i = 0; i < bs; ++i) eff[r + i] = 1;
    }
    for (int it = 0; it < iterations; ++it) {
        for (int64_t r = 0; r < m; ++r) {
            if (eff[r] == 0) continue;             // solved with its block
            if (eff[r] >= 2 && block_ok[r]) {
                const int64_t bs = eff[r];
                const int64_t a = ia[r], b = ib[r];
                double res[kMaxBlock] = {0}, dl[kMaxBlock] = {0};
                for (int64_t i = 0; i < bs; ++i) {
                    const double* j = J + 12 * (r + i);
                    double cdot = 0.0;
                    if (a >= 0) cdot += dot3(j, v + 3 * a) + dot3(j + 3, w + 3 * a);
                    if (b >= 0) cdot += dot3(j + 6, v + 3 * b) + dot3(j + 9, w + 3 * b);
                    res[i] = rhs[r + i] - cdot;
                }
                const double* ki = kinv.data() + kMaxBlock * kMaxBlock * r;
                for (int64_t i = 0; i < bs; ++i)
                    for (int64_t j = 0; j < bs; ++j) dl[i] += ki[i * bs + j] * res[j];
                for (int64_t i = 0; i < bs; ++i) {
                    lambda[r + i] += dl[i];
                    const double* mr = mj.data() + 12 * (r + i);
                    for (int c = 0; c < 3; ++c) {
                        if (a >= 0) {
                            v[3 * a + c] += mr[c] * dl[i];
                            w[3 * a + c] += mr[3 + c] * dl[i];
                        }
                        if (b >= 0) {
                            v[3 * b + c] += mr[6 + c] * dl[i];
                            w[3 * b + c] += mr[9 + c] * dl[i];
                        }
                    }
                }
                continue;
            }
            if (k[r] <= 1e-14) continue;
            const double* j = J + 12 * r;
            const int64_t a = ia[r], b = ib[r];
            double cdot = 0.0;
            if (a >= 0) cdot += dot3(j, v + 3 * a) + dot3(j + 3, w + 3 * a);
            if (b >= 0) cdot += dot3(j + 6, v + 3 * b) + dot3(j + 9, w + 3 * b);
            double lo_r = lo[r], hi_r = hi[r];
            if (parent[r] >= 0) {
                const double limit = mu[r] * lambda[parent[r]];
                lo_r = -limit;
                hi_r = limit;
            }
            const double old = lambda[r];
            const double next = std::min(hi_r, std::max(lo_r, old + (rhs[r] - cdot) / k[r]));
            const double dl = next - old;
            lambda[r] = next;
            if (dl == 0.0) continue;
            const double* mr = mj.data() + 12 * r;
            if (a >= 0) {
                for (int c = 0; c < 3; ++c) {
                    v[3 * a + c] += mr[c] * dl;
                    w[3 * a + c] += mr[3 + c] * dl;
                }
            }
            if (b >= 0) {
                for (int c = 0; c < 3; ++c) {
                    v[3 * b + c] += mr[6 + c] * dl;
                    w[3 * b + c] += mr[9 + c] * dl;
                }
            }
        }
    }
}

}  // namespace psim
