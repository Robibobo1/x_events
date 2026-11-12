#pragma once

// Cost function of the EKLT optimizer (ErrorRotation, equation (7) of the paper) with hand-written derivatives, in the
// form that Vitis HLS synthesizes: fixed sizes, plain arrays, no Ceres, OpenCV or Eigen, no dynamic memory.
//
// Same two passes and formulas as ErrorRotationAnalytic (src/x/eklt/error_analytic.cpp), and the same bicubic
// interpolation as ceres::BiCubicInterpolator (Ceres 1.14), but on a window of the template gradient around the
// feature instead of the whole image.
//
// T is the number type: float in EKLT-VIO (EKLT_COST=kernel) and for HLS, double only to check that the code gives
// the same numbers as the original. Both come from this one source text, so they can only differ in precision.

#include <cmath>

namespace x {
namespace kernel {

constexpr int kPatchSize = 19;                       // EKLT patch, eklt_patch_size
constexpr int kHalfSize = (kPatchSize - 1) / 2;
constexpr int kNumPixels = kPatchSize * kPatchSize;  // = number of residuals
constexpr int kNumParams = 4;                        // theta, tx, ty, phi
constexpr int kWindowSize = 48;                      // template gradient window, kWindowSize x kWindowSize x (Ix, Iy)

// ceres::CubicHermiteSpline for one value. The cubic goes through p1 at s = 0 and p2 at s = 1, with the slopes
// (p2 - p0) / 2 and (p3 - p1) / 2 there. Returns its value f and its slope dfds at s, same expressions as Ceres.
template <typename T>
inline void cubicHermite(T p0, T p1, T p2, T p3, T s, T &f, T &dfds) {
  const T a = T(0.5) * (-p0 + T(3.0) * p1 - T(3.0) * p2 + p3);
  const T b = T(0.5) * (T(2.0) * p0 - T(5.0) * p1 + T(4.0) * p2 - p3);
  const T c = T(0.5) * (-p0 + p2);
  const T d = p1;
  f = d + s * (c + s * (b + s * a));
  dfds = c + s * (T(2.0) * b + T(3.0) * a * s);
}

// index into the window, clamped to its edge as ceres::Grid2D clamps to the edge of its grid
inline int clampToWindow(int i) { return i < 0 ? 0 : (i > kWindowSize - 1 ? kWindowSize - 1 : i); }

// ceres::BiCubicInterpolator::Evaluate(r, c) on the window. r, c: image row and column of the point. window[0][0] is
// the image cell (row0, col0). Returns the gradient f = (Ix, Iy) at the point and its slopes along the rows (dfdr) and
// the columns (dfdc). The fraction inside the cell comes from the image coordinates, as in Ceres, the window only
// shifts the indexes. Returns false if the 4x4 cells around the point are not all inside the window: the values are
// then those of the window edge, not of the image.
template <typename T>
inline bool bicubic(const T window[kWindowSize][kWindowSize][2], int row0, int col0, T r, T c,
                    T f[2], T dfdr[2], T dfdc[2]) {
  const int row = static_cast<int>(std::floor(r));
  const int col = static_cast<int>(std::floor(c));
  const T sr = r - T(row), sc = c - T(col);

  // the 4x4 cells: rows row-1 .. row+2, columns col-1 .. col+2, in window indexes
  const int wr = row - row0, wc = col - col0;
  const bool inside = wr - 1 >= 0 && wr + 2 <= kWindowSize - 1 && wc - 1 >= 0 && wc + 2 <= kWindowSize - 1;
  const int c0 = clampToWindow(wc - 1), c1 = clampToWindow(wc), c2 = clampToWindow(wc + 1), c3 = clampToWindow(wc + 2);

  // along each of the 4 rows: value and slope along the columns at sc
  T f_row[4][2], dfdc_row[4][2];
  for (int j = 0; j < 4; ++j) {
    const int y = clampToWindow(wr - 1 + j);
    for (int k = 0; k < 2; ++k)
      cubicHermite(window[y][c0][k], window[y][c1][k], window[y][c2][k], window[y][c3][k], sc, f_row[j][k],
                   dfdc_row[j][k]);
  }

  // across the rows at sr: the value and the slope along the rows, then the slope along the columns
  for (int k = 0; k < 2; ++k) {
    cubicHermite(f_row[0][k], f_row[1][k], f_row[2][k], f_row[3][k], sr, f[k], dfdr[k]);
    T unused;
    cubicHermite(dfdc_row[0][k], dfdc_row[1][k], dfdc_row[2][k], dfdc_row[3][k], sr, dfdc[k], unused);
  }
  return inside;
}

// The cost function. Inputs:
//   window       template gradient (Ix, Iy) around the feature, window[0][0] = image cell (row0, col0)
//   event_term   e_i / |e|, the normalized event frame, row major (y, x)
//   origin_x/_y  image position of patch pixel (0, 0): patch center - kHalfSize
//   theta .. phi the parameters
// Outputs: residuals r_i, and if with_jacobian the Jacobian J[i] = d r_i / d(theta, tx, ty, phi).
// Returns the number of pixels whose 4x4 interpolation cells were not all inside the window (must stay 0).
//
// For pixel i, with q the position where the template gradient g is read, v = (cos phi, sin phi) and a = g . v:
//   r_i  = a_i / sqrt(S) + e_i / |e|,           S = 1e-3 + sum_j a_j^2
//   J_ik = (da_i/dk - a_i * P_k / S) / sqrt(S),  P_k = sum_j a_j * da_j/dk
// S and the four sums P_k need every pixel, hence two passes over the patch.
template <typename T>
int costKernel(const T window[kWindowSize][kWindowSize][2], int row0, int col0, const T event_term[kNumPixels],
               T origin_x, T origin_y, T theta, T tx, T ty, T phi, bool with_jacobian, T residuals[kNumPixels],
               T jacobian[kNumPixels][kNumParams]) {
  // motion direction v and rotation R(theta) = [a0 a1, a2 a3], named as in ErrorRotation::warp
  const T vx = std::cos(phi), vy = std::sin(phi);
  const T a0 = std::cos(theta), a1 = -std::sin(theta);
  const T a2 = -a1, a3 = a0;
  const T ox = origin_x, oy = origin_y;

  // results of pass 1, kept for pass 2: a_i and its derivatives
  T a_pix[kNumPixels];
  T da_pix[kNumPixels][kNumParams];

  // Pass 1: a_i and its derivatives, sums S and P_k
  T S = T(1e-3);
  T P[kNumParams] = {T(0), T(0), T(0), T(0)};  // sum of a_j * d a_j / d(theta, tx, ty, phi)
  int outside = 0;
  for (int y = 0; y < kPatchSize; ++y) {
    for (int x = 0; x < kPatchSize; ++x) {
      const int i = x + kPatchSize * y;

      // q = R(theta) u + t, the same expression as in ErrorRotation::warp
      const T x_p = a0 * T(x) + a1 * T(y) + a0 * ox + a1 * oy + tx;
      const T y_p = a2 * T(x) + a3 * T(y) + a2 * ox + a3 * oy + ty;

      // template gradient g = (Ix, Iy) at q, and its slopes along the rows (y) and the columns (x)
      T g[2], g_y[2], g_x[2];
      if (!bicubic(window, row0, col0, y_p, x_p, g, g_y, g_x))
        ++outside;

      const T a = g[0] * vx + g[1] * vy;
      a_pix[i] = a;
      S += a * a;
      if (!with_jacobian)
        continue;

      // how a changes when q moves one pixel right (a_x) or down (a_y)
      const T a_x = g_x[0] * vx + g_x[1] * vy;
      const T a_y = g_y[0] * vx + g_y[1] * vy;
      da_pix[i][0] = -(y_p - ty) * a_x + (x_p - tx) * a_y;  // theta: dq/dtheta = (-(q_y - t_y), q_x - t_x)
      da_pix[i][1] = a_x;                                   // tx
      da_pix[i][2] = a_y;                                   // ty
      da_pix[i][3] = -g[0] * vy + g[1] * vx;                // phi: g . (-sin phi, cos phi)
      for (int k = 0; k < kNumParams; ++k)
        P[k] += a * da_pix[i][k];
    }
  }

  // Pass 2: normalize. r_i = a_i / sqrt(S) + e_i / |e|, J_ik = (da_i/dk - a_i * P_k / S) / sqrt(S)
  const T inv_sqrt_S = T(1.0) / std::sqrt(S);
  T P_by_S[kNumParams];
  for (int k = 0; k < kNumParams; ++k)
    P_by_S[k] = P[k] / S;

  for (int i = 0; i < kNumPixels; ++i) {
    const T a = a_pix[i];
    if (with_jacobian)
      for (int k = 0; k < kNumParams; ++k)
        jacobian[i][k] = (da_pix[i][k] - a * P_by_S[k]) * inv_sqrt_S;
    residuals[i] = a * inv_sqrt_S + event_term[i];
  }
  return outside;
}

}  // namespace kernel
}  // namespace x
