#pragma once

// Variants of the EKLT cost function for the timing benchmark (test/eklt_cost_timing.cpp). Each one is a copy of
// ErrorRotationAnalytic (src/x/eklt/error_analytic.cpp) or of the cost kernel (x/eklt/hls/cost_kernel.h with its
// wrapper x/eklt/error_kernel.h) in which one thing can be changed at a time:
//   - where the template gradient is read: Ceres' BiCubicInterpolator on a Grid2D, or the kernel's bicubic() on its
//     48x48 window array
//   - the event term e_i / |e|: divided at every call (as the original) or once per solve (as the kernel)
//   - the patch size: read from the config at run time (as the hand-written version) or a compile-time constant
// Unchanged, a copy must give the same numbers and the same time as the code it copies (checked by the benchmark).
// Double only: these variants are about the code, not the number type.

#include <cmath>
#include <memory>
#include <utility>

#include <ceres/ceres.h>
#include <opencv2/core/core.hpp>

#include <x/eklt/error.h>
#include <x/eklt/hls/cost_kernel.h>

namespace x {
namespace timing {

constexpr int kN = kernel::kNumPixels;
constexpr int kW = kernel::kWindowSize;
using Window = double[kW][kW][2];

// Ceres' interpolator, as the original and the hand-written versions call it. derivatives false: only the value, as
// the hand-written version asks when Ceres wants no Jacobian. Always true: the window is never left.
struct CeresSampler {
  const Interpolator *interp;
  bool operator()(double r, double c, double g[2], double g_r[2], double g_c[2], bool derivatives) const {
    if (derivatives)
      interp->Evaluate(r, c, g, g_r, g_c);
    else
      interp->Evaluate(r, c, g, nullptr, nullptr);
    return true;
  }
};

// The kernel's bicubic() on a 48x48 window, window[0][0] = image cell (row0, col0). Always computes the derivatives,
// as the kernel does. Returns false if the point needs cells outside the window.
struct WindowSampler {
  const double (*window)[kW][2];
  int row0, col0;
  bool operator()(double r, double c, double g[2], double g_r[2], double g_c[2], bool) const {
    return kernel::bicubic<double>(window, row0, col0, r, c, g, g_r, g_c);
  }
};

// Copy of ceres::BiCubicInterpolator<ceres::Grid2D<double, 2>>::Evaluate with Grid2D::GetValue (Ceres 1.14) on a
// row-major grid of (Ix, Iy), to find which part of it costs the time. Unchanged (both false) it is the same code,
// inlined as GCC 9 inlines Ceres' in ErrorRotationAnalytic: everything except the Hermite step, which is called.
//   kVectorLoad:   GetValue loads the cell (Ix, Iy) as one Eigen vector, instead of writing its two values one by
//                  one into the vector that the Hermite step then reads as a whole
//   kInlineSpline: the Hermite step is a copy of ceres::CubicHermiteSpline<2> that is always inlined. Ceres' own is
//                  called, so each cell goes through memory: two 8-byte stores, then one 16-byte load in the call
template <bool kVectorLoad, bool kInlineSpline>
struct CeresCopySampler {
  using Vec = Eigen::Matrix<double, 2, 1>;
  const double *data;
  int row_begin, row_end, col_begin, col_end;

  __attribute__((always_inline)) inline void get(int r, int c, Vec &p) const {
    const int row_idx = std::min(std::max(row_begin, r), row_end - 1) - row_begin;
    const int col_idx = std::min(std::max(col_begin, c), col_end - 1) - col_begin;
    const int n = (col_end - col_begin) * row_idx + col_idx;
    if (kVectorLoad)
      p = Eigen::Map<const Vec>(data + 2 * n);
    else
      for (int i = 0; i < 2; ++i)
        p.data()[i] = data[2 * n + i];
  }

  // same expressions as ceres::CubicHermiteSpline<2> (Ceres 1.14)
  __attribute__((always_inline)) static inline void inlineSpline(const Vec &p0, const Vec &p1, const Vec &p2,
                                                                 const Vec &p3, double x, double *f, double *dfdx) {
    const Vec a = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
    const Vec b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
    const Vec c = 0.5 * (-p0 + p2);
    const Vec d = p1;
    if (f != nullptr)
      Eigen::Map<Vec>(f, 2) = d + x * (c + x * (b + x * a));
    if (dfdx != nullptr)
      Eigen::Map<Vec>(dfdx, 2) = c + x * (2.0 * b + 3.0 * a * x);
  }

  __attribute__((always_inline)) static inline void spline(const Vec &p0, const Vec &p1, const Vec &p2,
                                                           const Vec &p3, double x, double *f, double *dfdx) {
    if (kInlineSpline)
      inlineSpline(p0, p1, p2, p3, x, f, dfdx);
    else
      ceres::CubicHermiteSpline<2>(p0, p1, p2, p3, x, f, dfdx);
  }

  __attribute__((always_inline)) inline void evaluate(double r, double c, double *f, double *dfdr, double *dfdc) const {
    const int row = std::floor(r);
    const int col = std::floor(c);
    Vec p0, p1, p2, p3;
    Vec f0, f1, f2, f3;
    Vec df0dc, df1dc, df2dc, df3dc;
    get(row - 1, col - 1, p0);
    get(row - 1, col, p1);
    get(row - 1, col + 1, p2);
    get(row - 1, col + 2, p3);
    spline(p0, p1, p2, p3, c - col, f0.data(), df0dc.data());
    get(row, col - 1, p0);
    get(row, col, p1);
    get(row, col + 1, p2);
    get(row, col + 2, p3);
    spline(p0, p1, p2, p3, c - col, f1.data(), df1dc.data());
    get(row + 1, col - 1, p0);
    get(row + 1, col, p1);
    get(row + 1, col + 1, p2);
    get(row + 1, col + 2, p3);
    spline(p0, p1, p2, p3, c - col, f2.data(), df2dc.data());
    get(row + 2, col - 1, p0);
    get(row + 2, col, p1);
    get(row + 2, col + 1, p2);
    get(row + 2, col + 2, p3);
    spline(p0, p1, p2, p3, c - col, f3.data(), df3dc.data());
    spline(f0, f1, f2, f3, r - row, f, dfdr);
    if (dfdc != nullptr)
      spline(df0dc, df1dc, df2dc, df3dc, r - row, dfdc, nullptr);
  }

  __attribute__((always_inline)) inline bool operator()(double r, double c, double g[2], double g_r[2],
                                                        double g_c[2], bool derivatives) const {
    if (derivatives)
      evaluate(r, c, g, g_r, g_c);
    else
      evaluate(r, c, g, nullptr, nullptr);
    return true;
  }
};

// Copy of ErrorRotationAnalytic::Evaluate. kEventsOnce: e_i / |e| computed once in the constructor. kFixedSize: the
// patch size is the constant 19 instead of config->patch_size_.
template <typename Sampler, bool kEventsOnce, bool kFixedSize>
class AnalyticVariant : public ceres::CostFunction {
 public:
  AnalyticVariant(std::shared_ptr<CostFunctionConfig> config, Sampler sample)
      : config_(std::move(config)), sample_(sample) {
    set_num_residuals(config_->size_);
    mutable_parameter_block_sizes()->push_back(3);
    mutable_parameter_block_sizes()->push_back(1);
    if (kEventsOnce)
      for (int y = 0; y < config_->patch_size_; y++)
        for (int x = 0; x < config_->patch_size_; x++)
          event_term_[x + config_->patch_size_ * y] = config_->event_frame_->at<double>(y, x) / config_->norm_event_frame_;
  }

  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
    const CostFunctionConfig &c = *config_;
    const int n = kFixedSize ? kernel::kPatchSize : c.patch_size_;

    const double theta = parameters[0][0], tx = parameters[0][1], ty = parameters[0][2];
    const double phi = parameters[1][0];

    double *J_p = jacobians != nullptr ? jacobians[0] : nullptr;
    double *J_v = jacobians != nullptr ? jacobians[1] : nullptr;
    const bool with_jacobian = J_p != nullptr || J_v != nullptr;

    const double vx = std::cos(phi), vy = std::sin(phi);
    const double a0 = std::cos(theta), a1 = -std::sin(theta);
    const double a2 = -a1, a3 = a0;
    const double ox = c.feature_.x - c.half_size_;
    const double oy = c.feature_.y - c.half_size_;

    double S = 1e-3;
    double P[4] = {0.0, 0.0, 0.0, 0.0};
    for (int y = 0; y < n; y++) {
      for (int x = 0; x < n; x++) {
        const int i = x + n * y;
        const double x_p = a0 * x + a1 * y + a0 * ox + a1 * oy + tx;
        const double y_p = a2 * x + a3 * y + a2 * ox + a3 * oy + ty;

        double g[2], g_y[2], g_x[2];
        sample_(y_p, x_p, g, g_y, g_x, with_jacobian);

        const double a = g[0] * vx + g[1] * vy;
        residuals[i] = a;
        S += a * a;
        if (!with_jacobian)
          continue;

        const double a_x = g_x[0] * vx + g_x[1] * vy;
        const double a_y = g_y[0] * vx + g_y[1] * vy;
        const double da[4] = {
            -(y_p - ty) * a_x + (x_p - tx) * a_y,
            a_x,
            a_y,
            -g[0] * vy + g[1] * vx,
        };
        for (int k = 0; k < 4; k++)
          P[k] += a * da[k];
        if (J_p != nullptr) {
          J_p[3 * i + 0] = da[0];
          J_p[3 * i + 1] = da[1];
          J_p[3 * i + 2] = da[2];
        }
        if (J_v != nullptr)
          J_v[i] = da[3];
      }
    }

    const double inv_sqrt_S = 1.0 / std::sqrt(S);
    double P_by_S[4];
    for (int k = 0; k < 4; k++)
      P_by_S[k] = P[k] / S;

    for (int y = 0; y < n; y++) {
      for (int x = 0; x < n; x++) {
        const int i = x + n * y;
        const double a = residuals[i];
        if (J_p != nullptr) {
          J_p[3 * i + 0] = (J_p[3 * i + 0] - a * P_by_S[0]) * inv_sqrt_S;
          J_p[3 * i + 1] = (J_p[3 * i + 1] - a * P_by_S[1]) * inv_sqrt_S;
          J_p[3 * i + 2] = (J_p[3 * i + 2] - a * P_by_S[2]) * inv_sqrt_S;
        }
        if (J_v != nullptr)
          J_v[i] = (J_v[i] - a * P_by_S[3]) * inv_sqrt_S;
        if (kEventsOnce)
          residuals[i] = a * inv_sqrt_S + event_term_[i];
        else
          residuals[i] = a * inv_sqrt_S + c.event_frame_->at<double>(y, x) / c.norm_event_frame_;
      }
    }
    return true;
  }

 private:
  std::shared_ptr<CostFunctionConfig> config_;
  Sampler sample_;
  double event_term_[kN];
};

// Copy of kernel::costKernel (double), with the interpolation passed as sample. kEventsPerCall: the residual takes
// event_frame(i / 19, i % 19) / norm at every call, as ErrorRotation does, instead of the precomputed event_term.
template <bool kEventsPerCall, typename Sampler>
int costKernelVariant(const Sampler &sample, const double event_term[kN], const cv::Mat *event_frame, double norm,
                      double origin_x, double origin_y, double theta, double tx, double ty, double phi,
                      bool with_jacobian, double residuals[kN], double jacobian[kN][kernel::kNumParams]) {
  using T = double;
  constexpr int kNumParams = kernel::kNumParams;
  constexpr int kPatchSize = kernel::kPatchSize;
  const T vx = std::cos(phi), vy = std::sin(phi);
  const T a0 = std::cos(theta), a1 = -std::sin(theta);
  const T a2 = -a1, a3 = a0;
  const T ox = origin_x, oy = origin_y;

  T a_pix[kN];
  T da_pix[kN][kNumParams];

  T S = T(1e-3);
  T P[kNumParams] = {T(0), T(0), T(0), T(0)};
  int outside = 0;
  for (int y = 0; y < kPatchSize; ++y) {
    for (int x = 0; x < kPatchSize; ++x) {
      const int i = x + kPatchSize * y;
      const T x_p = a0 * T(x) + a1 * T(y) + a0 * ox + a1 * oy + tx;
      const T y_p = a2 * T(x) + a3 * T(y) + a2 * ox + a3 * oy + ty;

      T g[2], g_y[2], g_x[2];
      if (!sample(y_p, x_p, g, g_y, g_x, true))
        ++outside;

      const T a = g[0] * vx + g[1] * vy;
      a_pix[i] = a;
      S += a * a;
      if (!with_jacobian)
        continue;

      const T a_x = g_x[0] * vx + g_x[1] * vy;
      const T a_y = g_y[0] * vx + g_y[1] * vy;
      da_pix[i][0] = -(y_p - ty) * a_x + (x_p - tx) * a_y;
      da_pix[i][1] = a_x;
      da_pix[i][2] = a_y;
      da_pix[i][3] = -g[0] * vy + g[1] * vx;
      for (int k = 0; k < kNumParams; ++k)
        P[k] += a * da_pix[i][k];
    }
  }

  const T inv_sqrt_S = T(1.0) / std::sqrt(S);
  T P_by_S[kNumParams];
  for (int k = 0; k < kNumParams; ++k)
    P_by_S[k] = P[k] / S;

  for (int i = 0; i < kN; ++i) {
    const T a = a_pix[i];
    if (with_jacobian)
      for (int k = 0; k < kNumParams; ++k)
        jacobian[i][k] = (da_pix[i][k] - a * P_by_S[k]) * inv_sqrt_S;
    if (kEventsPerCall)
      residuals[i] = a * inv_sqrt_S + event_frame->at<double>(i / kPatchSize, i % kPatchSize) / norm;
    else
      residuals[i] = a * inv_sqrt_S + event_term[i];
  }
  return outside;
}

// Copy of ErrorRotationKernel<double>. kCeres: the kernel reads the gradient through Ceres' interpolator (interp)
// instead of its own window. kEventsPerCall: see costKernelVariant.
template <bool kCeres, bool kEventsPerCall>
class KernelVariant : public ceres::CostFunction {
 public:
  KernelVariant(const cv::Point2d &feature, const cv::Point2d &init_feature, const cv::Mat &event_frame,
                const Grid &grad_grid, const Interpolator *interp)
      : interp_(interp), event_frame_(&event_frame) {
    set_num_residuals(kN);
    mutable_parameter_block_sizes()->push_back(3);
    mutable_parameter_block_sizes()->push_back(1);
    row0_ = static_cast<int>(std::lround(init_feature.y)) - kW / 2;
    col0_ = static_cast<int>(std::lround(init_feature.x)) - kW / 2;
    for (int wy = 0; wy < kW; ++wy)
      for (int wx = 0; wx < kW; ++wx)
        grad_grid.GetValue(row0_ + wy, col0_ + wx, window_[wy][wx]);
    norm_ = cv::norm(event_frame);
    for (int y = 0; y < kernel::kPatchSize; ++y)
      for (int x = 0; x < kernel::kPatchSize; ++x)
        event_term_[x + kernel::kPatchSize * y] = event_frame.at<double>(y, x) / norm_;
    origin_x_ = feature.x - kernel::kHalfSize;
    origin_y_ = feature.y - kernel::kHalfSize;
  }

  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
    double *J_p = jacobians != nullptr ? jacobians[0] : nullptr;
    double *J_v = jacobians != nullptr ? jacobians[1] : nullptr;
    const bool with_jacobian = J_p != nullptr || J_v != nullptr;

    double r[kN];
    double J[kN][kernel::kNumParams];
    const double th = parameters[0][0], tx = parameters[0][1], ty = parameters[0][2], phi = parameters[1][0];
    if (kCeres)
      costKernelVariant<kEventsPerCall>(CeresSampler{interp_}, event_term_, event_frame_, norm_, origin_x_, origin_y_,
                                        th, tx, ty, phi, with_jacobian, r, J);
    else
      costKernelVariant<kEventsPerCall>(WindowSampler{window_, row0_, col0_}, event_term_, event_frame_, norm_,
                                        origin_x_, origin_y_, th, tx, ty, phi, with_jacobian, r, J);

    for (int i = 0; i < kN; ++i) {
      residuals[i] = r[i];
      if (J_p != nullptr)
        for (int k = 0; k < 3; ++k)
          J_p[3 * i + k] = J[i][k];
      if (J_v != nullptr)
        J_v[i] = J[i][3];
    }
    return true;
  }

 private:
  Window window_;
  int row0_, col0_;
  const Interpolator *interp_;
  const cv::Mat *event_frame_;
  double norm_;
  double event_term_[kN];
  double origin_x_, origin_y_;
};

}  // namespace timing
}  // namespace x
