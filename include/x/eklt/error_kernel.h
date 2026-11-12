#pragma once

#include <cmath>
#include <stdexcept>

#include <ceres/ceres.h>
#include <opencv2/core/core.hpp>

#include <x/eklt/hls/cost_kernel.h>
#include <x/eklt/types.h>


namespace x {

/**
 * @brief How often the cost kernel read pixels outside its window, summed over every wrapper that shares it.
 * Not thread safe: the EKLT optimizer runs Ceres with one thread.
 */
struct KernelWindowCounter {
  long calls = 0;           // cost function calls
  long calls_outside = 0;   // calls with at least one pixel outside the window
  long pixels_outside = 0;  // patch pixels whose 4x4 interpolation cells were not all inside the window
};

/**
 * @brief Ceres wrapper around the cost kernel (x/eklt/hls/cost_kernel.h), with the same residuals, Jacobian and
 * parameter blocks as ErrorRotation and ErrorRotationAnalytic: (theta, tx, ty) and (flow angle phi). It plays the part
 * that the CPU driver code would play with the kernel on the FPGA.
 *
 * Once per solve (constructor): copies the window of the template gradient out of the image and normalizes the event
 * frame, both in double, then converts them to T.
 * At every call (Evaluate): converts the parameters to T, runs the kernel, converts the residuals and the Jacobian back
 * to double and splits the Jacobian into Ceres' two blocks.
 *
 * T: float in EKLT-VIO (EKLT_COST=kernel), double only to check the kernel against the original.
 */
template <typename T>
class ErrorRotationKernel : public ceres::CostFunction {
 public:
  // feature: patch center in the current image. init_feature: feature position in the template frame, where the
  // window is centered. grad_grid: the template gradient image. counter: may be null.
  ErrorRotationKernel(const cv::Point2d &feature, const cv::Point2d &init_feature, const cv::Mat &event_frame,
                      const Grid &grad_grid, KernelWindowCounter *counter)
      : counter_(counter) {
    if (event_frame.rows != kernel::kPatchSize || event_frame.cols != kernel::kPatchSize ||
        event_frame.type() != CV_64F)
      throw std::invalid_argument("ErrorRotationKernel: the event frame must be 19x19 doubles (eklt_patch_size 19)");
    set_num_residuals(kernel::kNumPixels);
    mutable_parameter_block_sizes()->push_back(3);  // theta, tx, ty
    mutable_parameter_block_sizes()->push_back(1);  // flow angle phi

    // window centered on the feature position in the template frame, placed as in the optimizer dump. Outside the
    // image, Grid2D::GetValue repeats the nearest border pixel, as it does for the original interpolator.
    row0_ = static_cast<int>(std::lround(init_feature.y)) - kernel::kWindowSize / 2;
    col0_ = static_cast<int>(std::lround(init_feature.x)) - kernel::kWindowSize / 2;
    for (int wy = 0; wy < kernel::kWindowSize; ++wy) {
      for (int wx = 0; wx < kernel::kWindowSize; ++wx) {
        double g[2];
        grad_grid.GetValue(row0_ + wy, col0_ + wx, g);
        window_[wy][wx][0] = static_cast<T>(g[0]);
        window_[wy][wx][1] = static_cast<T>(g[1]);
      }
    }

    // event term e_i / |e|, as in ErrorRotation: the norm is computed once per solve, in double
    const double norm = cv::norm(event_frame);
    for (int y = 0; y < kernel::kPatchSize; ++y)
      for (int x = 0; x < kernel::kPatchSize; ++x)
        event_term_[x + kernel::kPatchSize * y] = static_cast<T>(event_frame.at<double>(y, x) / norm);

    // image position of patch pixel (0, 0)
    origin_x_ = static_cast<T>(feature.x - kernel::kHalfSize);
    origin_y_ = static_cast<T>(feature.y - kernel::kHalfSize);
  }

  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
    // Ceres asks for the residuals only (jacobians == nullptr), or also for the Jacobian of one or both parameter
    // blocks, row major: J_p[3 * i + 0, 1, 2] = d r_i / d(theta, tx, ty), J_v[i] = d r_i / d phi
    double *J_p = jacobians != nullptr ? jacobians[0] : nullptr;
    double *J_v = jacobians != nullptr ? jacobians[1] : nullptr;
    const bool with_jacobian = J_p != nullptr || J_v != nullptr;

    T r[kernel::kNumPixels];
    T J[kernel::kNumPixels][kernel::kNumParams];
    const int outside = kernel::costKernel<T>(window_, row0_, col0_, event_term_, origin_x_, origin_y_,
                                              static_cast<T>(parameters[0][0]), static_cast<T>(parameters[0][1]),
                                              static_cast<T>(parameters[0][2]), static_cast<T>(parameters[1][0]),
                                              with_jacobian, r, J);

    for (int i = 0; i < kernel::kNumPixels; ++i) {
      residuals[i] = static_cast<double>(r[i]);
      if (J_p != nullptr)
        for (int k = 0; k < 3; ++k)
          J_p[3 * i + k] = static_cast<double>(J[i][k]);
      if (J_v != nullptr)
        J_v[i] = static_cast<double>(J[i][3]);
    }

    if (counter_ != nullptr) {
      ++counter_->calls;
      if (outside > 0)
        ++counter_->calls_outside;
      counter_->pixels_outside += outside;
    }
    return true;
  }

  // ceres takes ownership of the returned object
  static ceres::CostFunction *Create(const cv::Point2d &feature, const cv::Point2d &init_feature,
                                     const cv::Mat *event_frame, const Grid &grad_grid,
                                     KernelWindowCounter *counter) {
    return new ErrorRotationKernel<T>(feature, init_feature, *event_frame, grad_grid, counter);
  }

 private:
  T window_[kernel::kWindowSize][kernel::kWindowSize][2];  // template gradient (Ix, Iy), [0][0] = image (row0_, col0_)
  int row0_, col0_;
  T event_term_[kernel::kNumPixels];
  T origin_x_, origin_y_;
  KernelWindowCounter *counter_;
};

}  // namespace x
