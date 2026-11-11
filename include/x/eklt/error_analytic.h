#pragma once

#include <memory>

#include <ceres/ceres.h>

#include <x/eklt/error.h>


namespace x {

/**
 * @brief The cost function of ErrorRotation (equation (7) of the paper) with hand-written derivatives instead of
 * Ceres' automatic differentiation. Same residuals, same bicubic interpolation, same parameter blocks: (theta, tx, ty)
 * and (flow angle phi).
 *
 * For pixel i, with q the position where the template gradient g is read, v = (cos phi, sin phi) and a = g . v:
 *
 *   r_i  = a_i / sqrt(S) + e_i / |e|,          S = 1e-3 + sum_j a_j^2
 *   J_ik = (da_i/dk - a_i * P_k / S) / sqrt(S), P_k = sum_j a_j * da_j/dk
 *
 * S and the four sums P_k need every pixel, hence two passes over the patch:
 *   1. per pixel: q, g and its slopes, a and its 4 derivatives, accumulate S and P
 *   2. per pixel: residual and Jacobian row
 */
class ErrorRotationAnalytic : public ceres::CostFunction {
 public:
  explicit ErrorRotationAnalytic(std::shared_ptr<CostFunctionConfig> config);

  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

  // same arguments as Generator::Create, ceres takes ownership of the returned object
  static ceres::CostFunction *Create(cv::Point2d feature,
                                     cv::Point2d init_feature,
                                     const cv::Mat *event_frame,
                                     Interpolator *grad_interp);

 private:
  std::shared_ptr<CostFunctionConfig> config_;
};

}  // namespace x
