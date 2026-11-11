#include <x/eklt/error_analytic.h>

#include <cmath>
#include <utility>


using namespace x;

ErrorRotationAnalytic::ErrorRotationAnalytic(std::shared_ptr<CostFunctionConfig> config) : config_(std::move(config)) {
  set_num_residuals(config_->size_);
  mutable_parameter_block_sizes()->push_back(3);  // theta, tx, ty
  mutable_parameter_block_sizes()->push_back(1);  // flow angle phi
}

ceres::CostFunction *ErrorRotationAnalytic::Create(cv::Point2d feature,
                                                   cv::Point2d init_feature,
                                                   const cv::Mat *event_frame,
                                                   Interpolator *grad_interp) {
  return new ErrorRotationAnalytic(std::make_shared<CostFunctionConfig>(std::move(feature), std::move(init_feature),
                                                                        event_frame, grad_interp));
}

bool ErrorRotationAnalytic::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  const CostFunctionConfig &c = *config_;
  const int n = c.patch_size_;

  const double theta = parameters[0][0], tx = parameters[0][1], ty = parameters[0][2];
  const double phi = parameters[1][0];

  // Ceres asks for the residuals only (jacobians == nullptr), or also for the Jacobian of one or both parameter
  // blocks, row major, one row per residual: J_p[3 * i + 0, 1, 2] = d r_i / d(theta, tx, ty), J_v[i] = d r_i / d phi
  double *J_p = jacobians != nullptr ? jacobians[0] : nullptr;
  double *J_v = jacobians != nullptr ? jacobians[1] : nullptr;
  const bool with_jacobian = J_p != nullptr || J_v != nullptr;

  // motion direction v and rotation R(theta) = [a0 a1, a2 a3], named as in ErrorRotation::warp
  const double vx = std::cos(phi), vy = std::sin(phi);
  const double a0 = std::cos(theta), a1 = -std::sin(theta);
  const double a2 = -a1, a3 = a0;
  // pixel (x, y) of the patch is at u = (x + ox, y + oy) in the current image
  const double ox = c.feature_.x - c.half_size_;
  const double oy = c.feature_.y - c.half_size_;

  // Pass 1: a_i (kept in residuals[i]) and its derivatives (kept in J_p, J_v), sums S and P_k
  double S = 1e-3;
  double P[4] = {0.0, 0.0, 0.0, 0.0};  // sum of a_j * d a_j / d(theta, tx, ty, phi)
  for (int y = 0; y < n; y++) {
    for (int x = 0; x < n; x++) {
      const int i = x + n * y;

      // q = R(theta) u + t, the same expression as in ErrorRotation::warp, so the gradient is read at the same point
      const double x_p = a0 * x + a1 * y + a0 * ox + a1 * oy + tx;
      const double y_p = a2 * x + a3 * y + a2 * ox + a3 * oy + ty;

      // template gradient g = (Ix, Iy) at q, and its slopes along the rows (y) and the columns (x)
      double g[2], g_y[2], g_x[2];
      if (with_jacobian)
        c.grad_interp_->Evaluate(y_p, x_p, g, g_y, g_x);
      else
        c.grad_interp_->Evaluate(y_p, x_p, g, nullptr, nullptr);

      const double a = g[0] * vx + g[1] * vy;
      residuals[i] = a;
      S += a * a;
      if (!with_jacobian)
        continue;

      // how a changes when q moves one pixel right (a_x) or down (a_y)
      const double a_x = g_x[0] * vx + g_x[1] * vy;
      const double a_y = g_y[0] * vx + g_y[1] * vy;
      const double da[4] = {
          -(y_p - ty) * a_x + (x_p - tx) * a_y,  // theta: dq/dtheta = (-(q_y - t_y), q_x - t_x)
          a_x,                                   // tx
          a_y,                                   // ty
          -g[0] * vy + g[1] * vx,                // phi: g . (-sin phi, cos phi)
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

  // Pass 2: normalize. r_i = a_i / sqrt(S) + e_i / |e|, J_ik = (da_i/dk - a_i * P_k / S) / sqrt(S)
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
      residuals[i] = a * inv_sqrt_S + c.event_frame_->at<double>(y, x) / c.norm_event_frame_;
    }
  }
  return true;
}
