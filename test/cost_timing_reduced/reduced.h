#pragma once
// Types of the reduced copy of ErrorRotationAnalytic::Evaluate (src/x/eklt/error_analytic.cpp): OpenCV removed, so
// that any compiler can build it with Ceres 1.14's interpolator and see how it inlines it. Build: see driver.cpp
#include <cmath>
#include <ceres/cubic_interpolation.h>
#include <x/eklt/hls/cost_kernel.h>
using Grid = ceres::Grid2D<double, 2>;
using Interpolator = ceres::BiCubicInterpolator<Grid>;
struct Point { double x, y; };
struct Config { Point feature_; int patch_size_, half_size_; const double *event_frame_; double norm_event_frame_; Interpolator *grad_interp_;
                const double (*window)[x::kernel::kWindowSize][2]; int row0, col0; };
bool evaluateAnalyticKernelInterp(const Config &c, double const *const *parameters, double *residuals, double **jacobians);
bool evaluateAnalytic(const Config &c, double const *const *parameters, double *residuals, double **jacobians);
