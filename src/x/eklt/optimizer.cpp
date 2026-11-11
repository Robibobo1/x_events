
#include <x/eklt/optimizer.h>
#include <easy/profiler.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <x/eklt/error_analytic.h>
#include <x/eklt/utils.h>


using namespace x;

namespace {

/**
 * @brief Passes every call through to the original cost function, unchanged, and notes the parameters and the cost
 * of each evaluation. Used only for the solves written to the optimizer dump (EKLT_DUMP_EVERY).
 */
class RecordingCostFunction : public ceres::CostFunction {
 public:
  RecordingCostFunction(ceres::CostFunction *original, std::vector<dump::Evaluation> *evals)
    : original_(original), evals_(evals) {
    set_num_residuals(original_->num_residuals());
    *mutable_parameter_block_sizes() = original_->parameter_block_sizes();
  }

  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
    const bool ok = original_->Evaluate(parameters, residuals, jacobians);

    dump::Evaluation e{};
    e.params[0] = parameters[0][0];
    e.params[1] = parameters[0][1];
    e.params[2] = parameters[0][2];
    e.params[3] = parameters[1][0];
    e.with_jacobian = jacobians != nullptr ? 1.0 : 0.0;
    double sum_sq = 0;
    for (int i = 0; i < num_residuals(); ++i)
      sum_sq += residuals[i] * residuals[i];
    e.cost = 0.5 * sum_sq;
    evals_->push_back(e);
    return ok;
  }

 private:
  std::unique_ptr<ceres::CostFunction> original_;  // owned, deleted with this wrapper by ceres::Problem
  std::vector<dump::Evaluation> *evals_;
};

/**
 * @brief Evaluates the original cost function once and writes its residuals and its Jacobian as one row per residual:
 * d/dtheta, d/dtx, d/dty, d/dphi.
 */
void evaluateReference(const ceres::CostFunction &cost_function, const double p[3], const double v[1],
                       double *residuals, double *jacobian) {
  std::vector<double> J_p(dump::kNumPixels * 3), J_v(dump::kNumPixels);
  const double *parameters[] = {p, v};
  double *jacobians[] = {J_p.data(), J_v.data()};
  cost_function.Evaluate(parameters, residuals, jacobians);
  for (int i = 0; i < dump::kNumPixels; ++i) {
    jacobian[i * 4 + 0] = J_p[i * 3 + 0];
    jacobian[i * 4 + 1] = J_p[i * 3 + 1];
    jacobian[i * 4 + 2] = J_p[i * 3 + 2];
    jacobian[i * 4 + 3] = J_v[i];
  }
}

}  // namespace

Optimizer::Optimizer(Params params, EkltPerformanceLoggerPtr perf_logger)
  : params_(std::move(params)), perf_logger_(std::move(perf_logger)), patch_size_(params_.eklt_patch_size) {
  // EDIT: let ceres take ownership of these objects function to avoid leaks / double frees
//  prob_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
//  prob_options.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
//  prob_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

  solver_options.minimizer_progress_to_stdout = false;
  solver_options.num_threads = 1;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.logging_type = ceres::SILENT;
  solver_options.max_num_iterations = params_.eklt_max_num_iterations;
  solver_options.use_nonmonotonic_steps = true;

  if (const char *every = std::getenv("EKLT_DUMP_EVERY"))
    dump_every_ = std::max(0, std::atoi(every));
  if (dump_every_ > 0)
    std::cout << "EKLT optimizer dump: every " << dump_every_ << "-th solve" << std::endl;

  if (const char *cost = std::getenv("EKLT_COST")) {
    const std::string name(cost);
    if (name == "analytic")
      analytic_cost_ = true;
    else if (name != "autodiff")  // stop rather than silently run the default on a typo
      throw std::invalid_argument("EKLT_COST=" + name + ": expected autodiff or analytic");
    std::cout << "EKLT cost function derivatives: " << name << std::endl;
  }

  if (const char *perturb = std::getenv("EKLT_PERTURB")) {
    const std::string dir(perturb);
    if (dir == "up")
      perturb_ = 1;
    else if (dir == "down")
      perturb_ = -1;
    else
      throw std::invalid_argument("EKLT_PERTURB=" + dir + ": expected up or down");
    std::cout << "EKLT optimizer: starting values of every solve moved one last bit " << dir << std::endl;
  }
}


void Optimizer::setParams(const Params &params) {
  params_ = params;
}

Optimizer::~Optimizer() {
  for (auto &it : optimizer_data_)
    it.second.clear();
}

void Optimizer::decrementCounter(double time) {
  // keep track of reference counter to release gradient data
  if (optimizer_data_.find(time) == optimizer_data_.end())
    return;

  --optimizer_data_[time].ref_counter_;
  if (optimizer_data_[time].ref_counter_ == 0) {
    optimizer_data_[time].clear();
    optimizer_data_.erase(time);
  }
}

void Optimizer::getLogGradients(const cv::Mat &img, cv::Mat &I_x, cv::Mat &I_y) {
  computeLogImgGradients(img, I_x, I_y, params_.eklt_log_eps, params_.eklt_use_linlog_scale);
}

void Optimizer::precomputeLogImageArray(const EkltPatches &patches, const ImageBuffer::iterator &image_it) {
  cv::Mat I_x, I_y;
  getLogGradients(image_it->second, I_x, I_y);

  std::vector<double> grad;

  // initialize grid that is used by ceres interpolator
  for (int row = 0; row < image_it->second.rows; row++) {
    for (int col = 0; col < image_it->second.cols; col++) {
      grad.push_back(I_x.at<double>(row, col));
      grad.push_back(I_y.at<double>(row, col));
    }
  }

  // store this for later use
  double t = image_it->first;
  optimizer_data_[t] = OptimizerDatum(grad, image_it->second, patches.size());
}

void Optimizer::optimizeParameters(const cv::Mat &event_frame, EkltPatch &patch, double t) {
  auto start = profiler::now();
//  EASY_FUNCTION();

  ceres::Problem problem(prob_options);
  ceres::Solver::Summary summary;

  // for now 3 free parameters for warp, x, y translation and theta rotation
  double p0[3];
  ErrorRotation::getP0(p0, patch.warping_);
  double v0[] = {patch.flow_angle_};

  // rounding control: move the starting values by one last bit (next representable double up or down)
  if (perturb_ != 0) {
    const double to = perturb_ > 0 ? HUGE_VAL : -HUGE_VAL;
    for (double &p : p0)
      p = std::nextafter(p, to);
    v0[0] = std::nextafter(v0[0], to);
  }

  // create cost functor that depends on the event frame, and current and initial
  // patch location
  ceres::CostFunction *cost_function;
  if (analytic_cost_)
    cost_function = ErrorRotationAnalytic::Create(patch.getCenter(),
                                                  patch.init_center_,
                                                  &event_frame,
                                                  optimizer_data_[patch.t_init_].grad_interp_);
  else
    cost_function = Generator::Create(patch.getCenter(),
                                      patch.init_center_,
                                      &event_frame,
                                      optimizer_data_[patch.t_init_].grad_interp_);

  // optional dump (golden vectors): wrap the cost function to note every evaluation of this solve
  std::unique_ptr<dump::Record> dump_record;
  std::vector<dump::Evaluation> dump_evals;
  const ceres::CostFunction *original_cost_function = cost_function;
  if (dump_every_ > 0 && perf_logger_ && solve_index_ % dump_every_ == 0) {
    if (event_frame.rows != dump::kPatchSize || event_frame.cols != dump::kPatchSize) {
      std::cerr << "EKLT optimizer dump disabled: it needs eklt_patch_size = " << dump::kPatchSize << std::endl;
      dump_every_ = 0;
    } else {
      dump_record = startDumpRecord(event_frame, patch, t, p0, v0);
      cost_function = new RecordingCostFunction(cost_function, &dump_evals);
    }
  }
  ++solve_index_;

  problem.AddResidualBlock(cost_function, nullptr, p0, v0);

  auto solve_start = profiler::now();
  ceres::Solve(solver_options, &problem, &summary);
  auto solve_stop = profiler::now();

  if (dump_record)
    finishDumpRecord(*dump_record, *original_cost_function, p0, v0, summary, dump_evals);

  // update patch according to new optimizer
  cv::Mat camera_warp;
  ErrorRotation::getWarp(p0, camera_warp);

  // update state of patch
  patch.warping_ = camera_warp.clone();
  patch.flow_angle_ = fmod(v0[0], 2 * M_PI);

  //remap tracking cost to 0-1 (1 is best and 0 is bad)
  patch.tracking_quality_ = 1 - summary.final_cost / 2;
  patch.updateCenter(t);

  if (perf_logger_)
    perf_logger_->optimizations_csv.addRow(start, profiler::now(), summary.iterations.size(), summary.final_cost,
                                           solve_start, solve_stop,
                                           summary.preprocessor_time_in_seconds, summary.minimizer_time_in_seconds,
                                           summary.postprocessor_time_in_seconds, summary.total_time_in_seconds,
                                           summary.jacobian_evaluation_time_in_seconds,
                                           summary.residual_evaluation_time_in_seconds,
                                           summary.linear_solver_time_in_seconds,
                                           summary.num_jacobian_evaluations, summary.num_residual_evaluations,
                                           summary.num_linear_solves);
}

void Optimizer::setPerfLogger(const EkltPerformanceLoggerPtr &perf_logger) {
  perf_logger_ = perf_logger;
}

std::unique_ptr<dump::Record> Optimizer::startDumpRecord(const cv::Mat &event_frame, const EkltPatch &patch, double t,
                                                         const double p0[3], const double v0[1]) {
  auto r = std::make_unique<dump::Record>();
  std::memset(r.get(), 0, sizeof(dump::Record));

  r->solve_index = solve_index_;
  r->patch_id = patch.getId();
  r->t = t;
  r->t_init = patch.t_init_;
  r->center[0] = patch.getCenter().x;
  r->center[1] = patch.getCenter().y;
  r->init_center[0] = patch.init_center_.x;
  r->init_center[1] = patch.init_center_.y;
  r->params_init[0] = p0[0];
  r->params_init[1] = p0[1];
  r->params_init[2] = p0[2];
  r->params_init[3] = v0[0];

  for (int y = 0; y < dump::kPatchSize; ++y)
    for (int x = 0; x < dump::kPatchSize; ++x)
      r->event_frame[y * dump::kPatchSize + x] = event_frame.at<double>(y, x);
  r->event_frame_norm = cv::norm(event_frame);

  // window of the template gradient centered on the feature position in the template frame, where the warped patch
  // is sampled; outside the image the nearest border value is repeated, as ceres::Grid2D does
  const OptimizerDatum &datum = optimizer_data_.at(patch.t_init_);
  r->image_rows = datum.rows_;
  r->image_cols = datum.cols_;
  r->window_row0 = static_cast<int>(std::lround(patch.init_center_.y)) - dump::kWindowSize / 2;
  r->window_col0 = static_cast<int>(std::lround(patch.init_center_.x)) - dump::kWindowSize / 2;
  for (int wy = 0; wy < dump::kWindowSize; ++wy) {
    const int row = std::min(std::max(r->window_row0 + wy, 0), datum.rows_ - 1);
    for (int wx = 0; wx < dump::kWindowSize; ++wx) {
      const int col = std::min(std::max(r->window_col0 + wx, 0), datum.cols_ - 1);
      for (int c = 0; c < 2; ++c)
        r->grad_window[2 * (wy * dump::kWindowSize + wx) + c] = datum.grad_[2 * (row * datum.cols_ + col) + c];
    }
  }
  return r;
}

void Optimizer::finishDumpRecord(dump::Record &r, const ceres::CostFunction &cost_function, const double p[3],
                                 const double v[1], const ceres::Solver::Summary &summary,
                                 const std::vector<dump::Evaluation> &evals) {
  r.params_final[0] = p[0];
  r.params_final[1] = p[1];
  r.params_final[2] = p[2];
  r.params_final[3] = v[0];
  r.initial_cost = summary.initial_cost;
  r.final_cost = summary.final_cost;
  r.num_iterations = static_cast<int32_t>(summary.iterations.size());
  r.termination_type = static_cast<int32_t>(summary.termination_type);
  r.num_jacobian_evaluations = summary.num_jacobian_evaluations;
  r.num_residual_evaluations = summary.num_residual_evaluations;

  r.num_evals = static_cast<int32_t>(evals.size());
  std::copy_n(evals.begin(), std::min<size_t>(evals.size(), dump::kMaxEvals), r.evals);

  // reference outputs of the original cost function at the starting and at the final parameters
  evaluateReference(cost_function, r.params_init, r.params_init + 3, r.residuals_init, r.jacobian_init);
  evaluateReference(cost_function, p, v, r.residuals_final, r.jacobian_final);

  if (!dump_file_.is_open()) {
    const fs::path path = perf_logger_->output_path / "optimizer_dump.bin";
    dump_file_.open(path.string(), std::ios::binary | std::ios::trunc);
    if (!dump_file_) {
      std::cerr << "EKLT optimizer dump: cannot open " << path << ", dump disabled" << std::endl;
      dump_every_ = 0;
      return;
    }
    dump::Header h{};
    std::memcpy(h.magic, "EKLTDUMP", 8);
    h.version = dump::kVersion;
    h.header_size = sizeof(dump::Header);
    h.record_size = sizeof(dump::Record);
    h.patch_size = dump::kPatchSize;
    h.window_size = dump::kWindowSize;
    h.max_evals = dump::kMaxEvals;
    dump_file_.write(reinterpret_cast<const char *>(&h), sizeof(h));
  }
  dump_file_.write(reinterpret_cast<const char *>(&r), sizeof(r));
  dump_file_.flush();  // the program does not always shut down cleanly (see evaluate.cpp), keep every record on disk
}


