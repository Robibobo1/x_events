// Validation level 1 of the EKLT cost function on the golden vectors (optimizer_dump.bin, format in
// x/eklt/optimizer_dump.h). For every recorded solve, the inputs of the cost function are rebuilt from the record
// alone, then the original cost function (ErrorRotation, Ceres' automatic derivatives) and the one with hand-written
// derivatives (ErrorRotationAnalytic) are evaluated and compared with the outputs the original code stored. The cost
// kernel for HLS (x/eklt/hls/cost_kernel.h, through its Ceres wrapper ErrorRotationKernel) is compared with the
// original in the same way, built in double (same numbers expected, up to rounding) and in float.
//
//   eklt_cost_check <optimizer_dump.bin>... [--tolerance 1e-9] [--kernel-tolerance 2e-4]
//
// Relative difference of a vector (the residuals, or one column of the Jacobian): largest absolute difference divided
// by the largest absolute reference value. Of a cost: absolute difference divided by the reference cost.
// Returns 0 if every relative difference of the hand-written version and of the kernel in double stays below the
// tolerance, every one of the kernel in float below the kernel tolerance, and the kernel never reads outside its
// window.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>

#include <x/eklt/error.h>
#include <x/eklt/error_analytic.h>
#include <x/eklt/error_kernel.h>
#include <x/eklt/optimizer_dump.h>

using namespace x;

namespace {

constexpr int N = dump::kNumPixels;
constexpr int K = dump::kNumParams;
const char *kColumn[K] = {"d/dtheta", "d/dtx", "d/dty", "d/dphi"};

// Evaluates a cost function at params = (theta, tx, ty, phi). jacobian (N x 4, row major) null: residuals only.
void evaluate(const ceres::CostFunction &f, const double params[K], double *residuals, double *jacobian) {
  const double p[3] = {params[0], params[1], params[2]};
  const double v[1] = {params[3]};
  const double *parameters[] = {p, v};
  if (jacobian == nullptr) {
    f.Evaluate(parameters, residuals, nullptr);
    return;
  }
  std::vector<double> J_p(N * 3), J_v(N);
  double *jacobians[] = {J_p.data(), J_v.data()};
  f.Evaluate(parameters, residuals, jacobians);
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < 3; ++k)
      jacobian[K * i + k] = J_p[3 * i + k];
    jacobian[K * i + 3] = J_v[i];
  }
}

double cost(const double *r) {
  double s = 0;
  for (int i = 0; i < N; ++i)
    s += r[i] * r[i];
  return 0.5 * s;
}

// largest |a - b| / largest |b|, over n values taken every stride
double relDiff(const double *a, const double *b, int n, int stride = 1) {
  double d = 0, m = 0;
  for (int i = 0; i < n; ++i) {
    d = std::max(d, std::abs(a[i * stride] - b[i * stride]));
    m = std::max(m, std::abs(b[i * stride]));
  }
  return m > 0 ? d / m : d;
}

double relDiffCost(double a, double b) { return std::abs(a - b) / std::abs(b); }

// share of values that are equal to the last bit
struct Identical {
  long same = 0, total = 0;
  void add(const double *a, const double *b, int n) {
    for (int i = 0; i < n; ++i)
      same += a[i] == b[i];
    total += n;
  }
  double percent() const { return total > 0 ? 100.0 * same / total : 0; }
};

// relative differences of one quantity over all records or calls
struct Stat {
  std::vector<double> v;
  void add(double x) { v.push_back(x); }
  double max() const { return v.empty() ? 0 : *std::max_element(v.begin(), v.end()); }
  double median() const {
    if (v.empty())
      return 0;
    std::vector<double> s(v);
    std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
    return s[s.size() / 2];
  }
};

void print(const char *name, const Stat &s) {
  std::printf("    %-12s %9.1e  %9.1e\n", name, s.median(), s.max());
}

// one version of the cost kernel against the original, at every recorded call
struct KernelStats {
  const char *name;
  Stat r, J[K], cost;
  KernelWindowCounter window;  // over every evaluation of this version, two per recorded call

  double worst() const {
    double w = std::max(r.max(), cost.max());
    for (int k = 0; k < K; ++k)
      w = std::max(w, J[k].max());
    return w;
  }

  void printAll(long calls) const {
    std::printf("  Kernel in %s vs original derivatives, at every recorded call (%ld points)\n", name, calls);
    print("residuals", r);
    for (int k = 0; k < K; ++k)
      print(kColumn[k], J[k]);
    std::printf("  Kernel in %s, cost of every call vs stored (with or without Jacobian, as Ceres asked)\n", name);
    print("cost", cost);
    std::printf("    pixels read outside the %dx%d window: %ld, in %ld of %ld evaluations\n", kernel::kWindowSize,
                kernel::kWindowSize, window.pixels_outside, window.calls_outside, window.calls);
  }
};

// Returns true if the hand-written derivatives and the kernel pass on every record of the file
bool check(const std::string &path, double tolerance, double kernel_tolerance) {
  std::ifstream in(path, std::ios::binary);
  dump::Header h{};
  if (!in.read(reinterpret_cast<char *>(&h), sizeof(h)) || std::memcmp(h.magic, "EKLTDUMP", 8) != 0 ||
      h.version != dump::kVersion || h.record_size != sizeof(dump::Record)) {
    std::cerr << path << ": not an optimizer dump of version " << dump::kVersion << std::endl;
    return false;
  }
  in.seekg(h.header_size);

  // check of the checker: the original cost function on the rebuilt inputs must give exactly the stored outputs
  Stat self_r, self_J, self_cost;
  // hand-written derivatives against the stored values, at the start and the end of each solve
  Stat end_r, end_J[K];
  Identical end_same_r, end_same_J;
  // hand-written against the original, both evaluated with the Jacobian at every recorded call
  Stat call_r, call_J[K];
  // hand-written, evaluated as Ceres asked (with or without Jacobian) at every recorded call, against the stored cost
  Stat call_cost;
  // the cost kernel in double and in float, against the original at every recorded call
  KernelStats kernel_double{"double"}, kernel_float{"float"};

  auto rec = std::make_unique<dump::Record>();
  long records = 0, calls = 0;
  std::vector<double> r_orig(N), J_orig(N * K), r_an(N), J_an(N * K), r_k(N), J_k(N * K);

  while (in.read(reinterpret_cast<char *>(rec.get()), sizeof(dump::Record))) {
    const dump::Record &r = *rec;
    ++records;

    // inputs of the cost function: the gradient window placed at its image position (same values for the
    // interpolator as the full image, ceres::Grid2D clamps outside the window as it did outside the image), the event
    // frame, the patch center
    Grid grid(r.grad_window, r.window_row0, r.window_row0 + dump::kWindowSize,
              r.window_col0, r.window_col0 + dump::kWindowSize);
    Interpolator interp(grid);
    cv::Mat event_frame(dump::kPatchSize, dump::kPatchSize, CV_64F, const_cast<double *>(r.event_frame));
    const cv::Point2d center(r.center[0], r.center[1]), init_center(r.init_center[0], r.init_center[1]);
    std::unique_ptr<ceres::CostFunction> original(Generator::Create(center, init_center, &event_frame, &interp));
    std::unique_ptr<ceres::CostFunction> analytic(
        ErrorRotationAnalytic::Create(center, init_center, &event_frame, &interp));
    // the kernel's 40x40 window is copied from this grid, it lies inside the stored 48x48 window
    std::unique_ptr<ceres::CostFunction> kernel_d(
        ErrorRotationKernel<double>::Create(center, init_center, &event_frame, grid, &kernel_double.window));
    std::unique_ptr<ceres::CostFunction> kernel_f(
        ErrorRotationKernel<float>::Create(center, init_center, &event_frame, grid, &kernel_float.window));
    const std::pair<const ceres::CostFunction *, KernelStats *> kernels[] = {{kernel_d.get(), &kernel_double},
                                                                             {kernel_f.get(), &kernel_float}};

    // start and end of the solve, where the record holds the residuals and the Jacobian
    const double *params[2] = {r.params_init, r.params_final};
    const double *r_ref[2] = {r.residuals_init, r.residuals_final};
    const double *J_ref[2] = {r.jacobian_init, r.jacobian_final};
    for (int s = 0; s < 2; ++s) {
      evaluate(*original, params[s], r_orig.data(), J_orig.data());
      self_r.add(relDiff(r_orig.data(), r_ref[s], N));
      self_J.add(relDiff(J_orig.data(), J_ref[s], N * K));

      evaluate(*analytic, params[s], r_an.data(), J_an.data());
      end_r.add(relDiff(r_an.data(), r_ref[s], N));
      for (int k = 0; k < K; ++k)
        end_J[k].add(relDiff(J_an.data() + k, J_ref[s] + k, N, K));
      end_same_r.add(r_an.data(), r_ref[s], N);
      end_same_J.add(J_an.data(), J_ref[s], N * K);
    }

    // every recorded call of the solve
    const int m = std::min<int>(r.num_evals, dump::kMaxEvals);
    for (int e = 0; e < m; ++e) {
      const dump::Evaluation &ev = r.evals[e];
      ++calls;
      const bool with_jacobian = ev.with_jacobian > 0;

      evaluate(*original, ev.params, r_orig.data(), with_jacobian ? J_orig.data() : nullptr);
      self_cost.add(relDiffCost(cost(r_orig.data()), ev.cost));
      evaluate(*analytic, ev.params, r_an.data(), with_jacobian ? J_an.data() : nullptr);
      call_cost.add(relDiffCost(cost(r_an.data()), ev.cost));

      evaluate(*original, ev.params, r_orig.data(), J_orig.data());
      evaluate(*analytic, ev.params, r_an.data(), J_an.data());
      call_r.add(relDiff(r_an.data(), r_orig.data(), N));
      for (int k = 0; k < K; ++k)
        call_J[k].add(relDiff(J_an.data() + k, J_orig.data() + k, N, K));

      // the kernel: cost as Ceres asked, then residuals and Jacobian against the original's (r_orig, J_orig above)
      for (const auto &version : kernels) {
        const ceres::CostFunction &f = *version.first;
        KernelStats &s = *version.second;
        evaluate(f, ev.params, r_k.data(), with_jacobian ? J_k.data() : nullptr);
        s.cost.add(relDiffCost(cost(r_k.data()), ev.cost));
        evaluate(f, ev.params, r_k.data(), J_k.data());
        s.r.add(relDiff(r_k.data(), r_orig.data(), N));
        for (int k = 0; k < K; ++k)
          s.J[k].add(relDiff(J_k.data() + k, J_orig.data() + k, N, K));
      }
    }
  }

  std::printf("%s\n%ld records, %ld cost function calls\n", path.c_str(), records, calls);
  std::printf("\nRelative difference                      median        max\n");
  std::printf("  Check of the checker: original cost function on the rebuilt inputs vs stored (must be 0)\n");
  print("residuals", self_r);
  print("Jacobian", self_J);
  print("cost", self_cost);
  std::printf("  Hand-written derivatives vs stored, at the start and end of each solve (%ld points)\n", 2 * records);
  print("residuals", end_r);
  for (int k = 0; k < K; ++k)
    print(kColumn[k], end_J[k]);
  std::printf("    identical to the last bit: residuals %.1f%%, Jacobian %.1f%%\n", end_same_r.percent(),
              end_same_J.percent());
  std::printf("  Hand-written vs original derivatives, at every recorded call (%ld points)\n", calls);
  print("residuals", call_r);
  for (int k = 0; k < K; ++k)
    print(kColumn[k], call_J[k]);
  std::printf("  Hand-written, cost of every call vs stored (with or without Jacobian, as Ceres asked)\n");
  print("cost", call_cost);
  kernel_double.printAll(calls);
  kernel_float.printAll(calls);

  double worst = std::max({end_r.max(), call_r.max(), call_cost.max()});
  for (int k = 0; k < K; ++k)
    worst = std::max({worst, end_J[k].max(), call_J[k].max()});
  const bool pass_analytic = records > 0 && worst < tolerance;
  const bool pass_double = records > 0 && kernel_double.worst() < tolerance;
  const bool pass_float = records > 0 && kernel_float.worst() < kernel_tolerance;
  const bool pass_window = kernel_double.window.pixels_outside == 0 && kernel_float.window.pixels_outside == 0;
  std::printf("\n%s: largest relative difference of the hand-written derivatives %.1e, tolerance %.0e\n",
              pass_analytic ? "PASS" : "FAIL", worst, tolerance);
  std::printf("%s: largest relative difference of the kernel in double %.1e, tolerance %.0e\n",
              pass_double ? "PASS" : "FAIL", kernel_double.worst(), tolerance);
  std::printf("%s: largest relative difference of the kernel in float %.1e, tolerance %.0e\n",
              pass_float ? "PASS" : "FAIL", kernel_float.worst(), kernel_tolerance);
  std::printf("%s: pixels read outside the kernel window %ld\n\n", pass_window ? "PASS" : "FAIL",
              kernel_double.window.pixels_outside + kernel_float.window.pixels_outside);
  return pass_analytic && pass_double && pass_float && pass_window;
}

}  // namespace

int main(int argc, char **argv) {
  std::vector<std::string> paths;
  double tolerance = 1e-9, kernel_tolerance = 2e-4;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--tolerance" && i + 1 < argc)
      tolerance = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--kernel-tolerance" && i + 1 < argc)
      kernel_tolerance = std::atof(argv[++i]);
    else
      paths.emplace_back(argv[i]);
  }
  if (paths.empty()) {
    std::cerr << "usage: eklt_cost_check <optimizer_dump.bin>... [--tolerance 1e-9] [--kernel-tolerance 2e-4]"
              << std::endl;
    return 2;
  }
  bool pass = true;
  for (const auto &p : paths)
    pass = check(p, tolerance, kernel_tolerance) && pass;
  return pass ? 0 : 1;
}
