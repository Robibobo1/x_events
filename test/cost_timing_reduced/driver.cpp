// Compiler test of the hand-written cost function: times evaluateAnalytic (Ceres' bicubic interpolator) and
// evaluateAnalyticKernelInterp (the cost kernel's interpolation), both in analytic_reduced.cpp, on the calls of an
// optimizer dump, and checks the cost of every call against the recorded one.
//   driver <optimizer_dump.bin> [reps]
//
// analytic_reduced.cpp is ErrorRotationAnalytic::Evaluate without OpenCV, so that it builds with any compiler from
// Ceres 1.14's header-only interpolator, Eigen 3.4 and glog alone. It is compiled separately from the driver (no LTO),
// as error_analytic.cpp is in libx. With GCC 9 it takes the same time as the real function.
//
// Build with each compiler CC (g++-9, g++-10, clang++-12 in the eklt-vio:noetic image after apt-get install g++-10
// clang-12; g++-13, g++-14, clang++-18 in ubuntu:24.04 after apt-get install g++-13 g++-14 clang-18
// libgoogle-glog-dev), from the root of x_events, with CERES = a folder holding Ceres 1.14's include/ceres (in the
// image: /catkin_ws/devel/include, symlinks to resolve with cp -rL when copying it out) and EIGEN = Eigen 3.4:
//   F="-std=c++17 -O3 -DNDEBUG -funroll-loops -fno-math-errno -funsafe-math-optimizations -ffinite-math-only
//      -fno-signed-zeros -Iinclude -I$CERES -I$EIGEN"
//   $CC $F -c test/cost_timing_reduced/analytic_reduced.cpp -o fn.o
//   $CC $F -c test/cost_timing_reduced/driver.cpp -o drv.o
//   $CC drv.o fn.o -lglog -o driver_$CC
// Add -S to the first line to read how the compiler inlined the interpolator.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>
#include <x/eklt/optimizer_dump.h>
#include "reduced.h"
using namespace x;
using Fn = bool (*)(const Config &, double const *const *, double *, double **);
void timeFn(const char *name, Fn fn, const std::vector<std::unique_ptr<dump::Record>> &recs, int reps);
int main(int argc, char **argv) {
  const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
  std::ifstream in(argv[1], std::ios::binary);
  dump::Header h{};
  in.read(reinterpret_cast<char *>(&h), sizeof(h));
  if (h.record_size != sizeof(dump::Record)) { std::printf("bad dump\n"); return 1; }
  in.seekg(h.header_size);
  std::vector<std::unique_ptr<dump::Record>> recs;
  while (true) { auto r = std::make_unique<dump::Record>(); if (!in.read(reinterpret_cast<char *>(r.get()), sizeof(dump::Record))) break; recs.push_back(std::move(r)); }
  timeFn("Ceres interpolator ", evaluateAnalytic, recs, reps);
  timeFn("kernel interpolation", evaluateAnalyticKernelInterp, recs, reps);
  return 0;
}

void timeFn(const char *name, Fn fn, const std::vector<std::unique_ptr<dump::Record>> &recs, int reps) {
  constexpr int N = dump::kNumPixels;
  static double res[N], Jp[N * 3], Jv[N];
  double *jac[2] = {Jp, Jv};
  std::vector<std::vector<float>> times;  // per call, one time per repetition
  std::vector<bool> with_j;
  double worst = 0;
  for (int rep = -1; rep < reps; ++rep) {   // rep -1: warm-up and check, not timed
    size_t k = 0;
    for (auto &rp : recs) {
      const dump::Record &r = *rp;
      Grid grid(r.grad_window, r.window_row0, r.window_row0 + dump::kWindowSize, r.window_col0, r.window_col0 + dump::kWindowSize);
      Interpolator interp(grid);
      const Config c{{r.center[0], r.center[1]}, dump::kPatchSize, (dump::kPatchSize - 1) / 2, r.event_frame, r.event_frame_norm, &interp,
                     reinterpret_cast<const double(*)[x::kernel::kWindowSize][2]>(r.grad_window), r.window_row0, r.window_col0};
      for (int e = 0; e < std::min<int>(r.num_evals, dump::kMaxEvals); ++e, ++k) {
        const dump::Evaluation &ev = r.evals[e];
        const bool wj = ev.with_jacobian > 0;
        const double p[3] = {ev.params[0], ev.params[1], ev.params[2]}, v[1] = {ev.params[3]};
        const double *par[2] = {p, v};
        const auto t0 = std::chrono::steady_clock::now();
        fn(c, par, res, wj ? jac : nullptr);
        const auto t1 = std::chrono::steady_clock::now();
        if (rep < 0) {
          double s = 0;
          for (int i = 0; i < N; ++i) s += res[i] * res[i];
          worst = std::max(worst, std::abs(0.5 * s - ev.cost) / ev.cost);
          times.emplace_back();
          with_j.push_back(wj);
        } else {
          times[k].push_back(std::chrono::duration<float, std::nano>(t1 - t0).count());
        }
      }
    }
  }
  double sum[2] = {0, 0};
  long n[2] = {0, 0};
  for (size_t k = 0; k < times.size(); ++k) {
    auto &v = times[k];
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    sum[with_j[k]] += v[v.size() / 2];
    ++n[with_j[k]];
  }
  std::printf("  %s: cost vs recorded, largest relative difference %.1e | time per call (us): with Jacobian %.2f, without %.2f\n",
              name, worst, sum[1] / n[1] / 1000, sum[0] / n[0] / 1000);
}
