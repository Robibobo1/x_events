// Times the versions of the EKLT cost function on the calls recorded in the golden vectors (optimizer_dump.bin,
// format in x/eklt/optimizer_dump.h), to find where the cost kernel gains its time on the CPU. Besides the versions
// EKLT-VIO uses (original with Ceres' automatic derivatives, hand-written, kernel in double and in float), it times
// variants that change one thing at a time (test/cost_timing_variants.h).
//
//   eklt_cost_timing <optimizer_dump.bin>... [--reps 5] [--evict] [--csv per_call.csv]
//
// For every record (one solve), the inputs are rebuilt as in eklt_cost_check, with the template gradient as a whole
// 240x180 image (as EKLT-VIO holds it) made from the stored 48x48 window, edge values repeated outside it. Then each
// version runs the recorded calls of that solve in order, each call timed alone. The whole pass is repeated --reps
// times, the order of the versions rotated, and the median of the repetitions is kept for each call.
// --evict: before each version runs a solve, 4 MB are read and written to push the inputs out of the L1 and L2 caches,
// as when other patches were solved in between.
// Before timing, every version is evaluated with its Jacobian at every call and compared with the version it is
// derived from, to make sure the times compare the same computation.
//
// Build (no catkin, in the eklt-vio:noetic image, flags of x_events' CMakeLists.txt in Release):
//   g++ -std=c++17 -O3 -DNDEBUG -funsafe-loop-optimizations -fsee -funroll-loops -fno-math-errno
//       -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros
//       -D_LINUX -D_REENTRANT -DEIGEN_MATRIXBASE_PLUGIN='<x/common/eigen_matrix_base_plugin.h>'
//       -Iinclude -I/catkin_ws/devel/include -I/usr/include/eigen3 -I/usr/include/opencv4
//       test/eklt_cost_timing.cpp src/x/eklt/error_analytic.cpp -L/catkin_ws/devel/lib -Wl,-rpath,/catkin_ws/devel/lib
//       -lceres -lglog -lgflags -lopencv_core -lpthread -o eklt_cost_timing

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include <x/eklt/error.h>
#include <x/eklt/error_analytic.h>
#include <x/eklt/error_kernel.h>
#include <x/eklt/optimizer_dump.h>

#include "cost_timing_variants.h"

using namespace x;

namespace {

constexpr int N = dump::kNumPixels;
constexpr int K = dump::kNumParams;

struct Version {
  const char *id;
  const char *what;
  const char *parent;  // id of the version it is checked against, nullptr: none
  double tolerance;  // largest relative difference to the parent
};

// order = order of build() below
const Version kVersions[] = {
    {"O", "original, autodiff, whole image", nullptr, 0},
    {"O-w", "original on a 48x48 window grid", "O", 1e-9},
    {"H", "hand-written, whole image", "O", 1e-9},
    {"H-w", "hand-written on a 48x48 window grid", "H", 1e-9},
    {"H'", "copy of H (whole image)", "H", 1e-9},
    {"H-e", "H' + events divided once per solve", "H", 1e-9},
    {"H-n", "H' + patch size constant", "H", 1e-9},
    {"H-k", "H' + kernel interpolation on window", "H", 1e-9},
    {"H-ekn", "H' + all three", "H", 1e-9},
    {"H-c", "H' + copy of Ceres' interpolation", "H", 1e-9},
    {"H-cv", "H-c + cell loaded as one vector", "H-c", 1e-9},
    {"H-ci", "H-c + Hermite step inlined", "H-c", 1e-9},
    {"H-cvi", "H-c + both", "H-c", 1e-9},
    {"Kd", "kernel double", "O", 1e-9},
    {"Kd'", "copy of Kd", "Kd", 1e-9},
    {"Kd-e", "Kd' + events divided at every call", "Kd", 1e-9},
    {"Kd-cw", "Kd' + Ceres interpolation, window grid", "Kd", 1e-9},
    {"Kd-ci", "Kd' + Ceres interpolation, whole image", "Kd", 1e-9},
    {"Kf", "kernel float", "Kd", 2e-4},
};
constexpr int V = sizeof(kVersions) / sizeof(kVersions[0]);

// index of the parent version, -1: none
int parentOf(int v) {
  if (kVersions[v].parent == nullptr)
    return -1;
  for (int p = 0; p < V; ++p)
    if (std::strcmp(kVersions[p].id, kVersions[v].parent) == 0)
      return p;
  throw std::logic_error(std::string("unknown parent ") + kVersions[v].parent);
}

// Inputs of the cost functions of one record. Not movable: the interpolators point to the grids.
struct Inputs {
  Grid window_grid;
  Interpolator window_interp;
  std::vector<double> &image;
  Grid image_grid;
  Interpolator image_interp;
  cv::Mat event_frame;
  cv::Point2d center, init_center;

  Inputs(const dump::Record &r, std::vector<double> &image_buffer)
      : window_grid(r.grad_window, r.window_row0, r.window_row0 + dump::kWindowSize, r.window_col0,
                    r.window_col0 + dump::kWindowSize),
        window_interp(window_grid),
        image(fill(r, image_buffer)),
        image_grid(image.data(), 0, r.image_rows, 0, r.image_cols),
        image_interp(image_grid),
        event_frame(dump::kPatchSize, dump::kPatchSize, CV_64F, const_cast<double *>(r.event_frame)),
        center(r.center[0], r.center[1]),
        init_center(r.init_center[0], r.init_center[1]) {}

  // whole image: every pixel takes the value of the nearest window pixel, which is what the window grid returns
  static std::vector<double> &fill(const dump::Record &r, std::vector<double> &image) {
    image.resize(static_cast<size_t>(r.image_rows) * r.image_cols * 2);
    for (int y = 0; y < r.image_rows; ++y) {
      const int wy = std::min(std::max(y - r.window_row0, 0), dump::kWindowSize - 1);
      for (int x = 0; x < r.image_cols; ++x) {
        const int wx = std::min(std::max(x - r.window_col0, 0), dump::kWindowSize - 1);
        for (int k = 0; k < 2; ++k)
          image[(static_cast<size_t>(y) * r.image_cols + x) * 2 + k] = r.grad_window[(wy * dump::kWindowSize + wx) * 2 + k];
      }
    }
    return image;
  }
};

std::vector<std::unique_ptr<ceres::CostFunction>> build(const dump::Record &r, const Inputs &in) {
  using namespace timing;
  const auto config = [&](const Interpolator *interp) {
    return std::make_shared<CostFunctionConfig>(in.center, in.init_center, &in.event_frame,
                                                const_cast<Interpolator *>(interp));
  };
  const CeresSampler image{&in.image_interp};
  const WindowSampler window{reinterpret_cast<const double(*)[kW][2]>(r.grad_window), r.window_row0, r.window_col0};
  Interpolator *ii = const_cast<Interpolator *>(&in.image_interp);
  Interpolator *wi = const_cast<Interpolator *>(&in.window_interp);

  std::vector<std::unique_ptr<ceres::CostFunction>> f;
  f.emplace_back(Generator::Create(in.center, in.init_center, &in.event_frame, ii));
  f.emplace_back(Generator::Create(in.center, in.init_center, &in.event_frame, wi));
  f.emplace_back(ErrorRotationAnalytic::Create(in.center, in.init_center, &in.event_frame, ii));
  f.emplace_back(ErrorRotationAnalytic::Create(in.center, in.init_center, &in.event_frame, wi));
  f.emplace_back(new AnalyticVariant<CeresSampler, false, false>(config(ii), image));
  f.emplace_back(new AnalyticVariant<CeresSampler, true, false>(config(ii), image));
  f.emplace_back(new AnalyticVariant<CeresSampler, false, true>(config(ii), image));
  f.emplace_back(new AnalyticVariant<WindowSampler, false, false>(config(ii), window));
  f.emplace_back(new AnalyticVariant<WindowSampler, true, true>(config(ii), window));
  const auto copy = [&](auto sampler) {
    sampler.data = in.image.data();
    sampler.row_begin = 0, sampler.row_end = r.image_rows, sampler.col_begin = 0, sampler.col_end = r.image_cols;
    return sampler;
  };
  f.emplace_back(new AnalyticVariant<CeresCopySampler<false, false>, false, false>(config(ii), copy(CeresCopySampler<false, false>{})));
  f.emplace_back(new AnalyticVariant<CeresCopySampler<true, false>, false, false>(config(ii), copy(CeresCopySampler<true, false>{})));
  f.emplace_back(new AnalyticVariant<CeresCopySampler<false, true>, false, false>(config(ii), copy(CeresCopySampler<false, true>{})));
  f.emplace_back(new AnalyticVariant<CeresCopySampler<true, true>, false, false>(config(ii), copy(CeresCopySampler<true, true>{})));
  f.emplace_back(ErrorRotationKernel<double>::Create(in.center, in.init_center, &in.event_frame, in.image_grid, nullptr));
  f.emplace_back(new KernelVariant<false, false>(in.center, in.init_center, in.event_frame, in.image_grid, nullptr));
  f.emplace_back(new KernelVariant<false, true>(in.center, in.init_center, in.event_frame, in.image_grid, nullptr));
  f.emplace_back(new KernelVariant<true, false>(in.center, in.init_center, in.event_frame, in.image_grid, wi));
  f.emplace_back(new KernelVariant<true, false>(in.center, in.init_center, in.event_frame, in.image_grid, ii));
  f.emplace_back(ErrorRotationKernel<float>::Create(in.center, in.init_center, &in.event_frame, in.image_grid, nullptr));
  if (static_cast<int>(f.size()) != V)
    throw std::logic_error("build() and kVersions disagree");
  return f;
}

// output buffers as Ceres passes them: residuals, and the Jacobian of each parameter block, row major
struct Buffers {
  double r[N], J_p[N * 3], J_v[N];
  double *jacobians[2] = {J_p, J_v};
};

inline void evaluate(const ceres::CostFunction &f, const double params[K], bool with_jacobian, Buffers &b) {
  const double p[3] = {params[0], params[1], params[2]};
  const double v[1] = {params[3]};
  const double *parameters[] = {p, v};
  f.Evaluate(parameters, b.r, with_jacobian ? b.jacobians : nullptr);
}

// largest |a - b| / largest |b|
double relDiff(const double *a, const double *b, int n) {
  double d = 0, m = 0;
  for (int i = 0; i < n; ++i) {
    d = std::max(d, std::abs(a[i] - b[i]));
    m = std::max(m, std::abs(b[i]));
  }
  return m > 0 ? d / m : d;
}

std::vector<std::unique_ptr<dump::Record>> load(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  dump::Header h{};
  if (!in.read(reinterpret_cast<char *>(&h), sizeof(h)) || std::memcmp(h.magic, "EKLTDUMP", 8) != 0 ||
      h.version != dump::kVersion || h.record_size != sizeof(dump::Record))
    throw std::runtime_error(path + ": not an optimizer dump of version " + std::to_string(dump::kVersion));
  in.seekg(h.header_size);
  std::vector<std::unique_ptr<dump::Record>> records;
  while (true) {
    auto r = std::make_unique<dump::Record>();
    if (!in.read(reinterpret_cast<char *>(r.get()), sizeof(dump::Record)))
      break;
    records.push_back(std::move(r));
  }
  return records;
}

int numCalls(const dump::Record &r) { return std::min<int>(r.num_evals, dump::kMaxEvals); }

// every version against its parent, with the Jacobian, at every recorded call. Returns true if all pass.
bool check(const std::vector<std::unique_ptr<dump::Record>> &records) {
  std::vector<double> worst(V, 0.0);
  std::vector<double> image;
  std::vector<Buffers> out(V);
  for (const auto &rec : records) {
    const Inputs in(*rec, image);
    const auto f = build(*rec, in);
    for (int e = 0; e < numCalls(*rec); ++e) {
      for (int v = 0; v < V; ++v)
        evaluate(*f[v], rec->evals[e].params, true, out[v]);
      for (int v = 0; v < V; ++v) {
        const int p = parentOf(v);
        if (p < 0)
          continue;
        worst[v] = std::max({worst[v], relDiff(out[v].r, out[p].r, N), relDiff(out[v].J_p, out[p].J_p, N * 3),
                             relDiff(out[v].J_v, out[p].J_v, N)});
      }
    }
  }
  bool pass = true;
  std::printf("\nCheck: largest relative difference to the parent version (residuals, Jacobian)\n");
  for (int v = 0; v < V; ++v) {
    if (parentOf(v) < 0)
      continue;
    const bool ok = worst[v] < kVersions[v].tolerance;
    pass = pass && ok;
    std::printf("  %-6s vs %-4s %9.1e  %s\n", kVersions[v].id, kVersions[v].parent, worst[v],
                ok ? "ok" : "FAIL");
  }
  return pass;
}

// overwrites the L1 and L2 caches
void evict() {
  static std::vector<unsigned char> buffer(4 << 20);
  for (size_t i = 0; i < buffer.size(); i += 64)
    buffer[i]++;
}

double median(std::vector<float> &v) {
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

void run(const std::string &path, int reps, bool do_evict, FILE *csv) {
  const auto records = load(path);
  std::vector<int> first_call(records.size() + 1, 0);
  for (size_t i = 0; i < records.size(); ++i)
    first_call[i + 1] = first_call[i] + numCalls(*records[i]);
  const int calls = first_call.back();
  long with_j = 0;
  for (const auto &rec : records)
    for (int e = 0; e < numCalls(*rec); ++e)
      with_j += rec->evals[e].with_jacobian > 0;
  std::printf("%s\n%zu records, %d calls (%ld with Jacobian, %ld without), %d repetitions%s\n", path.c_str(),
              records.size(), calls, with_j, calls - with_j, reps, do_evict ? ", caches evicted before each solve" : "");

  if (!check(records))
    std::printf("  some versions do not compute the same thing as their parent: their times are not comparable\n");

  // times[v][call][rep], nanoseconds
  std::vector<std::vector<std::vector<float>>> times(V, std::vector<std::vector<float>>(calls, std::vector<float>(reps)));
  std::vector<double> image;
  Buffers b;
  using Clock = std::chrono::steady_clock;

  // warm-up, not timed: brings the core to its working frequency
  for (size_t i = 0; i < std::min<size_t>(records.size(), 100); ++i) {
    const Inputs in(*records[i], image);
    const auto f = build(*records[i], in);
    for (int v = 0; v < V; ++v)
      for (int e = 0; e < numCalls(*records[i]); ++e)
        evaluate(*f[v], records[i]->evals[e].params, records[i]->evals[e].with_jacobian > 0, b);
  }

  for (int rep = 0; rep < reps; ++rep) {
    for (size_t i = 0; i < records.size(); ++i) {
      const dump::Record &rec = *records[i];
      const Inputs in(rec, image);
      const auto f = build(rec, in);
      for (int j = 0; j < V; ++j) {
        const int v = static_cast<int>((j + rep + i) % V);
        if (do_evict)
          evict();
        for (int e = 0; e < numCalls(rec); ++e) {
          const dump::Evaluation &ev = rec.evals[e];
          const auto t0 = Clock::now();
          evaluate(*f[v], ev.params, ev.with_jacobian > 0, b);
          const auto t1 = Clock::now();
          times[v][first_call[i] + e][rep] = std::chrono::duration<float, std::nano>(t1 - t0).count();
        }
      }
    }
  }

  // per call: median of the repetitions. Per version: mean and median over the calls, with and without Jacobian
  std::vector<std::vector<double>> per_call(V, std::vector<double>(calls));
  for (int v = 0; v < V; ++v)
    for (int c = 0; c < calls; ++c)
      per_call[v][c] = median(times[v][c]);

  double mean[V][2], med[V][2];
  for (int v = 0; v < V; ++v) {
    for (int j = 0; j < 2; ++j) {
      std::vector<float> sel;
      for (size_t i = 0; i < records.size(); ++i)
        for (int e = 0; e < numCalls(*records[i]); ++e)
          if ((records[i]->evals[e].with_jacobian > 0) == (j == 0))
            sel.push_back(static_cast<float>(per_call[v][first_call[i] + e]));
      double s = 0;
      for (float x : sel)
        s += x;
      mean[v][j] = sel.empty() ? 0 : s / sel.size() / 1000.0;
      med[v][j] = sel.empty() ? 0 : median(sel) / 1000.0;
    }
  }

  std::printf("\nTime per call (us): mean over the calls (median of the repetitions per call), median over the calls"
              "\n                                                  with Jacobian           without Jacobian\n"
              "  id     version                                 mean  median  vs O     mean  median  vs O"
              "    vs parent (with, without)\n");
  for (int v = 0; v < V; ++v) {
    std::printf("  %-6s %-38s %6.2f  %6.2f  %+4.0f%%  %6.2f  %6.2f  %+4.0f%%", kVersions[v].id, kVersions[v].what,
                mean[v][0], med[v][0], 100 * (mean[v][0] / mean[0][0] - 1), mean[v][1], med[v][1],
                100 * (mean[v][1] / mean[0][1] - 1));
    const int p = parentOf(v);
    if (p >= 0)
      std::printf("    %-4s %+4.0f%% %+4.0f%%", kVersions[p].id, 100 * (mean[v][0] / mean[p][0] - 1),
                  100 * (mean[v][1] / mean[p][1] - 1));
    std::printf("\n");
  }
  std::printf("\n");

  if (csv != nullptr)
    for (size_t i = 0; i < records.size(); ++i)
      for (int e = 0; e < numCalls(*records[i]); ++e)
        for (int v = 0; v < V; ++v)
          std::fprintf(csv, "%s,%zu,%d,%d,%s,%d,%.0f\n", path.c_str(), i, e, records[i]->evals[e].with_jacobian > 0 ? 1 : 0,
                       kVersions[v].id, do_evict ? 1 : 0, per_call[v][first_call[i] + e]);
}

}  // namespace

int main(int argc, char **argv) {
  std::vector<std::string> paths;
  int reps = 5;
  bool do_evict = false;
  std::string csv_path;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--reps" && i + 1 < argc)
      reps = std::atoi(argv[++i]);
    else if (a == "--evict")
      do_evict = true;
    else if (a == "--csv" && i + 1 < argc)
      csv_path = argv[++i];
    else
      paths.push_back(a);
  }
  if (paths.empty()) {
    std::cerr << "usage: eklt_cost_timing <optimizer_dump.bin>... [--reps 5] [--evict] [--csv per_call.csv]" << std::endl;
    return 2;
  }
  FILE *csv = nullptr;
  if (!csv_path.empty()) {
    csv = std::fopen(csv_path.c_str(), "w");
    std::fprintf(csv, "file,record,call,with_jacobian,version,evict,ns\n");
  }
  for (const auto &p : paths)
    run(p, reps, do_evict, csv);
  if (csv != nullptr)
    std::fclose(csv);
  return 0;
}
