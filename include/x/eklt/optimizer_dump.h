#pragma once

// Record format of the optional optimizer dump (golden vectors), written by Optimizer::optimizeParameters when the
// environment variable EKLT_DUMP_EVERY=N is set: one record every N-th solve, in optimizer_dump.bin next to
// optimizations.csv. The file is one Header followed by Records, little endian, no padding.
//
// Plain C++ without dependencies, so that a test bench (e.g. Vitis HLS) can read the file with the same structs.

#include <cstdint>

namespace x {
namespace dump {

constexpr uint32_t kVersion = 1;
constexpr int kPatchSize = 19;                      // EKLT patch, eklt_patch_size
constexpr int kNumPixels = kPatchSize * kPatchSize; // = number of residuals
constexpr int kNumParams = 4;                       // theta, tx, ty, flow angle
constexpr int kWindowSize = 48;                     // template gradient window, kWindowSize x kWindowSize x (Ix, Iy)
constexpr int kMaxEvals = 32;                       // cost function evaluations stored per solve

struct Header {
  char magic[8];          // "EKLTDUMP"
  uint32_t version;       // kVersion
  uint32_t header_size;   // sizeof(Header)
  uint32_t record_size;   // sizeof(Record)
  uint32_t patch_size;    // kPatchSize
  uint32_t window_size;   // kWindowSize
  uint32_t max_evals;     // kMaxEvals
};

// One call of the cost function by Ceres
struct Evaluation {
  double params[kNumParams];  // theta, tx, ty, flow angle
  double with_jacobian;       // 1 if Ceres asked for the Jacobian, else 0
  double cost;                // 0.5 * sum of the squared residuals
};

struct Record {
  int64_t solve_index;        // row of optimizations.csv (0 = first solve)
  int32_t patch_id;
  int32_t num_evals;          // calls of the cost function in this solve (only the first kMaxEvals are stored)
  int32_t window_row0;        // image row and column of grad_window[0], can be negative: outside the image the
  int32_t window_col0;        // nearest border value is repeated, as ceres::Grid2D does
  int32_t image_rows;
  int32_t image_cols;
  int32_t num_iterations;     // summary.iterations.size(), as in optimizations.csv
  int32_t termination_type;   // ceres::TerminationType
  int32_t num_jacobian_evaluations;  // from ceres::Solver::Summary
  int32_t num_residual_evaluations;
  double t;                   // timestamp of the solve
  double t_init;              // timestamp of the template frame
  double center[2];           // patch center (x, y) given to the cost function
  double init_center[2];      // feature position (x, y) in the template frame
  double params_init[kNumParams];   // theta, tx, ty, flow angle before the solve
  double params_final[kNumParams];  // after the solve
  double initial_cost;        // from ceres::Solver::Summary
  double final_cost;
  double event_frame_norm;    // cv::norm(event_frame), as used by the cost function
  double event_frame[kNumPixels];                     // row major (y, x)
  double grad_window[kWindowSize * kWindowSize * 2];  // row major (y, x), Ix and Iy interleaved, as in ceres::Grid2D
  Evaluation evals[kMaxEvals];
  double residuals_init[kNumPixels];                  // original cost function at params_init
  double jacobian_init[kNumPixels * kNumParams];      // row major, one row per residual: d/dtheta, d/dtx, d/dty, d/dphi
  double residuals_final[kNumPixels];                 // same at params_final
  double jacobian_final[kNumPixels * kNumParams];
};

static_assert(sizeof(Header) == 32, "dump header layout changed");
static_assert(sizeof(Record) == 70352, "dump record layout changed: update kVersion and the readers");

}  // namespace dump
}  // namespace x
