#pragma once

#include <fstream>
#include <memory>
#include <vector>

#include <x/eklt/error.h>
#include <x/eklt/error_kernel.h>
#include <x/eklt/eklt_patch.h>
#include <x/eklt/optimizer_dump.h>
#include <x/eklt/types.h>


namespace x {

  /**
   * @brief The Optimizer performs optimization to find the best warp and optical flow for each patch.
   */
  struct Optimizer {
    explicit Optimizer(Params params, EkltPerformanceLoggerPtr perf_logger = nullptr);

    ~Optimizer();

    /**
     * @brief updates the EKLT parameters
     */
    void setParams(const Params &params);

    void setPerfLogger(const EkltPerformanceLoggerPtr& perf_logger);

    /**
     * @brief Counts how many features are using the current image with timestamp time. 
     * If none are using it, free the memory.
     */
    void decrementCounter(double time);

    /**
     * @ brief precomputes log gradients of the image
     */
    void getLogGradients(const cv::Mat &img, cv::Mat &I_x, cv::Mat &I_y);

    void precomputeLogImageArray(const EkltPatches &patches, const ImageBuffer::iterator &image_it);

    /**
     * @brief perform optimization of cost function (7) in the original paper.
     */
    void optimizeParameters(const cv::Mat &event_frame, EkltPatch &patch, double t);

    Params params_;
    EkltPerformanceLoggerPtr perf_logger_;

    ceres::Problem::Options prob_options;
    ceres::Solver::Options solver_options;

    OptimizerData optimizer_data_;

    int patch_size_;

    // Optional dump of every N-th solve (golden vectors), enabled by the environment variable EKLT_DUMP_EVERY=N.
    // Record format in optimizer_dump.h. The file optimizer_dump.bin is written next to optimizations.csv.
    int dump_every_ = 0;
    int64_t solve_index_ = 0;
    std::ofstream dump_file_;

    // Cost function, chosen by the environment variable EKLT_COST: autodiff (default, Ceres' automatic
    // differentiation of ErrorRotation, the published code), analytic (ErrorRotationAnalytic, hand-written
    // derivatives), kernel (ErrorRotationKernel<float>, the cost kernel for HLS on a window of the gradient) or
    // kernel_double (the same kernel in double, to tell the effect of float from the effect of the window). With a
    // dump, the stored reference outputs come from the cost function in use.
    enum class Cost { Autodiff, Analytic, Kernel, KernelDouble };
    Cost cost_ = Cost::Autodiff;
    // EKLT_COST=kernel or kernel_double: pixels the kernel read outside its window, over the whole run, printed at the
    // end
    KernelWindowCounter kernel_window_;

  private:
    std::unique_ptr<dump::Record> startDumpRecord(const cv::Mat &event_frame, const EkltPatch &patch, double t,
                                                  const double p0[3], const double v0[1]);

    void finishDumpRecord(dump::Record &record, const ceres::CostFunction &cost_function, const double p[3],
                          const double v[1], const ceres::Solver::Summary &summary,
                          const std::vector<dump::Evaluation> &evals);
  };

}
