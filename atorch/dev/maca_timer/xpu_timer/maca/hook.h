#pragma once
#include <bvar/bvar.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <functional>
#include <string>
#include <unordered_set>
#include<chrono>

#include "xpu_timer/common/macro.h"
#include "xpu_timer/common/xpu_timer.h"

namespace atorch {

// get abs offset in shared object by address of function
ptrdiff_t get_offset(const void *symbol);

class MetaxGpuTimer : public XpuTimer {
  /* Use cuda event to timing kernel.
   */
 public:
  explicit MetaxGpuTimer() {
    cudaEventCreate(&startEvent_);
    cudaEventCreate(&stopEvent_);
  }
  // interfaces
  void startRecord() override;  
  void endRecord() override;
  void stopRecord();
  void resumeRecord();
  bool isReady() override;
  uint64_t getDuration() override;
  const std::string getName() override;
  const std::string getType() override;
  const std::string getFlop() override;

  // the event is in object pool, we reset it by different kernel.
  void reset(cudaStream_t s, std::function<const std::string()> des,
             const std::string &type, uint64_t flop);
  
  void reset(const::std::string  &des,
          const std::string &type);

  // parse nccl syms if needed.
  static void doPrepare();

  std::string name;
 private:
  cudaEvent_t startEvent_, stopEvent_;  // owned
  cudaStream_t stream_;                 // not owned
  // return kernel name, it's callback function and called in background thread,
  // be careful of the lifetime of object in closure.
  std::function<const std::string()> description_;
  // kernel type, current is batched matmul, matmul, coll
  std::string type_;
  uint64_t flop_;
  bool cpuEvent_;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  std::chrono::steady_clock::time_point stop;
  uint64_t duration;
};

}  // namespace atorch

#ifdef __cplusplus
extern "C" {
#endif
typedef struct mcblasContext *mcblasHandle_t;
typedef enum {} mcblasStatus_t;

typedef enum {} mcblasOperation_t;

typedef enum {} mcblasGemmAlgo_t;

typedef enum {} mcblasComputeType_t;

typedef cudaError_t (*mcLaunchKernelFn)(const void *, dim3, dim3, void **,
                                          size_t, cudaStream_t);
// typedef cudaError_t (*cudaLaunchKernelExCFn)(const cudaLaunchConfig_t *,
//                                              const void *, void **);
typedef mcblasStatus_t (*mcblasGemmExFn)(mcblasHandle_t, mcblasOperation_t,
                                         mcblasOperation_t, int, int, int,
                                         const void *, const void *,
                                         cudaDataType, int, const void *,
                                         cudaDataType, int, const void *,
                                         void *, cudaDataType, int,
                                         cudaDataType, mcblasGemmAlgo_t);

typedef mcblasStatus_t (*mcblasGetStreamFn)(mcblasHandle_t, cudaStream_t *);
typedef mcblasStatus_t (*mcblasGemmStridedBatchedExFn)(
    mcblasHandle_t, mcblasOperation_t, mcblasOperation_t, int, int, int,
    const void *, const void *, cudaDataType_t, int, long long int,
    const void *, cudaDataType_t, int, long long int, const void *, void *,
    cudaDataType_t, int, long long int, int, mcblasComputeType_t,
    mcblasGemmAlgo_t);

static mcblasGemmStridedBatchedExFn orig_mcblasGemmStridedBatchedEx = NULL;
static mcblasGetStreamFn orig_mcblasGetStream = NULL;
// static cudaLaunchKernelExCFn orig_cudaLaunchKernelExC = NULL;
static mcLaunchKernelFn orig_mcLaunchKernel = NULL;
static mcblasGemmExFn orig_mcblasGemmEx = NULL;


std::unordered_set<const void*> fns_to_skip;
std::unordered_map<const void*, std::string> fns_to_name;
std::unordered_map<ptrdiff_t, std::string> addr_to_name{
};

#ifdef __cplusplus
}
#endif
