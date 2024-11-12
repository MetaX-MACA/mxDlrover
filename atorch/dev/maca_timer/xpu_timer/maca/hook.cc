

#include <cuda_runtime_api.h>
#include <cxxabi.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "xpu_timer/maca/hook.h"
#include "xpu_timer/common/manager.h"

namespace atorch {

bool findNCCLSymsOrcudaLaunch(std::string* s, cudaError_t* status, auto & fn,
                              const void* func) {
  if (fns_to_skip.find(func) != fns_to_skip.end()) {
    *status = fn();
    return false;
  }
  if (fns_to_name.find(func) == fns_to_name.end()) {
    ptrdiff_t offset = get_offset(func);
    atorch::GpuTimerManager<atorch::MetaxGpuTimer>::getInstance();
    if (addr_to_name.find(offset) != addr_to_name.end()) {
      fns_to_name[func] = addr_to_name[offset];
      LOG(INFO) << "find symbols " << addr_to_name[offset] << " offset: " << std::hex << offset;
    } else {
      fns_to_skip.insert(func);
      *status = fn();
      LOG(INFO) << "skips  offset: " << std::hex << offset;
      return false;
    }
  }
  *s = fns_to_name[func];
  return true;
}

ptrdiff_t get_offset(const void* symbol) {
  Dl_info info;
  if (dladdr(symbol, &info) != 0) {
    LOG(INFO) << "launch lib: " << info.dli_fname << " name: " << info.dli_sname << " offset" << ((char*)symbol - (char*)info.dli_fbase);
    return (char*)symbol - (char*)info.dli_fbase;
  }
  return 0;
}

void resetNcclSymsMap() {
  const char* env_var = std::getenv("XPU_TIMER_LIB_PATH");
  if (env_var == nullptr) {
    return;
  }

  std::ostringstream oss;
  oss << "nm " << env_var
      << " | grep mcclKernel | c++filt | "
      << R"(awk 'BEGIN{a=""} /mcclKernel_/{fn=$3;sub(/\(.*/, "", fn);a = "0x"$1 "," fn "\n" a} END{print a}')";
  std::string nm_output = util::execShellCommand(oss.str().c_str());
  addr_to_name.clear();
  for (auto& each_line : util::split(nm_output, "\n")) {
    if (each_line.empty()) continue;
    std::vector<std::string> tokens = util::split(each_line, ",");
    ptrdiff_t addr = std::stoll(tokens[0], nullptr, 16);  // hex
    std::string func_name = tokens[1];
    addr_to_name[addr] = func_name;
    oss.str("");
    oss.clear();
    oss << std::hex << addr;
    LOG(INFO) << "read symbols " << oss.str() << ":" << func_name;
  }
}

void MetaxGpuTimer::doPrepare() { resetNcclSymsMap(); }
void MetaxGpuTimer::startRecord() { cudaEventRecord(startEvent_, stream_); }
void MetaxGpuTimer::endRecord() { 
  if (!cpuEvent_) {cudaEventRecord(stopEvent_, stream_); }
  else {end = std::chrono::steady_clock::now();}
}

void MetaxGpuTimer::stopRecord() { stop =  std::chrono::steady_clock::now();}
void MetaxGpuTimer::resumeRecord() { duration += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - stop).count();}

bool MetaxGpuTimer::isReady() {
  return cudaEventQuery(stopEvent_) != cudaErrorNotReady;
}

uint64_t MetaxGpuTimer::getDuration() {
  if (!cpuEvent_) {
    float elapsedTime;  // ms
    cudaEventElapsedTime(&elapsedTime, startEvent_, stopEvent_);
    return uint64_t(elapsedTime * 1000);  // ms -> us
  } else {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() - duration;
  }
}

const std::string MetaxGpuTimer::getName() { 
  if (!cpuEvent_) {
   return description_();  
  }
  return name;
}
void MetaxGpuTimer::reset(cudaStream_t s,
                           std::function<const std::string()> des,
                           const std::string& type, uint64_t flop) {
  stream_ = s;
  description_ = des;
  type_ = type;
  flop_ = flop;
  cpuEvent_ = false;
  startRecord();
}

void MetaxGpuTimer::reset(const::std::string  &des,
          const std::string &type)
{
  name = des;
  type_ = type;
  flop_ = 0;
  cpuEvent_ = true;
  duration = 0;
  start = std::chrono::steady_clock::now();
}

const std::string MetaxGpuTimer::getType() { return type_; }
const std::string MetaxGpuTimer::getFlop() { return std::to_string(flop_); }

std::string demangle(const char* mangledName) {
  int status = -1;
  char* demangled = abi::__cxa_demangle(mangledName, NULL, NULL, &status);
  std::string result = (status == 0) ? demangled : mangledName;
  free(demangled);
  return result;
}

}  // namespace atorch

#ifdef __cplusplus
extern "C" {
#endif

EXPOSE_API
cudaError_t mcLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem,
                             cudaStream_t stream) {
  SETUP_DLSYM(mcLaunchKernel);
  cudaError_t status;
  std::string name;
  auto record =
      atorch::GpuTimerManager<atorch::MetaxGpuTimer>::getInstance().getEvent();
  auto origin_fun = [&func, gridDim, blockDim, args, sharedMem,
                     stream, record]() -> cudaError_t {
    record->stopRecord();
    return orig_mcLaunchKernel(func, gridDim, blockDim, args, sharedMem,
                                 stream);
  };
  record->reset("mcLaunchKernel", "hook");
  if (!atorch::findNCCLSymsOrcudaLaunch(&name, &status, origin_fun, func))
  {
    record->resumeRecord();
    atorch::GpuTimerManager<atorch::MetaxGpuTimer>::getInstance().recordEvent(
      record);
    return status;
  }
  auto fn = [name, blockDim, gridDim]() -> std::string {
    std::ostringstream oss;
    oss << "atorch_" << name << "__grid_" << gridDim.x << "_" << gridDim.y
        << "_" << gridDim.z << "_block_" << blockDim.x << "_" << blockDim.y
        << "_" << blockDim.z;
    return oss.str();
  };
  auto event =
      atorch::GpuTimerManager<atorch::MetaxGpuTimer>::getInstance().getEvent();
  event->reset(stream, fn, "coll", 0);
  record->stopRecord();
  status = origin_fun();
  record->resumeRecord();
  atorch::GpuTimerManager<atorch::MetaxGpuTimer>::getInstance().recordEvent(
      event);
  record->name = record->name + "_mccl";
  atorch::GpuTimerManager<atorch::MetaxGpuTimer>::getInstance().recordEvent(
      record);
  return status;
}

EXPOSE_API
mcblasStatus_t mcblasGemmStridedBatchedEx(
    mcblasHandle_t handle, mcblasOperation_t transa, mcblasOperation_t transb,
    int m, int n, int k, const void* alpha, const void* A, cudaDataType_t Atype,
    int lda, long long int strideA, const void* B, cudaDataType_t Btype,
    int ldb, long long int strideB, const void* beta, void* C,
    cudaDataType_t Ctype, int ldc, long long int strideC, int batchCount,
    mcblasComputeType_t computeType, mcblasGemmAlgo_t algo) {
  SETUP_DLSYM(mcblasGemmStridedBatchedEx);
  SETUP_DLSYM(mcblasGetStream);
  cudaStream_t s;
  auto record =
      atorch::GpuTimerManager<atorch::MetaxGpuTimer>::getInstance().getEvent();
  record->reset("mcblasGemmStridedBatchedEx", "hook");
  orig_mcblasGetStream(handle, &s);
  auto fn = [batchCount, m, n, k]() -> std::string {
    std::ostringstream oss;
    oss << "atorch_bmm"
        << "_bmnk"
        << "_" << batchCount << "_" << m << "_" << n << "_" << k;
    return oss.str();
  };
  auto event =
      atorch::GpuTimerManager<atorch::MetaxGpuTimer>::getInstance().getEvent();
  uint64_t flop = 2;
  event->reset(s, fn, "bmm", flop * batchCount * m * n * k);
  record->stopRecord();
  auto status = orig_mcblasGemmStridedBatchedEx(
      handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype,
      ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType,
      algo);
  record->resumeRecord();
  atorch::GpuTimerManager<atorch::MetaxGpuTimer>::getInstance().recordEvent(
      event);
  atorch::GpuTimerManager<atorch::MetaxGpuTimer>::getInstance().recordEvent(
      record);
  return status;
}

EXPOSE_API
mcblasStatus_t mcblasGemmEx(mcblasHandle_t handle, mcblasOperation_t transa,
                            mcblasOperation_t transb, int m, int n, int k,
                            const void* alpha, const void* A,
                            cudaDataType Atype, int lda, const void* B,
                            cudaDataType Btype, int ldb, const void* beta,
                            void* C, cudaDataType Ctype, int ldc,
                            cudaDataType computeType, mcblasGemmAlgo_t algo) {
  SETUP_DLSYM(mcblasGemmEx);
  SETUP_DLSYM(mcblasGetStream);
  cudaStream_t s;
  auto record =
      atorch::GpuTimerManager<atorch::MetaxGpuTimer>::getInstance().getEvent();
  record->reset("mcblasGemmEx", "hook");
  orig_mcblasGetStream(handle, &s);
  auto fn = [m, n, k]() -> std::string {
    std::ostringstream oss;
    oss << "atorch_mm"
        << "_mnk"
        << "_" << m << "_" << n << "_" << k;
    return oss.str();
  };
  uint64_t flop = 2;
  auto event =
      atorch::GpuTimerManager<atorch::MetaxGpuTimer>::getInstance().getEvent();
  event->reset(s, fn, "mm", flop * m * n * k);
  record->stopRecord();
  auto status =
      orig_mcblasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda,
                        B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
  record->resumeRecord();
  atorch::GpuTimerManager<atorch::MetaxGpuTimer>::getInstance().recordEvent(
      event);
  atorch::GpuTimerManager<atorch::MetaxGpuTimer>::getInstance().recordEvent(
      record);
  return status;
}

#ifdef __cplusplus
}
#endif
