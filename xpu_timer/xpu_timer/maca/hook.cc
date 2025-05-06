// 2025-Modified by MetaX Integrated Circuits (Shanghai)Co., Ltd.All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xpu_timer/maca/hook.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "xpu_timer/common/constant.h"
#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/manager.h"
#include "xpu_timer/common/util.h"
#include "xpu_timer/maca/maca_timer.h"

static void getMatrixDimensions(const mcblasLtMatrixLayout_t& layout,
                                cudaDataType_t& dtype, int32_t& b,
                                uint64_t& rows, uint64_t& cols, uint64_t& ld,
                                int64_t& stride) {
  orig_mcblasLtMatrixLayoutGetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &b, sizeof(b), NULL);
  orig_mcblasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ROWS,
                                        &rows, sizeof(rows), NULL);
  orig_mcblasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_COLS,
                                        &cols, sizeof(cols), NULL);
  orig_mcblasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_LD, &ld,
                                        sizeof(ld), NULL);
  orig_mcblasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_TYPE,
                                        &dtype, sizeof(dtype), NULL);
  orig_mcblasLtMatrixLayoutGetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
      sizeof(stride), NULL);
}

#ifdef __cplusplus
extern "C" {
#endif

EXPOSE_API
cudaError_t mcLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem,
                             cudaStream_t stream) {
  SETUP_DLSYM(mcLaunchKernel);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_mcLaunchKernel(func, gridDim, blockDim, args, sharedMem,
                                 stream);
  const xpu_timer::maca::InterceptSymbol* sym;
  if (!xpu_timer::GpuTimerManager<
           xpu_timer::maca::MacaGpuTimer>::getInstance()
           .intercept_manager.isIntercepted(func, &sym)) {
    return orig_mcLaunchKernel(func, gridDim, blockDim, args, sharedMem,
                                 stream);
  }
  bool skip_tp = false;
  auto fn =
      xpu_timer::GpuTimerManager<
          xpu_timer::maca::MacaGpuTimer>::getInstance()
          .intercept_manager.handleCudaLaunchKernel(
              func, gridDim, blockDim, args, sharedMem, stream, sym, &skip_tp);
  if (skip_tp)
    return orig_mcLaunchKernel(func, gridDim, blockDim, args, sharedMem,
                                 stream);
  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::maca::MacaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(stream, fn,
               sym->func_type == "NCCL"
                   ? xpu_timer::constant::Metrics::CollMetrics::TYPE
                   : xpu_timer::constant::Metrics::MatmulMetrics::TYPE);
  cudaError_t status =
      orig_mcLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

// EXPOSE_API
// cudaError_t mcLaunchKernelExC(const cudaLaunchConfig_t* config,
//                                 const void* func, void** args) {
//   SETUP_DLSYM(mcLaunchKernelExC);
//   if (!::xpu_timer::util::config::GlobalConfig::enable)
//     return orig_mcLaunchKernelExC(config, func, args);
//   cudaError_t status;
//   const xpu_timer::maca::InterceptSymbol* sym;

//   if (!xpu_timer::GpuTimerManager<
//            xpu_timer::maca::MacaGpuTimer>::getInstance()
//            .intercept_manager.isIntercepted(func, &sym)) {
//     return orig_mcLaunchKernelExC(config, func, args);
//   }
//   bool skip_tp = false;

//   auto fn = xpu_timer::GpuTimerManager<
//                 xpu_timer::maca::MacaGpuTimer>::getInstance()
//                 .intercept_manager.handleCudaLaunchKernelExC(config, func, args,
//                                                              sym, &skip_tp);
//   if (skip_tp) return orig_mcLaunchKernelExC(config, func, args);
//   auto event = xpu_timer::GpuTimerManager<
//                    xpu_timer::maca::MacaGpuTimer>::getInstance()
//                    .getEvent();
//   event->reset(config->stream, fn,
//                xpu_timer::constant::Metrics::CollMetrics::TYPE);
//   status = orig_mcLaunchKernelExC(config, func, args);
//   xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
//       .recordEvent(event);
//   return status;
// }

EXPOSE_API
mcblasStatus_t mcGemmStridedBatchedEx(
    mcblasHandle_t handle, mcblasOperation_t transa, mcblasOperation_t transb,
    int m, int n, int k, const void* alpha, const void* A, cudaDataType_t Atype,
    int lda, long long int strideA, const void* B, cudaDataType_t Btype,
    int ldb, long long int strideB, const void* beta, void* C,
    cudaDataType_t Ctype, int ldc, long long int strideC, int batch_count,
    mcblasComputeType_t computeType, mcblasGemmAlgo_t algo) {
  SETUP_DLSYM_WITH_CUBLAS(mcblasGemmStridedBatchedEx);
  SETUP_DLSYM_WITH_CUBLAS(mcblasGetStream);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_mcblasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B,
        Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batch_count,
        computeType, algo);
  cudaStream_t s;
  orig_mcblasGetStream(handle, &s);
  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::maca::MacaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(s, xpu_timer::constant::Metrics::MatmulMetrics::TYPE,
               {batch_count, m, n, k}, {lda, ldb, ldc},
               {strideA, strideB, strideC}, static_cast<int>(transa),
               static_cast<int>(transb), static_cast<int>(algo),
               "xpu_timer_bmm_bias_bmnk_", Atype, "cublasGemmStridedBatchedEx");
  auto status = orig_mcblasGemmStridedBatchedEx(
      handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype,
      ldb, strideB, beta, C, Ctype, ldc, strideC, batch_count, computeType,
      algo);
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .recordEvent(event);
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
  SETUP_DLSYM_WITH_CUBLAS(mcblasGemmEx);
  SETUP_DLSYM_WITH_CUBLAS(mcblasGetStream);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_mcblasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype,
                             lda, B, Btype, ldb, beta, C, Ctype, ldc,
                             computeType, algo);
  cudaStream_t s;
  orig_mcblasGetStream(handle, &s);
  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::maca::MacaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(s, xpu_timer::constant::Metrics::MatmulMetrics::TYPE,
               {1, m, n, k}, {lda, ldb, ldc}, {0, 0, 0},
               static_cast<int>(transa), static_cast<int>(transb),
               static_cast<int>(algo), "xpu_timer_mm_bmnk_", Atype,
               "cublasGemmEx");

  auto status =
      orig_mcblasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda,
                        B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
mcblasStatus_t mcblasSgemm(mcblasHandle_t handle, mcblasOperation_t transa,
                           mcblasOperation_t transb, int m, int n, int k,
                           const float* alpha, const float* A, int lda,
                           const float* B, int ldb, const float* beta, float* C,
                           int ldc) {
  SETUP_DLSYM_WITH_CUBLAS(mcblasSgemm);
  SETUP_DLSYM_WITH_CUBLAS(mcblasGetStream);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_mcblasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B,
                            ldb, beta, C, ldc);
  cudaStream_t s;
  orig_mcblasGetStream(handle, &s);
  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::maca::MacaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(s, xpu_timer::constant::Metrics::MatmulMetrics::TYPE,
               {1, m, n, k}, {lda, ldb, ldc}, {0, 0, 0},
               static_cast<int>(transa), static_cast<int>(transb), -1,
               "xpu_timer_mm_bmnk_", CUDA_R_32F, "cublasSgemm");

  auto status = orig_mcblasSgemm(handle, transa, transb, m, n, k, alpha, A, lda,
                                 B, ldb, beta, C, ldc);
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
mcblasStatus_t mcblasSgemmStridedBatched(
    mcblasHandle_t handle, mcblasOperation_t transa, mcblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    long long int strideA, const float* B, int ldb, long long int strideB,
    const float* beta, float* C, int ldc, long long int strideC,
    int batch_count) {
  SETUP_DLSYM_WITH_CUBLAS(mcblasSgemmStridedBatched);
  SETUP_DLSYM_WITH_CUBLAS(mcblasGetStream);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_mcblasSgemmStridedBatched(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb,
        strideB, beta, C, ldc, strideC, batch_count);
  cudaStream_t s;
  orig_mcblasGetStream(handle, &s);
  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::maca::MacaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(s, xpu_timer::constant::Metrics::MatmulMetrics::TYPE,
               {batch_count, m, n, k}, {lda, ldb, ldc},
               {strideA, strideB, strideC}, static_cast<int>(transa),
               static_cast<int>(transb), -1, "xpu_timer_bmm_bmnk_", CUDA_R_32F,
               "cublasSgemmStridedBatched");
  auto status = orig_mcblasSgemmStridedBatched(
      handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB,
      beta, C, ldc, strideC, batch_count);

  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
mcblasStatus_t mcblasLtMatmul(
    mcblasLtHandle_t lightHandle, mcblasLtMatmulDesc_t computeDesc,
    const void* alpha, const void* A, mcblasLtMatrixLayout_t Adesc,
    const void* B, mcblasLtMatrixLayout_t Bdesc, const void* beta,
    const void* C, mcblasLtMatrixLayout_t Cdesc, void* D,
    mcblasLtMatrixLayout_t Ddesc, const mcblasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream) {
  SETUP_DLSYM_WITH_CUBLASLT(mcblasLtMatmul);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_mcblasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc, B,
                               Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace,
                               workspaceSizeInBytes, stream);

  SETUP_DLSYM_WITH_CUBLASLT(mcblasLtMatrixLayoutGetAttribute);
  SETUP_DLSYM_WITH_CUBLASLT(mcblasLtMatmulAlgoConfigGetAttribute);
  SETUP_DLSYM_WITH_CUBLASLT(mcblasLtMatmulDescGetAttribute);
  cudaDataType_t dtype_a, dtype_b, dtype_c;
  int32_t batch_a, batch_b, batch_c;
  uint64_t rows_a, rows_b, rows_c;
  uint64_t cols_a, cols_b, cols_c;
  uint64_t ld_a, ld_b, ld_c;
  int64_t stride_a, stride_b, stride_c;
  int32_t trans_a, trans_b, trans_c;
  getMatrixDimensions(Adesc, dtype_a, batch_a, rows_a, cols_a, ld_a, stride_a);
  getMatrixDimensions(Bdesc, dtype_b, batch_b, rows_b, cols_b, ld_b, stride_b);
  getMatrixDimensions(Cdesc, dtype_c, batch_c, rows_c, cols_c, ld_c, stride_c);
  orig_mcblasLtMatmulDescGetAttribute(
      computeDesc,
      static_cast<mcblasLtMatmulDescAttributes_t>(
          3) /*CUBLASLT_MATMUL_DESC_TRANSA*/,
      &trans_a, sizeof(trans_a), NULL);
  orig_mcblasLtMatmulDescGetAttribute(
      computeDesc,
      static_cast<mcblasLtMatmulDescAttributes_t>(
          4) /*CUBLASLT_MATMUL_DESC_TRANSB*/,
      &trans_b, sizeof(trans_b), NULL);
  orig_mcblasLtMatmulDescGetAttribute(
      computeDesc,
      static_cast<mcblasLtMatmulDescAttributes_t>(
          5) /*CUBLASLT_MATMUL_DESC_TRANSC*/,
      &trans_c, sizeof(trans_c), NULL);

  int m = trans_a ? cols_a : rows_a;
  int k = trans_a ? rows_a : cols_a;
  int n = trans_b ? rows_b : cols_b;

  int algo_id;
  size_t size_written;
  orig_mcblasLtMatmulAlgoConfigGetAttribute(
      algo,
      static_cast<mcblasLtMatmulAlgoConfigAttributes_t>(
          0) /*CUBLASLT_ALGO_CONFIG_ID*/,
      &algo_id, sizeof(algo_id), &size_written);
  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::maca::MacaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(stream, xpu_timer::constant::Metrics::MatmulMetrics::TYPE,
               {batch_a, m, n, k}, {ld_a, ld_b, ld_c},
               {stride_a, stride_b, stride_c}, trans_a, trans_b, algo_id,
               "xpu_timer_bmm_bias_bmnk_", dtype_a, "cublasLtMatmul",
               1 /*has bias*/);
  auto status = orig_mcblasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc,
                                    B, Bdesc, beta, C, Cdesc, D, Ddesc, algo,
                                    workspace, workspaceSizeInBytes, stream);
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
mcclResult_t mcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           mcclDataType_t datatype, mcclRedOp_t op,
                           mcclComm_t comm, cudaStream_t stream) {
  SETUP_DLSYM_WITH_NCCL(mcclAllReduce);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_mcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm,
                              stream);

  // Get more NCCL info in advance.
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .intercept_manager.interceptNcclInfo<xpu_timer::constant::SKIP_TP>(
          count, datatype, comm, stream);

  return orig_mcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm,
                            stream);
}

EXPOSE_API
mcclResult_t mcclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        mcclDataType_t datatype, mcclRedOp_t op, int root,
                        mcclComm_t comm, cudaStream_t stream) {
  SETUP_DLSYM_WITH_NCCL(mcclReduce);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_mcclReduce(sendbuff, recvbuff, count, datatype, op, root, comm,
                           stream);

  // Get more NCCL info in advance.
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .intercept_manager.interceptNcclInfo(count, datatype, comm, stream);

  return orig_mcclReduce(sendbuff, recvbuff, count, datatype, op, root, comm,
                         stream);
}

EXPOSE_API
mcclResult_t mcclAllGather(const void* sendbuff, void* recvbuff,
                           size_t sendcount, mcclDataType_t datatype,
                           mcclComm_t comm, cudaStream_t stream) {
  SETUP_DLSYM_WITH_NCCL(mcclAllGather);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_mcclAllGather(sendbuff, recvbuff, sendcount, datatype, comm,
                              stream);

  // Get more NCCL info in advance.
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .intercept_manager.interceptNcclInfo<xpu_timer::constant::SKIP_TP>(
          sendcount, datatype, comm, stream);

  return orig_mcclAllGather(sendbuff, recvbuff, sendcount, datatype, comm,
                            stream);
}

EXPOSE_API
mcclResult_t mcclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t recvcount, mcclDataType_t datatype,
                               mcclRedOp_t op, mcclComm_t comm,
                               cudaStream_t stream) {
  SETUP_DLSYM_WITH_NCCL(mcclReduceScatter);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_mcclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op,
                                  comm, stream);

  // Get more NCCL info in advance.
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .intercept_manager.interceptNcclInfo<xpu_timer::constant::SKIP_TP>(
          recvcount, datatype, comm, stream);

  return orig_mcclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op,
                                comm, stream);
}

EXPOSE_API
mcclResult_t mcclSend(const void* sendbuff, size_t count,
                      mcclDataType_t datatype, int peer, mcclComm_t comm,
                      cudaStream_t stream) {
  SETUP_DLSYM_WITH_NCCL(mcclSend);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_mcclSend(sendbuff, count, datatype, peer, comm, stream);

  // Get more NCCL info in advance.
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .intercept_manager.interceptNcclInfo(
          count, datatype, comm, stream,
          xpu_timer::maca::InterceptManager::SendRecvType::Send);

  return orig_mcclSend(sendbuff, count, datatype, peer, comm, stream);
}

EXPOSE_API
mcclResult_t mcclRecv(void* recvbuff, size_t count, mcclDataType_t datatype,
                      int peer, mcclComm_t comm, cudaStream_t stream) {
  SETUP_DLSYM_WITH_NCCL(mcclRecv);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_mcclRecv(recvbuff, count, datatype, peer, comm, stream);

  // Get more NCCL info in advance.
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .intercept_manager.interceptNcclInfo(
          count, datatype, comm, stream,
          xpu_timer::maca::InterceptManager::SendRecvType::Recv);

  return orig_mcclRecv(recvbuff, count, datatype, peer, comm, stream);
}

EXPOSE_API
mcclResult_t mcclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           mcclDataType_t datatype, int root, mcclComm_t comm,
                           cudaStream_t stream) {
  SETUP_DLSYM_WITH_NCCL(mcclBroadcast);
  if (!::xpu_timer::util::config::GlobalConfig::enable)
    return orig_mcclBroadcast(sendbuff, recvbuff, count, datatype, root, comm,
                              stream);

  // Get more NCCL info in advance.
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .intercept_manager.interceptNcclInfo<xpu_timer::constant::SKIP_TP>(
          count, datatype, comm, stream);

  return orig_mcclBroadcast(sendbuff, recvbuff, count, datatype, root, comm,
                            stream);
}

EXPOSE_API
cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream) {
  SETUP_DLSYM(cudaFreeAsync);
  auto fn = xpu_timer::GpuTimerManager<
                xpu_timer::maca::MacaGpuTimer>::getInstance()
                .intercept_manager.deviceMemory("cudaFreeAsync", 1, "", false);

  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::maca::MacaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(stream, fn, xpu_timer::constant::Metrics::MemMetrics::TYPE);
  auto status = orig_cudaFreeAsync(devPtr, stream);
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream) {
  SETUP_DLSYM(cudaMemcpyAsync);
  // https://docs.maca.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b
  // cudaMemcpyHostToHost = 0   Host -> Host
  // cudaMemcpyHostToDevice = 1   Host -> Device
  // cudaMemcpyDeviceToHost = 2   Device -> Host
  // cudaMemcpyDeviceToDevice = 3   Device -> Device
  static std::vector<std::string> copy_kind{"H2H", "H2D", "D2H", "D2D",
                                            "Unkonwn"};
  auto fn = xpu_timer::GpuTimerManager<
                xpu_timer::maca::MacaGpuTimer>::getInstance()
                .intercept_manager.deviceMemory("cudaMemcpyAsync", count,
                                                copy_kind[kind], false);

  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::maca::MacaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(stream, fn, xpu_timer::constant::Metrics::MemMetrics::TYPE);
  auto status = orig_cudaMemcpyAsync(dst, src, count, kind, stream);

  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
cudaError_t cudaMalloc(void** devPtr, size_t size) {
  SETUP_DLSYM(cudaMalloc);
  auto fn = xpu_timer::GpuTimerManager<
                xpu_timer::maca::MacaGpuTimer>::getInstance()
                .intercept_manager.deviceMemory("cudaMalloc", size, "", true);
  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::maca::MacaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(fn, xpu_timer::constant::Metrics::MemMetrics::TYPE);
  auto status = orig_cudaMalloc(devPtr, size);
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
cudaError_t cudaFree(void* devPtr) {
  SETUP_DLSYM(cudaFree);
  auto fn = xpu_timer::GpuTimerManager<
                xpu_timer::maca::MacaGpuTimer>::getInstance()
                .intercept_manager.deviceMemory("cudaFree", 1, "", true);

  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::maca::MacaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(fn, xpu_timer::constant::Metrics::MemMetrics::TYPE);
  auto status = orig_cudaFree(devPtr);
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size,
                                    cudaMemPool_t memPool,
                                    cudaStream_t stream) {
  SETUP_DLSYM(cudaMallocFromPoolAsync);
  auto fn = xpu_timer::GpuTimerManager<
                xpu_timer::maca::MacaGpuTimer>::getInstance()
                .intercept_manager.deviceMemory("cudaMallocFromPoolAsync", size,
                                                "", false);

  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::maca::MacaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(stream, fn, xpu_timer::constant::Metrics::MemMetrics::TYPE);
  auto status = orig_cudaMallocFromPoolAsync(ptr, size, memPool, stream);
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
  SETUP_DLSYM(cudaHostAlloc);
  auto fn =
      xpu_timer::GpuTimerManager<
          xpu_timer::maca::MacaGpuTimer>::getInstance()
          .intercept_manager.deviceMemory("cudaHostAlloc", size, "", true);

  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::maca::MacaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(fn, xpu_timer::constant::Metrics::MemMetrics::TYPE);
  auto status = orig_cudaHostAlloc(ptr, size, flags);
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

EXPOSE_API
cudaError_t cudaMallocHost(void** ptr, size_t size) {
  SETUP_DLSYM(cudaMallocHost);
  auto fn =
      xpu_timer::GpuTimerManager<
          xpu_timer::maca::MacaGpuTimer>::getInstance()
          .intercept_manager.deviceMemory("cudaMallocHost", size, "", true);

  auto event = xpu_timer::GpuTimerManager<
                   xpu_timer::maca::MacaGpuTimer>::getInstance()
                   .getEvent();
  event->reset(fn, xpu_timer::constant::Metrics::MemMetrics::TYPE);
  auto status = orig_cudaMallocHost(ptr, size);
  xpu_timer::GpuTimerManager<xpu_timer::maca::MacaGpuTimer>::getInstance()
      .recordEvent(event);
  return status;
}

#ifdef __cplusplus
}
#endif
