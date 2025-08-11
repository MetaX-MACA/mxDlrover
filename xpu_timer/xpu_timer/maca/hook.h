// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
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

#pragma once
#include <dlfcn.h>

#include <functional>
#include <string>

#include "xpu_timer/common/macro.h"
#include "xpu_timer/common/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mcblasContext *mcblasHandle_t;
typedef struct mcblasLtContext *mcblasLtHandle_t;
// uint64 array, we use pointer to mock it
typedef uint64_t* mcblasLtMatmulDesc_t;
// uint64 array, we use pointer to mock it
typedef uint64_t* mcblasLtMatrixLayout_t;

typedef enum {} mcblasLtMatmulAlgo_t;

typedef enum {} mcblasStatus_t;

typedef enum {} mcblasOperation_t;

typedef enum {} mcblasGemmAlgo_t;

typedef enum {} mcblasComputeType_t;

typedef enum {
  CUBLASLT_MATRIX_LAYOUT_TYPE = 0,
  CUBLASLT_MATRIX_LAYOUT_ORDER = 1,
  CUBLASLT_MATRIX_LAYOUT_ROWS = 2,
  CUBLASLT_MATRIX_LAYOUT_COLS = 3,
  CUBLASLT_MATRIX_LAYOUT_LD = 4,
  CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,
  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6,
  CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET = 7,
} mcblasLtMatrixLayoutAttribute_t;

typedef enum {} mcblasLtMatmulDescAttributes_t;

typedef enum {} mcblasLtMatmulAlgoConfigAttributes_t;

typedef cudaError_t (*mcLaunchKernelFn)(const void *, dim3, dim3, void **,
                                          size_t, cudaStream_t);

// typedef cudaError_t (*cudaLaunchKernelExCFn)(const cudaLaunchConfig_t*,
                                            //  const void*, void**);
typedef mcblasStatus_t (*mcblasGemmExFn)(mcblasHandle_t, mcblasOperation_t,
                                         mcblasOperation_t, int, int, int,
                                         const void*, const void*, cudaDataType,
                                         int, const void*, cudaDataType, int,
                                         const void*, void*, cudaDataType, int,
                                         cudaDataType, mcblasGemmAlgo_t);
typedef mcblasStatus_t (*mcblasGetStreamFn)(mcblasHandle_t, cudaStream_t*);
typedef mcblasStatus_t (*mcblasGemmStridedBatchedExFn)(
    mcblasHandle_t, mcblasOperation_t, mcblasOperation_t, int, int, int,
    const void*, const void*, cudaDataType_t, int, long long int, const void*,
    cudaDataType_t, int, long long int, const void*, void*, cudaDataType_t, int,
    long long int, int, mcblasComputeType_t, mcblasGemmAlgo_t);
typedef mcblasStatus_t (*mcblasSgemmFn)(
    mcblasHandle_t handle, mcblasOperation_t transa, mcblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc);
typedef mcblasStatus_t (*mcblasSgemmStridedBatchedFn)(
    mcblasHandle_t handle, mcblasOperation_t transa, mcblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    long long int strideA, const float* B, int ldb, long long int strideB,
    const float* beta, float* C, int ldc, long long int strideC,
    int batchCount);

typedef mcblasStatus_t (*mcblasLtMatmulFn)(
    mcblasLtHandle_t lightHandle, mcblasLtMatmulDesc_t computeDesc,
    const void* alpha, const void* A, mcblasLtMatrixLayout_t Adesc,
    const void* B, mcblasLtMatrixLayout_t Bdesc, const void* beta,
    const void* C, mcblasLtMatrixLayout_t Cdesc, void* D,
    mcblasLtMatrixLayout_t Ddesc, const mcblasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream);

typedef mcblasStatus_t (*mcblasLtMatrixLayoutGetAttributeFn)(
    mcblasLtMatrixLayout_t matLayout, mcblasLtMatrixLayoutAttribute_t attr,
    void* buf, size_t sizeInBytes, size_t* sizeWritten);

typedef mcblasStatus_t (*mcblasLtMatmulAlgoConfigGetAttributeFn)(
    const mcblasLtMatmulAlgo_t* algo, mcblasLtMatmulAlgoConfigAttributes_t attr,
    void* buf, size_t sizeInBytes, size_t* sizeWritten);

typedef mcblasStatus_t (*mcblasLtMatmulDescGetAttributeFn)(
    mcblasLtMatmulDesc_t matmulDesc, mcblasLtMatmulDescAttributes_t attr,
    void* buf, size_t sizeInBytes, size_t* sizeWritten);

typedef mcclResult_t (*mcclAllReduceFn)(const void* sendbuff, void* recvbuff,
                                        size_t count, mcclDataType_t datatype,
                                        mcclRedOp_t op, mcclComm_t comm,
                                        cudaStream_t stream);
typedef mcclResult_t (*mcclReduceFn)(const void* sendbuff, void* recvbuff,
                                     size_t count, mcclDataType_t datatype,
                                     mcclRedOp_t op, int root, mcclComm_t comm,
                                     cudaStream_t stream);

typedef mcclResult_t (*mcclAllGatherFn)(const void* sendbuff, void* recvbuff,
                                        size_t sendcount,
                                        mcclDataType_t datatype,
                                        mcclComm_t comm, cudaStream_t stream);

typedef mcclResult_t (*mcclReduceScatterFn)(const void* sendbuff,
                                            void* recvbuff, size_t recvcount,
                                            mcclDataType_t datatype,
                                            mcclRedOp_t op, mcclComm_t comm,
                                            cudaStream_t stream);

typedef mcclResult_t (*mcclSendFn)(const void* sendbuff, size_t count,
                                   mcclDataType_t datatype, int peer,
                                   mcclComm_t comm, cudaStream_t stream);

typedef mcclResult_t (*mcclRecvFn)(void* recvbuff, size_t count,
                                   mcclDataType_t datatype, int peer,
                                   mcclComm_t comm, cudaStream_t stream);

typedef mcclResult_t (*mcclBroadcastFn)(const void* sendbuff, void* recvbuff,
                                        size_t count, mcclDataType_t datatype,
                                        int root, mcclComm_t comm,
                                        cudaStream_t stream);

typedef cudaError_t (*cudaMemcpyAsyncFn)(void* dst, const void* src,
                                         size_t count, cudaMemcpyKind kind,
                                         cudaStream_t stream);

typedef cudaError_t (*cudaFreeAsyncFn)(void* dst, cudaStream_t stream);

typedef cudaError_t (*cudaFreeFn)(void* devPtr);

typedef cudaError_t (*cudaMallocFn)(void** devPtr, size_t size);

typedef cudaError_t (*cudaMallocAsyncFn)(void** ptr, size_t size,
                                         cudaMemPool_t memPool,
                                         cudaStream_t stream);

typedef cudaError_t (*cudaMallocFromPoolAsyncFn)(void** ptr, size_t size,
                                                 cudaMemPool_t memPool,
                                                 cudaStream_t stream);
typedef cudaError_t (*cudaHostAllocFn)(void** ptr, size_t size,
                                       unsigned int flags);

typedef cudaError_t (*cudaMallocHostFn)(void** ptr, size_t size);

static mcblasGemmStridedBatchedExFn orig_mcblasGemmStridedBatchedEx = NULL;
static mcblasGetStreamFn orig_mcblasGetStream = NULL;

// static cudaLaunchKernelExCFn orig_mcLaunchKernelExC = NULL;

static mcLaunchKernelFn orig_mcLaunchKernel = NULL;
static mcblasGemmExFn orig_mcblasGemmEx = NULL;
static mcblasSgemmFn orig_mcblasSgemm = NULL;
static mcblasSgemmStridedBatchedFn orig_mcblasSgemmStridedBatched = NULL;
static mcblasLtMatmulFn orig_mcblasLtMatmul = NULL;
static mcblasLtMatrixLayoutGetAttributeFn
    orig_mcblasLtMatrixLayoutGetAttribute = NULL;
static mcblasLtMatmulAlgoConfigGetAttributeFn
    orig_mcblasLtMatmulAlgoConfigGetAttribute = NULL;
static mcblasLtMatmulDescGetAttributeFn orig_mcblasLtMatmulDescGetAttribute =
    NULL;
static mcclAllReduceFn orig_mcclAllReduce = NULL;
static mcclReduceFn orig_mcclReduce = NULL;
static mcclAllGatherFn orig_mcclAllGather = NULL;
static mcclReduceScatterFn orig_mcclReduceScatter = NULL;
static mcclSendFn orig_mcclSend = NULL;
static mcclRecvFn orig_mcclRecv = NULL;
static mcclBroadcastFn orig_mcclBroadcast = NULL;

static cudaMemcpyAsyncFn orig_cudaMemcpyAsync = NULL;
static cudaFreeAsyncFn orig_cudaFreeAsync = NULL;
static cudaMallocAsyncFn orig_cudaMallocAsync = NULL;

static cudaFreeFn orig_cudaFree = NULL;
static cudaMallocFn orig_cudaMalloc = NULL;
static cudaMallocFromPoolAsyncFn orig_cudaMallocFromPoolAsync = NULL;
static cudaHostAllocFn orig_cudaHostAlloc = NULL;
static cudaMallocHostFn orig_cudaMallocHost = NULL;

#ifdef __cplusplus
}
#endif
