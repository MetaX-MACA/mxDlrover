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

#include "xpu_timer/maca/maca_dtype_util.h"

namespace xpu_timer {
namespace maca {

const std::string CudaDataTypeUtils::UNKNOWN_CUDA_DTYPE = "UNKNOWN";
std::string CudaDataTypeUtils::gpu_ = "";

const std::unordered_map<cudaDataType_t, std::string>
    CudaDataTypeUtils::cudaDataTypeToStringMap = {
        {CUDA_R_16F, "fp16"},         {CUDA_C_16F, "complex_fp16"},
        {CUDA_R_16BF, "bf16"},        {CUDA_C_16BF, "complex_bf16"},
        {CUDA_R_32F, "fp32"},         {CUDA_C_32F, "complex_fp32"},
        {CUDA_R_64F, "fp64"},         {CUDA_C_64F, "complex_fp64"},
        {CUDA_R_8I, "int8"},          {CUDA_C_8I, "complex_int8"},
        {CUDA_R_8U, "uint8"},         {CUDA_C_8U, "complex_uint8"},
        {CUDA_R_32I, "int32"},        {CUDA_C_32I, "complex_int32"},
};

const std::unordered_map<ncclDataType_t, std::string>
    CudaDataTypeUtils::ncclDataTypeToStringMap = {
        {ncclInt8, "int8"},
        {ncclChar, "int8"},
        {ncclUint8, "uint8"},
        {ncclInt32, "int32"},
        {ncclInt, "int32"},
        {ncclUint32, "uint32"},
        {ncclInt64, "int64"},
        {ncclUint64, "uint64"},
        {ncclFloat16, "fp16"},
        {ncclHalf, "fp16"},
        {ncclFloat32, "fp32"},
        {ncclFloat, "fp32"},
        {ncclFloat64, "fp64"},
        {ncclDouble, "fp64"},
#if defined(__CUDA_BF16_TYPES_EXIST__)
        {ncclBfloat16, "bf16"},
#endif
        {ncclNumTypes, UNKNOWN_CUDA_DTYPE},
};

const std::unordered_map<std::string, uint64_t>
    CudaDataTypeUtils::dtypeSizeInBytes = {
        {"fp16", 2},
        {"bf16", 2},
        {"fp64", 8},
        {"fp32", 4},
        {"int8", 1},
        {"int32", 4},
        {"int64", 8},
        {"uint8", 1},
        {"uint32", 4},
        {"uint64", 8},
        {UNKNOWN_CUDA_DTYPE, 0},
};

const std::unordered_map<std::string, std::unordered_map<std::string, double>>
    CudaDataTypeUtils::gpuHardwareFlops = {
        {"C500",
         {
             {"fp16", 240},
             {"bf16", 240},
             {"fp32", 30},
             {"fp64", 15},
         }},
        {"C550",
         {
             {"fp16", 280},
             {"bf16", 280},
             {"fp32", 36},
             {"fp64", 18},
         }},
};

// Implementations of static methods
const std::string& CudaDataTypeUtils::getCudaDtype(cudaDataType_t dtype) {
  auto it = cudaDataTypeToStringMap.find(dtype);
  return it == cudaDataTypeToStringMap.end() ? UNKNOWN_CUDA_DTYPE : it->second;
}

const std::string& CudaDataTypeUtils::getNcclDataType(
    const ncclDataType_t& dtype) {
  auto it = ncclDataTypeToStringMap.find(dtype);
  return it == ncclDataTypeToStringMap.end() ? UNKNOWN_CUDA_DTYPE : it->second;
}

uint64_t CudaDataTypeUtils::getDtypeSizeInBytes(const std::string& dtype) {
  auto it = dtypeSizeInBytes.find(dtype);
  return it == dtypeSizeInBytes.end() ? 0 : it->second;
}

void CudaDataTypeUtils::setGpu(const std::string& gpu) { gpu_ = gpu; }

double CudaDataTypeUtils::getGpuHardwareFlops(const std::string& dtype) {
  static const std::unordered_map<std::string, double>* gpu_ptr = nullptr;
  if (!gpu_ptr) {
    auto it = gpuHardwareFlops.find(gpu_);
    if (it != gpuHardwareFlops.end()) {
      gpu_ptr = &it->second;
    } else {
      gpu_ptr = &gpuHardwareFlops.at("C500");
    }
  }

  auto it = gpu_ptr->find(dtype);
  if (it != gpu_ptr->end()) {
    return it->second;
  }
  // defaults to hafl on A100
  return 312.;
}

}  // namespace maca
}  // namespace xpu_timer
