#ifndef MXNET_DEPENDENCE_CUH_
#define MXNET_DEPENDENCE_CUH_
#include <torch/torch.h> 
#include <ATen/ATen.h>
#include <TH/TH.h>
#include <ATen/cuda/detail/KernelUtils.h>
//#include <caffe2/utils/math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

#define CHECK_LT(x,y) AT_ASSERTM(x<y, #x " not < " #y)

//typedef int64_t index_t;

// clone from caffe2/code/Tensorlmpl.h
/**
 * Return product of all dimensions starting from k
 */
inline int64_t size_from_dim_(int k, at::IntList dims) {
    int64_t r = 1;
    for (size_t i = k; i < dims.size(); ++i) {
        r *= dims[i];
    }
    return r;
}

/*
/// pytoch define CUDA_KERNEL_LOOP in <THCUNN/common.h>
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)
*/

#define cuda_get_num_blocks at::cuda::detail::GET_BLOCKS
#define CUDA_NUM_THREADS at::cuda::detail::CUDA_NUM_THREADS

#endif  // MXNET_DEPENDENCE_CUH_
