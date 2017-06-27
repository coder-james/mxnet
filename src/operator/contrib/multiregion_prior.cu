/*!
 * Copyright (c) 2017 by Contributors
 * \file multiregion_prior.cu
 * \brief generate multiregion prior boxes cuda kernels
 * \author Yidong Ma
*/

#include "./multiregion_prior-inl.h"
#include <mshadow/cuda/tensor_gpu-inl.cuh>

#define MULTIREGIONPRIOR_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

namespace mshadow {
namespace cuda {
template<typename DType>
__global__ void AssignPriors(DType *out, const float sw, const float sh,
                             const int in_width, const int in_height, 
			     const int stride, const int offset,
			     const int channels) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= in_width * in_height) return;
  int r = index / in_width;
  int c = index % in_width;
  DType *ptr = out + index * stride + offset * channels;
  ptr[0] = c;  // left x
  ptr[1] = r;  // top y
  ptr[2] = sw;  // bias w
  ptr[3] = sh;  // bias h
  ptr[4] = in_width; //feature width
  ptr[5] = in_height; //feature height
}
}  // namespace cuda

template<typename DType>
inline void MultiRegionPriorForward(const Tensor<gpu, 2, DType> &out,
                            const std::vector<float> &anchors,
                            const int in_width, const int in_height,
			    const int channels) {
  CHECK_EQ(out.CheckContiguous(), true);
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  DType *out_ptr = out.dptr_;
  const int num_ac = static_cast<int>(anchors.size());

  const int num_thread = cuda::kMaxThreadsPerBlock;
  dim3 dimBlock(num_thread);
  dim3 dimGrid((in_width * in_height - 1) / num_thread + 1);
  cuda::CheckLaunchParam(dimGrid, dimBlock, "MultiRegionPrior Forward");

  const int stride = (num_ac / 2) * channels;
  for (int i = 0; i < num_ac / 2; ++i) {
    cuda::AssignPriors<DType><<<dimGrid, dimBlock, 0, stream>>>(out_ptr,
      anchors[2 * i], anchors[2 * i + 1], in_width, in_height, stride, i, channels);
  }
  MULTIREGIONPRIOR_CUDA_CHECK(cudaPeekAtLastError());
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(MultiRegionPriorParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiRegionPriorOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
