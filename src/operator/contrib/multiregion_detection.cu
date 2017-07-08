/*!
 * Copyright (c) 2017 by Contributors
 * \file multiregion_detection.cu
 * \brief multiregion detection cuda kernels
 * \author Yidong Ma
*/
#include "./multiregion_detection-inl.h"
#include <mshadow/cuda/tensor_gpu-inl.cuh>

#define MULTIREGION_DETECTION_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

namespace mshadow {
namespace cuda {
template<typename DType>
__device__ void Clip(DType *value, const DType lower, const DType upper) {
  if ((*value) < lower) *value = lower;
  if ((*value) > upper) *value = upper;
}

template<typename DType>
__device__ void CalculateOverlap(const DType *a, const DType *b, DType *iou) {
  DType w = max(DType(0), min(a[2], b[2]) - max(a[0], b[0]));
  DType h = max(DType(0), min(a[3], b[3]) - max(a[1], b[1]));
  DType i = w * h;
  DType u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
  (*iou) =  u <= 0.f ? static_cast<DType>(0) : static_cast<DType>(i / u);
}

template<typename DType>
__global__ void DetectionForwardKernel(DType *out, const DType *obj_pred,
				       const DType *dxy_pred, const DType *wh_pred,
                                       const DType *cls_pred, const DType *anchors,
                                       DType *temp_space, const int num_classes,
                                       const int num_anchors, const int num_anchor_channel,
				       const int out_channel,
				       const float threshold, const float nms_threshold,
                                       const bool force_suppress, const int nms_topk) {
  const int nbatch = blockIdx.x; 
  int index = threadIdx.x;
  __shared__ int valid_count;
  out += nbatch * num_anchors * out_channel;
  obj_pred += nbatch * num_anchors;
  dxy_pred += nbatch * num_anchors * 2;
  wh_pred += nbatch * num_anchors * 2;
  cls_pred += nbatch * num_anchors * num_classes;

  if (index == 0) {
    valid_count = 0;
  }
  __syncthreads();

  for (int i = index; i < num_anchors; i += blockDim.x) {
    DType score = 0.0f;
    int id = -1;
    DType otmp = obj_pred[i];
    for (int j = 0; j < num_classes; ++j) {
      DType ctmp = cls_pred[i * num_classes + j];
      DType temp = otmp * ctmp;
      if (temp > score) {
        score = temp;
        id = j;
      }
    }
    if (id != -1 && score >= threshold) {
      // valid class
      int pos = atomicAdd(&valid_count, 1);
      DType *p_out = out + pos * out_channel;
      p_out[0] = id;  // restore original class id
      p_out[1] = score;
      int offset = i * 2;
      int offset_a = i * num_anchor_channel;
      DType al = anchors[offset_a];
      DType at = anchors[offset_a + 1];
      DType asw = anchors[offset_a + 2];
      DType ash = anchors[offset_a + 3];
      DType aW = anchors[offset_a + 4];
      DType aH = anchors[offset_a + 5];
      DType ox = (dxy_pred[offset] + al) / aW;
      DType oy = (dxy_pred[offset + 1] + at) / aH;
      DType ow = exp(wh_pred[offset]) * asw / 2;
      DType oh = exp(wh_pred[offset + 1]) * ash / 2;
      DType xmin = ox - ow;
      DType ymin = oy - oh;
      DType xmax = ox + ow;
      DType ymax = oy + oh;
      Clip(&xmin, DType(0), DType(1));
      Clip(&ymin, DType(0), DType(1));
      Clip(&xmax, DType(0), DType(1));
      Clip(&ymax, DType(0), DType(1));
      p_out[2] = xmin;
      p_out[3] = ymin;
      p_out[4] = xmax;
      p_out[5] = ymax;
    }
  }
  __syncthreads();

  if (valid_count < 1 || nms_threshold <= 0 || nms_threshold > 1) return;
  //if (index == 0) printf("%d\n", valid_count);
  const int size = valid_count;
  temp_space += nbatch * num_anchors * out_channel;
  DType *src = out;
  DType *dst = temp_space;
  for (int width = 2; width < (size << 1); width <<= 1) {
    int slices = (size - 1) / (blockDim.x * width) + 1;
    int start = width * index * slices;
    for (int slice = 0; slice < slices; ++slice) {
      if (start >= size) break;
      int middle = start + (width >> 1);
      if (middle > size) middle = size;
      int end = start + width;
      if (end > size) end = size;
      int i = start;
      int j = middle;
      for (int k = start; k < end; ++k) {
        DType score_i = i < size ? src[i * out_channel + 1] : DType(-1);
        DType score_j = j < size ? src[j * out_channel + 1] : DType(-1);
        if (i < middle && (j >= end || score_i > score_j)) {
          for (int n = 0; n < out_channel; ++n) {
            dst[k * out_channel + n] = src[i * out_channel + n];
          }
          ++i;
        } else {
          for (int n = 0; n < out_channel; ++n) {
            dst[k * out_channel + n] = src[j * out_channel + n];
          }
          ++j;
        }
      }
      start += width;
    }
    __syncthreads();
    src = src == out? temp_space : out;
    dst = dst == out? temp_space : out;
  }
  __syncthreads();

  if (src == temp_space) {
    for (int i = index; i < size * out_channel; i += blockDim.x) {
      out[i] = temp_space[i];
    }
    __syncthreads();
  }

  int ntop = size;
  if (nms_topk > 0 && nms_topk < ntop) {
    ntop = nms_topk;
    for (int i = ntop + index; i < size; i += blockDim.x) {
      out[i * out_channel] = -1;
    }
    __syncthreads();
  }

  // apply NMS
  for (int compare_pos = 0; compare_pos < ntop; ++compare_pos) {
    DType compare_id = out[compare_pos * out_channel];
    if (compare_id < 0) continue; 
    DType *compare_loc_ptr = out + compare_pos * out_channel + 2;
    for (int i = compare_pos + index + 1; i < ntop; i += blockDim.x) {
      DType class_id = out[i * out_channel];
      if (class_id < 0) continue;
      if (force_suppress || (class_id == compare_id)) {
        DType iou;
        CalculateOverlap(compare_loc_ptr, out + i * out_channel + 2, &iou);
        if (iou >= nms_threshold) {
          out[i * out_channel] = -1;
        }
      }
    }
    __syncthreads();
  }
}
}  // namespace cuda

template<typename DType>
inline void MultiRegionDetectionForward(const Tensor<gpu, 3, DType> &out,
                                     const Tensor<gpu, 2, DType> &obj_pred,
                                     const Tensor<gpu, 2, DType> &dxy_pred,
                                     const Tensor<gpu, 2, DType> &wh_pred,
                                     const Tensor<gpu, 2, DType> &cls_pred,
                                     const Tensor<gpu, 2, DType> &anchors,
                                     const Tensor<gpu, 3, DType> &temp_space,
                                     const float threshold,
                                     const float nms_threshold,
				     const int num_classes,
                                     const bool force_suppress,
                                     const int nms_topk) {
  const int num_batches = obj_pred.size(0);
  const int num_anchors = obj_pred.size(1);
  const int num_anchor_channel = anchors.size(1);
  const int out_channel = out.size(2);
  const int num_threads = cuda::kMaxThreadsPerBlock;
  int num_blocks = num_batches;
  cuda::CheckLaunchParam(num_blocks, num_threads, "MultiRegionDetection Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  cuda::DetectionForwardKernel<<<num_blocks, num_threads, 0, stream>>>(out.dptr_,
    obj_pred.dptr_, dxy_pred.dptr_, wh_pred.dptr_, cls_pred.dptr_, anchors.dptr_, 
    temp_space.dptr_, num_classes, num_anchors, num_anchor_channel, out_channel,
    threshold, nms_threshold, force_suppress, nms_topk);
  MULTIREGION_DETECTION_CUDA_CHECK(cudaPeekAtLastError());
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(MultiRegionDetectionParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiRegionDetectionOp<gpu, DType>(param);
  });
  return op;
}
}  // namespace op
}  // namespace mxnet
