/*!
 * Copyright (c) 2017 by Contributors
 * \file multiregion_target.cu
 * \brief genuerate multiregion target label cuda kernels
 * \author Yidong Ma
*/
#include "./multiregion_target-inl.h"
#include <mshadow/cuda/tensor_gpu-inl.cuh>

#define MULTIREGION_TARGET_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

namespace mshadow {
namespace cuda {
template<typename DType>
__device__ void MatchIOU(const DType *anchors, const DType *wh, const DType *label, DType *iou){
  float asw = anchors[2];
  float ash = anchors[3];
  float pw = exp(wh[0]) * asw / 2;
  float ph = exp(wh[1]) * ash / 2;
  float gw = (label[3] - label[1]) / 2;
  float gh = (label[4] - label[2]) / 2;
  float w = max(0.f, min(pw, gw) - max(-pw, -gw));
  float h = max(0.f, min(ph, gh) - max(-ph, -gh));
  float i = w * h;
  float u = 4 * pw * ph + 4 * gw * gh - i;
  (*iou) = u <= 0.f ? static_cast<DType>(0) : static_cast<DType>(i / u);
}
template<typename DType>
__device__ void CalculateIOU(const DType *anchors, const DType *dxy, const DType *wh, const DType *label, DType *iou) {
  float al = anchors[0];
  float at = anchors[1];
  float asw = anchors[2];
  float ash = anchors[3];
  float aW = anchors[4];
  float aH = anchors[5];
  float px = dxy[0];
  float py = dxy[1];
  float pw = wh[0];
  float ph = wh[1];
  float ox = (px + al) / aW;
  float oy = (py + at) / aH;
  float ow = exp(pw) * asw / 2;
  float oh = exp(ph) * ash / 2;
  float xmin = max(0.f, min(1.f, ox - ow));
  float ymin = max(0.f, min(1.f, oy - oh));
  float xmax = max(0.f, min(1.f, ox + ow));
  float ymax = max(0.f, min(1.f, oy + oh));
  float w = max(0.f, min(xmax, label[3]) - max(xmin, label[1]));
  float h = max(0.f, min(ymax, label[4]) - max(ymin, label[2]));
  float i = w * h;
  float u = (xmax - xmin) * (ymax - ymin) + (label[3] - label[1]) * (label[4] - label[2]) - i;
  (*iou) =  u <= 0.f ? static_cast<DType>(0) : static_cast<DType>(i / u);
}

template<typename DType>
__global__ void InitGroundTruthFlags(DType *gt_flags, const DType *labels,
                                     const int num_batches,
                                     const int num_labels,
                                     const int label_width) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_batches * num_labels) return;
  int b = index / num_labels;
  int l = index % num_labels;
  if (labels[b * num_labels * label_width + l * label_width] == -1.f) {
    gt_flags[b * num_labels + l] = 0;
  } else {
    gt_flags[b * num_labels + l] = 1;
  }
}

template<typename DType>
__global__ void AssignObjTargets(DType *nobj_target, const DType *obj_target,
				const DType *gt_flags, const DType *dxy_target,
				const DType *wh_target, const DType *anchors,
				const DType *labels, const int label_width,
                                const int num_anchors, const int num_anchor_channel, 
				const int num_labels,const float threshold) {
  int nbatch = blockIdx.x;
  gt_flags += nbatch * num_labels;
  dxy_target += nbatch * num_anchors * 2;
  wh_target += nbatch * num_anchors * 2;
  nobj_target += nbatch * num_anchors;
  obj_target += nbatch * num_anchors;
  labels += nbatch * num_labels * label_width;
  const int num_threads = kMaxThreadsPerBlock;

  for (int i = threadIdx.x; i < num_anchors; i += num_threads) {
    DType max_value = 1e-6;  // start with very small overlap
    for (int j = 0; j < num_labels; ++j) {
      if (gt_flags[j] > .5) {
	DType iou;
	CalculateIOU(anchors + i * num_anchor_channel, dxy_target + i * 2, wh_target + i * 2, labels + j * label_width, &iou);
        if (iou > max_value) {
          max_value = iou;
        }
      }
    }
    if(max_value > threshold){
      nobj_target[i] = obj_target[i];
    }
  }
  if(nbatch == 0 && threadIdx.x == 0){
    float avg_anyobj = 0.f;
    for(int i = 0;i < num_anchors;i++)
      avg_anyobj += obj_target[i];
    printf("NObj:%.4f,", avg_anyobj/num_anchors);
  }
}

template<typename DType>
__global__ void AssignTrainingTargets(DType *obj_target, DType *nobj_target,
				     DType *dxy_target, DType *wh_target,
                                     DType *cls_target, const DType *gt_flags, 
				     const DType *labels, const DType *anchors, 
				     DType *value, const int num_anchors,
				     const int num_anchor_channel, const int num_labels, 
				     const int label_width, const int num_classes) {
  const int nbatch = blockIdx.x;
  obj_target += nbatch * num_anchors;
  nobj_target += nbatch * num_anchors;
  dxy_target += nbatch * num_anchors * 2;
  wh_target += nbatch * num_anchors * 2;
  cls_target += nbatch * num_anchors * num_classes;
  labels += nbatch * num_labels * label_width;
  gt_flags += nbatch * num_labels;
  value += nbatch * num_labels * 5; //avg_iou, recall, avg_cat, avg_obj, mark
  const int num_threads = kMaxThreadsPerBlock;

  for (int i = threadIdx.x; i < num_labels; i += num_threads) {
    DType *p_value = value + i * 5;
    const DType *p_label = labels + i * label_width;
    if (gt_flags[i] > .5){
      DType max_overlap = 1e-6;
      int best_anchor = -1;
      float txmin = p_label[1];
      float tymin = p_label[2];
      float txmax = p_label[3];
      float tymax = p_label[4];
      for (int j = 0; j < num_anchors; ++j){
	const DType *p_anchor = anchors + j * num_anchor_channel;
	int l = p_anchor[0];
	int t = p_anchor[1];
	int W = p_anchor[4];
	int H = p_anchor[5];
	int gx = (txmax + txmin) / 2 * W;
	int gy = (tymax + tymin) / 2 * H;
	if(l == gx && t == gy){
          DType iou;
	  MatchIOU(p_anchor, wh_target + j * 2, p_label, &iou);
          if(iou > max_overlap){
            max_overlap = iou;
	    best_anchor = j;
          }
	}
      }
      if(best_anchor != -1){
	p_value[0] = max_overlap;
	if(max_overlap > .5)
	  p_value[1] = 1;
 	int tlabel = p_label[0];
	int offset_c = best_anchor * num_classes;
	p_value[2] = cls_target[offset_c + tlabel];
	p_value[3] = obj_target[best_anchor];
	p_value[4] = 1;

	nobj_target[best_anchor] = obj_target[best_anchor];
	obj_target[best_anchor] = max_overlap;
	int offset_a = best_anchor * num_anchor_channel;
        float al = anchors[offset_a];
        float at = anchors[offset_a + 1];
        float abw = anchors[offset_a + 2];
        float abh = anchors[offset_a + 3];
        float fw = anchors[offset_a + 4];
        float fh = anchors[offset_a + 5];
        float gl = p_label[1];
        float gt = p_label[2];
        float gr = p_label[3];
        float gb = p_label[4];
        float gw = gr - gl;
        float gh = gb - gt;
        float gx = (gl + gr) * 0.5;
        float gy = (gt + gb) * 0.5;
	int offset = best_anchor * 2;
        dxy_target[offset] = DType(gx*fw - al);
        dxy_target[offset + 1] = DType(gy*fh - at);
        wh_target[offset] = DType(log(gw / abw)); 
        wh_target[offset + 1] = DType(log(gh / abh));
	for(int c = 0;c < num_classes; c++)
	  cls_target[offset_c + c] = 0;
	cls_target[offset_c + tlabel] = 1;
      }
    }
  }
  __syncthreads();
  if(nbatch == 0 && threadIdx.x == 0){
    float avg_iou = 0.f, recall = 0.f, avg_cat = 0.f, avg_obj = 0.f;
    int count = 0;
    for(int j = 0;j < num_labels;j++){
      const DType *temp = value + j * 5;
      if(temp[4] == 1){
        avg_iou += temp[0];
        recall += temp[1];
        avg_cat += temp[2];
        avg_obj += temp[3];
        count++;
      }
    } 
    printf("Obj:%.4f,Avg IOU:%.4f,Class:%.4f,Avg Recall:%.4f,count:%.4d\n", avg_obj/count, avg_iou/count, avg_cat/count, recall/count, count);
  }
}
}  // namespace cuda

template<typename DType>
inline void MultiRegionTargetForward(const Tensor<gpu, 2, DType> &obj_target,
                           const Tensor<gpu, 2, DType> &nobj_target,
                           const Tensor<gpu, 2, DType> &dxy_target,
                           const Tensor<gpu, 2, DType> &wh_target,
                           const Tensor<gpu, 2, DType> &cls_target,
                           const Tensor<gpu, 2, DType> &anchors,
                           const Tensor<gpu, 3, DType> &labels,
                           const Tensor<gpu, 4, DType> &temp_space,
                           const float threshold,
                           const int num_classes) {
  const int num_batches = labels.size(0);
  const int num_labels = labels.size(1);
  const int label_width = labels.size(2);
  const int num_anchors = anchors.size(0);
  const int num_anchor_channel = anchors.size(1);
  CHECK_GE(num_batches, 1);
  CHECK_GT(num_labels, 2);
  CHECK_GE(num_anchors, 1);

  DType *gt_flags = temp_space[0].dptr_;
  const int num_threads = cuda::kMaxThreadsPerBlock;
  dim3 init_thread_dim(num_threads);
  dim3 init_block_dim((num_batches * num_labels - 1) / num_threads + 1);
  cuda::CheckLaunchParam(init_block_dim, init_thread_dim, "MultiRegionTarget Init");
  cuda::InitGroundTruthFlags<DType><<<init_block_dim, init_thread_dim>>>(
    gt_flags, labels.dptr_, num_batches, num_labels, label_width);
  MULTIREGION_TARGET_CUDA_CHECK(cudaPeekAtLastError());
 
  DType *value = temp_space[1].dptr_;
  cuda::CheckLaunchParam(num_batches, num_threads, "MultiRegionTarget Matching");
  cuda::AssignObjTargets<DType><<<num_batches, num_threads>>>(nobj_target.dptr_, obj_target.dptr_, 
    gt_flags, dxy_target.dptr_, wh_target.dptr_, anchors.dptr_, labels.dptr_, label_width,
    num_anchors, num_anchor_channel, num_labels, threshold);
  MULTIREGION_TARGET_CUDA_CHECK(cudaPeekAtLastError());

  cuda::AssignTrainingTargets<DType><<<num_batches, num_threads>>>(
    obj_target.dptr_, nobj_target.dptr_, dxy_target.dptr_, wh_target.dptr_, cls_target.dptr_, 
    gt_flags, labels.dptr_, anchors.dptr_, value, num_anchors, num_anchor_channel, num_labels, 
    label_width, num_classes);
  MULTIREGION_TARGET_CUDA_CHECK(cudaPeekAtLastError());
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(MultiRegionTargetParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiRegionTargetOp<gpu, DType, mshadow::op::identity>(param);
  });
  return op;
}
}  // namespace op
}  // namespace mxnet
