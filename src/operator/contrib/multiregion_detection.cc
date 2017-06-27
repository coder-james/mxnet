/*!
 * Copyright (c) 2017 by Contributors
 * \file multiregion_detection.cc
 * \brief multiregion detection cpu implement
 * \author Yidong Ma
*/
#include "./multiregion_detection-inl.h"
#include <algorithm>

namespace mshadow {
template<typename DType>
struct SortElemDescend {
  DType value;
  int index;

  SortElemDescend(DType v, int i) {
    value = v;
    index = i;
  }

  bool operator<(const SortElemDescend &other) const {
    return value > other.value;
  }
};

template<typename DType>
inline void TransformLocations(DType *out, const DType *anchors,
                               const DType *dxy_pred, const DType *wh_pred) {
  // transform predictions to detection results
  DType al = anchors[0];
  DType at = anchors[1];
  DType asw = anchors[2];
  DType ash = anchors[3];
  DType aW = anchors[4];
  DType aH = anchors[5];
  DType px = dxy_pred[0];
  DType py = dxy_pred[1];
  DType pw = wh_pred[0];
  DType ph = wh_pred[1];
  DType ox = (px + al) / aW;
  DType oy = (py + at) / aH;
  DType ow = exp(pw) * asw / 2;
  DType oh = exp(ph) * ash / 2;
  out[0] = std::max(DType(0), std::min(DType(1), ox - ow));
  out[1] = std::max(DType(0), std::min(DType(1), oy - oh));
  out[2] = std::max(DType(0), std::min(DType(1), ox + ow));
  out[3] = std::max(DType(0), std::min(DType(1), oy + oh));
}

template<typename DType>
inline DType CalculateOverlap(const DType *a, const DType *b) {
  DType w = std::max(DType(0), std::min(a[2], b[2]) - std::max(a[0], b[0]));
  DType h = std::max(DType(0), std::min(a[3], b[3]) - std::max(a[1], b[1]));
  DType i = w * h;
  DType u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
  return u <= 0.f ? static_cast<DType>(0) : static_cast<DType>(i / u);
}

template<typename DType>
inline void MultiRegionDetectionForward(const Tensor<cpu, 3, DType> &out,
                                     const Tensor<cpu, 2, DType> &obj_pred,
                                     const Tensor<cpu, 2, DType> &dxy_pred,
                                     const Tensor<cpu, 2, DType> &wh_pred,
                                     const Tensor<cpu, 2, DType> &cls_pred,
                                     const Tensor<cpu, 2, DType> &anchors,
                                     const Tensor<cpu, 3, DType> &temp_space,
                                     const float threshold,
                                     const float nms_threshold,
				     const int num_classes,
                                     const bool force_suppress,
                                     const int nms_topk) {
  const int num_batches = obj_pred.size(0);
  const int num_anchors = obj_pred.size(1);
  const int num_anchor_channel = anchors.size(1);
  const int out_channel = out.size(2);
  const DType *p_anchor = anchors.dptr_;
  for (int nbatch = 0; nbatch < num_batches; ++nbatch) {
    const DType *p_obj_pred = obj_pred.dptr_ + nbatch * num_anchors;
    const DType *p_dxy_pred = dxy_pred.dptr_ + nbatch * num_anchors * 2;
    const DType *p_wh_pred = wh_pred.dptr_ + nbatch * num_anchors * 2;
    const DType *p_cls_pred = cls_pred.dptr_ + nbatch * num_anchors * num_classes;
    DType *p_out = out.dptr_ + nbatch * num_anchors * out_channel;
    int valid_count = 0;
    for (int i = 0; i < num_anchors; ++i) {
      DType score = 0.0f;
      int id = -1;
      DType otmp = p_obj_pred[i];
      for (int j = 0; j < num_classes; ++j) {
        DType ctmp = p_cls_pred[i * num_classes + j];
	DType temp = otmp * ctmp;
        if (temp > score) {
          score = temp;
          id = j;
        }
      }
      if (id != -1 && score >= threshold) {
        p_out[valid_count * out_channel] = id;
        p_out[valid_count * out_channel + 1] = score;
        TransformLocations(p_out + valid_count * out_channel + 2, p_anchor + i * num_anchor_channel,
          p_dxy_pred + i * 2, p_wh_pred + i * 2);
        ++valid_count;
      }
    }  // end iter num_anchors

    if (valid_count < 1 || nms_threshold <= 0 || nms_threshold > 1) continue;

    Copy(temp_space[nbatch], out[nbatch], out.stream_);
    // sort confidence in descend order
    std::vector<SortElemDescend<DType>> sorter;
    sorter.reserve(valid_count);
    for (int i = 0; i < valid_count; ++i) {
      sorter.push_back(SortElemDescend<DType>(p_out[i * out_channel + 1], i));
    }
    std::stable_sort(sorter.begin(), sorter.end());
    DType *ptemp = temp_space.dptr_ + nbatch * num_anchors * out_channel;
    int nkeep = static_cast<int>(sorter.size());
    if (nms_topk > 0 && nms_topk < nkeep) {
      nkeep = nms_topk;
    }
    for (int i = 0; i < nkeep; ++i) {
      for (int j = 0; j < out_channel; ++j) {
        p_out[i * out_channel + j] = ptemp[sorter[i].index * out_channel + j];
      }
    }
    // apply nms
    for (int i = 0; i < valid_count; ++i) {
      int offset_i = i * out_channel;
      if (p_out[offset_i] < 0) continue;  // skip eliminated
      for (int j = i + 1; j < valid_count; ++j) {
        int offset_j = j * out_channel;
        if (p_out[offset_j] < 0) continue;  // skip eliminated
        if (force_suppress || (p_out[offset_i] == p_out[offset_j])) {
          // when foce_suppress == true or class_id equals
          DType iou = CalculateOverlap(p_out + offset_i + 2, p_out + offset_j + 2);
          if (iou >= nms_threshold) {
            p_out[offset_j] = -1;
          }
        }
      }
    }
  }  // end iter batch
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(MultiRegionDetectionParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiRegionDetectionOp<cpu, DType>(param);
  });
  return op;
}

Operator* MultiRegionDetectionProp::CreateOperatorEx(Context ctx,
                                                  std::vector<TShape> *in_shape,
                                                  std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(MultiRegionDetectionParam);
MXNET_REGISTER_OP_PROPERTY(_contrib_MultiRegionDetection, MultiRegionDetectionProp)
.describe("Generate multiregion detection.")
.add_argument("obj_pred", "NDArray-or-Symbol", "Object regression predictions.")
.add_argument("dxy_pred", "NDArray-or-Symbol", "Dxy regression predictions.")
.add_argument("wh_pred", "NDArray-or-Symbol", "WH regression predictions.")
.add_argument("cls_pred", "NDArray-or-Symbol", "Class probabilities.")
.add_argument("anchor", "NDArray-or-Symbol", "Multiregion prior anchor boxes")
.add_arguments(MultiRegionDetectionParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
