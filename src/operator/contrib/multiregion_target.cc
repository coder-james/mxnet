/*!
 * Copyright (c) 2017 by Contributors
 * \file multiregion_target.cc
 * \brief generate multiregion target label cpu implement
 * \author Yidong Ma
*/
#include <algorithm>
#include "./multiregion_target-inl.h"
#include "../mshadow_op.h"

namespace mshadow {
template<typename DType>
inline void AssignLocTargets(const DType *anchor, const DType *l, DType *dxy, DType *wh) {
  float al = anchor[0];
  float at = anchor[1];
  float abw = anchor[2];
  float abh = anchor[3];
  float fw = anchor[4];
  float fh = anchor[5];
  float gl = l[1];
  float gt = l[2];
  float gr = l[3];
  float gb = l[4];
  float gw = gr - gl;
  float gh = gb - gt;
  float gx = (gl + gr) * 0.5;
  float gy = (gt + gb) * 0.5;
  dxy[0] = DType(gx * fw - al);
  dxy[1] = DType(gy * fh - at);
  wh[0] = DType(std::log(gw / abw));
  wh[1] = DType(std::log(gh / abh));
}
template<typename DType>
inline void MatchBiasIOU(const DType *anchors, const DType *label, DType *iou){
  float abw = anchors[2];
  float abh = anchors[3];
  float pw = abw / 2;
  float ph = abh / 2;
  float gw = (label[3] - label[1]) / 2;
  float gh = (label[4] - label[2]) / 2;
  float w = std::max(0.f, std::min(pw, gw) - std::max(-pw, -gw));
  float h = std::max(0.f, std::min(ph, gh) - std::max(-ph, -gh));
  float i = w * h;
  float u = 4 * pw * ph + 4 * gw * gh - i;
  (*iou) = u <= 0.f ? static_cast<DType>(0) : static_cast<DType>(i / u);
}
template<typename DType>
inline void MatchIOU(const DType *anchors, const DType *wh, const DType *label, DType *iou){
  //match weight/height predictions instead of anchor weight/height
  float abw = anchors[2];
  float abh = anchors[3];
  float pw = exp(wh[0]) * abw / 2;
  float ph = exp(wh[1]) * abh / 2;
  float gw = (label[3] - label[1]) / 2;
  float gh = (label[4] - label[2]) / 2;
  float w = std::max(0.f, std::min(pw, gw) - std::max(-pw, -gw));
  float h = std::max(0.f, std::min(ph, gh) - std::max(-ph, -gh));
  float i = w * h;
  float u = 4 * pw * ph + 4 * gw * gh - i;
  (*iou) = u <= 0.f ? static_cast<DType>(0) : static_cast<DType>(i / u);
}
template<typename DType>
inline void CalculateIOU(const DType *anchors, const DType *dxy, const DType *wh, const DType *label, DType *iou){
  float al = anchors[0];
  float at = anchors[1];
  float abw = anchors[2];
  float abh = anchors[3];
  float aW = anchors[4];
  float aH = anchors[5];
  float px = dxy[0];
  float py = dxy[1];
  float pw = wh[0];
  float ph = wh[1];
  float ox = (px + al) / aW;
  float oy = (py + at) / aH;
  float ow = exp(pw) * abw / 2;
  float oh = exp(ph) * abh / 2;
  float xmin = std::max(0.f, std::min(1.f, ox - ow));
  float ymin = std::max(0.f, std::min(1.f, oy - oh));
  float xmax = std::max(0.f, std::min(1.f, ox + ow));
  float ymax = std::max(0.f, std::min(1.f, oy + oh));
  float gxmin = label[1];
  float gymin = label[2];
  float gxmax = label[3];
  float gymax = label[4];
  float w = std::max(0.f, std::min(xmax, gxmax) - std::max(xmin, gxmin));
  float h = std::max(0.f, std::min(ymax, gymax) - std::max(ymin, gymin));
  float i = w * h;
  float u = (xmax - xmin) * (ymax - ymin) + (gxmax - gxmin) * (gymax - gymin) - i;
  (*iou) = u <= 0.f ? static_cast<DType>(0) : static_cast<DType>(i / u);
}
template<typename DType>
inline void MultiRegionTargetForward(const Tensor<cpu, 2, DType> &obj_target,
                           const Tensor<cpu, 2, DType> &nobj_target,
                           const Tensor<cpu, 2, DType> &dxy_target,
                           const Tensor<cpu, 2, DType> &wh_target,
                           const Tensor<cpu, 2, DType> &cls_target,
                           const Tensor<cpu, 2, DType> &anchors,
                           const Tensor<cpu, 3, DType> &labels,
                           const Tensor<cpu, 4, DType> &temp_space,
                           const float threshold,
                           const int num_classes,
			   const bool match_bias) {
  const DType *p_anchor = anchors.dptr_;
  const int num_batches = labels.size(0);
  const int num_labels = labels.size(1);
  const int label_width = labels.size(2);
  const int num_anchors = anchors.size(0);
  const int num_anchor_channel = anchors.size(1);
  //training metric
  float avg_iou = 0.0f, recall = 0.0f, avg_cat = 0.0f, avg_obj = 0.0f, avg_anyobj = 0.0f;
  int count = 0;
  for (int nbatch = 0; nbatch < num_batches; ++nbatch) {
    const DType *p_label = labels.dptr_ + nbatch * num_labels * label_width;
    int num_valid_gt = 0;
    for (int i = 0; i < num_labels; ++i) {
      if (static_cast<float>(p_label[i * label_width]) == -1.0f) {
        break;
      }
      ++num_valid_gt;
    }  // end iterate labels

    if (num_valid_gt > 0) {
      //target same as predicions
      DType *p_obj_target = obj_target.dptr_ + nbatch * num_anchors;
      DType *p_nobj_target = nobj_target.dptr_ + nbatch * num_anchors;
      DType *p_dxy_target = dxy_target.dptr_ + nbatch * num_anchors * 2;
      DType *p_wh_target = wh_target.dptr_ + nbatch * num_anchors * 2;
      DType *p_cls_target = cls_target.dptr_ + nbatch * num_anchors * num_classes;

      for(int i = 0;i < num_anchors;i++){
	  float best_iou = 1e-6;
          for(int j = 0;j < num_valid_gt;j++){
	    DType iou;
            CalculateIOU(p_anchor + i * num_anchor_channel, p_dxy_target + i * 2, p_wh_target + i * 2, p_label + j * label_width, &iou);
	    if(iou > best_iou)
		best_iou = iou;
          }
	  avg_anyobj += p_obj_target[i];
	  if(best_iou > threshold)
	    p_nobj_target[i] = p_obj_target[i];
      }//end iterate anchors
      for (int j = 0; j < num_valid_gt; ++j) {
        float max_overlap = 1e-6; 
	int best_anchor = -1;
	const DType *pp_label = p_label + j * label_width;
	float txmin = pp_label[1];
	float tymin = pp_label[2];
	float txmax = pp_label[3];
	float tymax = pp_label[4];
        for (int i = 0; i < num_anchors; ++i) {
	  const DType *pp_anchor = p_anchor + i * num_anchor_channel;
	  int l = pp_anchor[0];
	  int t = pp_anchor[1];
	  int W = pp_anchor[4];
	  int H = pp_anchor[5];
	  int gx = (txmax + txmin) / 2 * W;
	  int gy = (tymax + tymin) / 2 * H;
          //match between gt center locations and anchor boxes
	  if(l == gx && t == gy){
	      DType iou;
              if(match_bias){
                  MatchBiasIOU(pp_anchor, pp_label, &iou);
              }else{
                  MatchIOU(pp_anchor, p_wh_target + i * 2, pp_label, &iou);
              }
              if(iou > max_overlap){
                max_overlap = iou;
	        best_anchor = i;
              }
	  }
        }
	if(best_anchor != -1){
	  avg_iou += max_overlap;
	  avg_obj += p_obj_target[best_anchor];
          int tlabel = pp_label[0];
	  avg_cat += p_cls_target[best_anchor * num_classes + tlabel];
	  if(max_overlap > .5) recall += 1;
	  count++;

	  p_nobj_target[best_anchor] = p_obj_target[best_anchor];
	  p_obj_target[best_anchor] = max_overlap;
	  int offset = best_anchor * 2;
          AssignLocTargets(p_anchor + best_anchor * num_anchor_channel,
            pp_label, p_dxy_target + offset, p_wh_target + offset);
	  for(int c = 0;c < num_classes;c++)
            p_cls_target[best_anchor * num_classes + c] = 0;
          p_cls_target[best_anchor * num_classes + tlabel] = 1;
	}
      }
    }
  }  // end iterate batches
  printf("Avg IOU:%.4f, Class:%.4f, Obj:%.4f, NObj:%.4f, Avg Recall:%.4f, count:%.4d\n", avg_iou/count, avg_cat/count, avg_obj/count,avg_anyobj/(num_batches * num_anchors), recall/count, count);
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(MultiRegionTargetParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiRegionTargetOp<cpu, DType, mshadow::op::identity>(param);
  });
  return op;
}

Operator* MultiRegionTargetProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                       std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(MultiRegionTargetParam);
MXNET_REGISTER_OP_PROPERTY(_contrib_MultiRegionTarget, MultiRegionTargetProp)
.describe("Generate multiregion target labels")
.add_argument("anchor", "NDArray-or-Symbol", "Generated anchor boxes.")
.add_argument("label", "NDArray-or-Symbol", "Ground truth labels.")
.add_argument("obj_pred", "NDArray-or-Symbol", "Object predictions.")
.add_argument("dxy_pred", "NDArray-or-Symbol", "Dxy predictions.")
.add_argument("wh_pred", "NDArray-or-Symbol", "Weight/Height predictions.")
.add_argument("cls_pred", "NDArray-or-Symbol", "Class predictions.")
.add_arguments(MultiRegionTargetParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
