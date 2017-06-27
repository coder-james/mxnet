/*!
 * Copyright (c) 2017 by Contributors
 * \file multiregion_detection-inl.h
 * \brief multiregion detection
 * \author Yidong Ma
*/
#ifndef MXNET_OPERATOR_CONTRIB_MULTIREGION_DETECTION_INL_H_
#define MXNET_OPERATOR_CONTRIB_MULTIREGION_DETECTION_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/base.h>
#include <nnvm/tuple.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <valarray>
#include "../operator_common.h"

namespace mxnet {
namespace op {
namespace mboxdet_enum {
enum MultiRegionDetectionOpInputs {kObjPred, kDxyPred, kWHPred, kClsPred, kAnchor};
enum MultiRegionDetectionOpOutputs {kOut};
enum MultiRegionDetectionOpResource {kTempSpace};
}  // namespace mboxdet_enum

struct MultiRegionDetectionParam : public dmlc::Parameter<MultiRegionDetectionParam> {
  float threshold;
  float nms_threshold;
  int num_classes;
  int nms_topk;
  bool force_suppress;
  DMLC_DECLARE_PARAMETER(MultiRegionDetectionParam) {
    DMLC_DECLARE_FIELD(threshold).set_default(0.1f)
    .describe("Threshold to predict.");
    DMLC_DECLARE_FIELD(nms_threshold).set_default(0.5f)
    .describe("Non-maximum suppression threshold.");
    DMLC_DECLARE_FIELD(num_classes).set_default(1)
    .describe("Number of Classes.");
    DMLC_DECLARE_FIELD(force_suppress).set_default(true)
    .describe("Suppress all detections regardless of class_id.");
    DMLC_DECLARE_FIELD(nms_topk).set_default(-1)
    .describe("Keep maximum top k detections before nms, -1 for no limit.");
  }
};  // struct MultiRegionDetectionParam

template<typename xpu, typename DType>
class MultiRegionDetectionOp : public Operator {
 public:
  explicit MultiRegionDetectionOp(MultiRegionDetectionParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
     using namespace mshadow;
     using namespace mshadow::expr;
     CHECK_EQ(in_data.size(), 5U) << "Input: [obj_pred, dxy_pred, wh_pred, cls_pred, anchor]";
     TShape ashape = in_data[mboxdet_enum::kAnchor].shape_;
     CHECK_EQ(out_data.size(), 1U);

     Stream<xpu> *s = ctx.get_stream<xpu>();
     Tensor<xpu, 2, DType> obj_pred = in_data[mboxdet_enum::kObjPred]
       .get<xpu, 2, DType>(s);
     Tensor<xpu, 2, DType> dxy_pred = in_data[mboxdet_enum::kDxyPred]
       .get<xpu, 2, DType>(s);
     Tensor<xpu, 2, DType> wh_pred = in_data[mboxdet_enum::kWHPred]
       .get<xpu, 2, DType>(s);
     Tensor<xpu, 2, DType> cls_pred = in_data[mboxdet_enum::kClsPred]
       .get<xpu, 2, DType>(s);
     Tensor<xpu, 2, DType> anchor = in_data[mboxdet_enum::kAnchor]
       .get_with_shape<xpu, 2, DType>(Shape2(ashape[1], ashape[2]), s);
     Tensor<xpu, 3, DType> out = out_data[mboxdet_enum::kOut]
       .get<xpu, 3, DType>(s);
     Tensor<xpu, 3, DType> temp_space = ctx.requested[mboxdet_enum::kTempSpace]
       .get_space_typed<xpu, 3, DType>(out.shape_, s);
     out = -1.f;
     MultiRegionDetectionForward(out, obj_pred, dxy_pred, wh_pred, cls_pred, anchor, 
       temp_space, param_.threshold, param_.nms_threshold, param_.num_classes, 
       param_.force_suppress, param_.nms_topk);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> grado = in_grad[mboxdet_enum::kObjPred].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> gradd = in_grad[mboxdet_enum::kDxyPred].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> gradw = in_grad[mboxdet_enum::kWHPred].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> gradc = in_grad[mboxdet_enum::kClsPred].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grada = in_grad[mboxdet_enum::kAnchor].FlatTo2D<xpu, DType>(s);
    grado = 0.f;
    gradd = 0.f;
    gradw = 0.f;
    gradc = 0.f;
    grada = 0.f;
}

 private:
  MultiRegionDetectionParam param_;
};  // class MultiRegionDetectionOp

template<typename xpu>
Operator *CreateOp(MultiRegionDetectionParam, int dtype);

#if DMLC_USE_CXX11
class MultiRegionDetectionProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"obj_pred", "dxy_pred", "wh_pred", "cls_pred", "anchor"};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 5U) << "Inputs: [obj_pred, dxy_pred, wh_pred, cls_pred, anchor]";
    TShape oshape = in_shape->at(mboxdet_enum::kObjPred);
    TShape dshape = in_shape->at(mboxdet_enum::kDxyPred);
    TShape wshape = in_shape->at(mboxdet_enum::kWHPred);
    TShape cshape = in_shape->at(mboxdet_enum::kClsPred);
    TShape ashape = in_shape->at(mboxdet_enum::kAnchor);
    CHECK_EQ(oshape.ndim(), 2U) << "Provided: " << oshape;
    CHECK_EQ(dshape.ndim(), 2U) << "Provided: " << dshape;
    CHECK_EQ(wshape.ndim(), 2U) << "Provided: " << wshape;
    CHECK_EQ(cshape.ndim(), 2U) << "Provided: " << cshape;
    CHECK_EQ(ashape.ndim(), 3U) << "Provided: " << ashape;
    CHECK_GT(ashape[1], 0U) << "Number of anchors must > 0";
    CHECK_EQ(oshape[1], ashape[1]) << "Number of anchors mismatch # obj";
    CHECK_EQ(dshape[1], ashape[1] * 2) << "# anchors mismatch with # dxy";
    CHECK_EQ(wshape[1], ashape[1] * 2) << "# anchors mismatch with # wh";
    TShape outshape = TShape(3);
    outshape[0] = oshape[0]; //batch
    outshape[1] = ashape[1]; //num_boxes
    outshape[2] = 6;  // [class_id, prob, xmin, ymin, xmax, ymax]
    out_shape->clear();
    out_shape->push_back(outshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MultiRegionDetectionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_MultiRegionDetection";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  MultiRegionDetectionParam param_;
};  // class MultiRegionDetectionProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_MULTIREGION_DETECTION_INL_H_
