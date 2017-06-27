/*!
 * Copyright (c) 2017 by Contributors
 * \file multiregion_target-inl.h
 * \brief generate multiregion target label
 * \author Yidong Ma
*/
#ifndef MXNET_OPERATOR_CONTRIB_MULTIREGION_TARGET_INL_H_
#define MXNET_OPERATOR_CONTRIB_MULTIREGION_TARGET_INL_H_
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
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

namespace mregtarget_enum {
enum MultiRegionTargetOpInputs {kAnchor, kLabel, kObjPred, kDxyPred, kWHPred, kClsPred};
enum MultiRegionTargetOpOutputs {kObj, kNObj, kDxy, kWH, kCls};
enum MultiRegionTargetOpResource {kTempSpace};
}  // namespace mregtarget_enum

struct MultiRegionTargetParam : public dmlc::Parameter<MultiRegionTargetParam> {
  float threshold;
  int num_classes;
  DMLC_DECLARE_PARAMETER(MultiRegionTargetParam) {
    DMLC_DECLARE_FIELD(threshold).set_default(0.5f)
    .describe("Threshold to be regarded as containing object.");
    DMLC_DECLARE_FIELD(num_classes).set_default(-1)
    .describe("Number of classes.");
  }
};  // struct MultiRegionTargetParam

template<typename xpu, typename DType, typename TargetOp>
class MultiRegionTargetOp : public Operator {
 public:
  explicit MultiRegionTargetOp(MultiRegionTargetParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow_op;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 6);
    CHECK_EQ(out_data.size(), 5);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> anchors = in_data[mregtarget_enum::kAnchor]
      .get_with_shape<xpu, 2, DType>(
      Shape2(in_data[mregtarget_enum::kAnchor].size(1), in_data[mregtarget_enum::kAnchor].size(2)), s);
    Tensor<xpu, 3, DType> labels = in_data[mregtarget_enum::kLabel]
      .get<xpu, 3, DType>(s);
    Tensor<xpu, 2, DType> obj_preds = in_data[mregtarget_enum::kObjPred]
      .get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> dxy_preds = in_data[mregtarget_enum::kDxyPred]
      .get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> wh_preds = in_data[mregtarget_enum::kWHPred]
      .get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> cls_preds = in_data[mregtarget_enum::kClsPred]
      .get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> obj_target = out_data[mregtarget_enum::kObj]
      .get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> nobj_target = out_data[mregtarget_enum::kNObj]
      .get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> dxy_target = out_data[mregtarget_enum::kDxy]
      .get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> wh_target = out_data[mregtarget_enum::kWH]
      .get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> cls_target = out_data[mregtarget_enum::kCls]
      .get<xpu, 2, DType>(s);
    Assign(obj_target, req[mregtarget_enum::kObj], F<TargetOp>(obj_preds))
    Assign(dxy_target, req[mregtarget_enum::kDxy], F<TargetOp>(dxy_preds))
    Assign(wh_target, req[mregtarget_enum::kWH], F<TargetOp>(wh_preds))
    Assign(cls_target, req[mregtarget_enum::kCls], F<TargetOp>(cls_preds))

    index_t num_batches = labels.size(0);
    index_t num_anchors = anchors.size(0);
    index_t num_labels = labels.size(1);

    Shape<4> temp_shape = Shape4(2, num_batches, num_anchors, num_labels);
    Tensor<xpu, 4, DType> temp_space = ctx.requested[mregtarget_enum::kTempSpace]
      .get_space_typed<xpu, 4, DType>(temp_shape, s);
    nobj_target = 0.0f;
    temp_space = 0.0f;
    CHECK_EQ(anchors.CheckContiguous(), true);
    CHECK_EQ(labels.CheckContiguous(), true);
    CHECK_EQ(obj_preds.CheckContiguous(), true);
    CHECK_EQ(obj_target.CheckContiguous(), true);
    CHECK_EQ(nobj_target.CheckContiguous(), true);
    CHECK_EQ(dxy_target.CheckContiguous(), true);
    CHECK_EQ(wh_target.CheckContiguous(), true);
    CHECK_EQ(cls_target.CheckContiguous(), true);
    CHECK_EQ(temp_space.CheckContiguous(), true);

    MultiRegionTargetForward(obj_target, nobj_target, dxy_target, wh_target, cls_target,
                          anchors, labels, temp_space,
                          param_.threshold, param_.num_classes);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 2, DType> ograd = in_grad[mregtarget_enum::kObjPred].FlatTo2D<xpu, DType>(s);
  Tensor<xpu, 2, DType> xygrad = in_grad[mregtarget_enum::kDxyPred].FlatTo2D<xpu, DType>(s);
  Tensor<xpu, 2, DType> whgrad = in_grad[mregtarget_enum::kWHPred].FlatTo2D<xpu, DType>(s);
  Tensor<xpu, 2, DType> cgrad = in_grad[mregtarget_enum::kClsPred].FlatTo2D<xpu, DType>(s);
  ograd = 0.f;
  xygrad = 0.f;
  whgrad = 0.f;
  cgrad = 0.f;
}

 private:
  MultiRegionTargetParam param_;
};  // class MultiRegionTargetOp

template<typename xpu>
Operator* CreateOp(MultiRegionTargetParam param, int dtype);

#if DMLC_USE_CXX11
class MultiRegionTargetProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"anchor", "label", "obj_pred", "dxy_pred", "wh_pred", "cls_pred"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"obj_target", "nobj_target", "dxy_target", "wh_target", "cls_target"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 6) << "Input: [anchor, label, objPred, dxyPred, whPred, clsPred]";
    TShape ashape = in_shape->at(mregtarget_enum::kAnchor);
    CHECK_EQ(ashape.ndim(), 3) << "Anchor: [1-num_box-channel(6)]";
    CHECK_EQ(ashape[0], 1) << "Anchors are shared across batches, first dim=1";
    CHECK_GT(ashape[1], 0) << "Number boxes should > 0";
    CHECK_EQ(ashape[2], 6) << "Box dimension should be 6: [left-top-sw-sh-w-h]";

    TShape lshape = in_shape->at(mregtarget_enum::kLabel);
    CHECK_EQ(lshape.ndim(), 3) << "Label should be [batch-num_labels-(>=5)] tensor";
    CHECK_GT(lshape[1], 0) << "Padded label should > 0";
    CHECK_GE(lshape[2], 5) << "Label width must >=5";

    TShape oshape = in_shape->at(mregtarget_enum::kObjPred);
    CHECK_EQ(oshape.ndim(), 2) << "Obj Predition: [batch-num_box]";
    CHECK_EQ(oshape[1], ashape[1]) << "Number of anchors mismatch";

    TShape xyshape = in_shape->at(mregtarget_enum::kDxyPred);
    CHECK_EQ(xyshape.ndim(), 2) << "Obj Predition: [batch-num_box*2]";
    CHECK_EQ(xyshape[1], ashape[1] * 2) << "Number of anchors mismatch";

    TShape whshape = in_shape->at(mregtarget_enum::kWHPred);
    CHECK_EQ(whshape.ndim(), 2) << "Obj Predition: [batch-num_box*2]";
    CHECK_EQ(whshape[1], ashape[1] * 2) << "Number of anchors mismatch";

    TShape pshape = in_shape->at(mregtarget_enum::kClsPred);
    CHECK_EQ(pshape.ndim(), 2) << "Cls Prediction: [batch-num_box*num_classes]";

    TShape dxy_shape = Shape2(lshape[0], ashape[1] * 2);  // batch - (num_box * 2)
    TShape obj_shape = Shape2(lshape[0], ashape[1]); // batch - num_box
    TShape nobj_shape = Shape2(lshape[0], ashape[1]); // batch - num_box
    TShape wh_shape = dxy_shape;
    TShape cls_shape = Shape2(lshape[0], pshape[1]);  // batch - num_box * num_classes
    out_shape->clear();
    out_shape->push_back(obj_shape);
    out_shape->push_back(nobj_shape);
    out_shape->push_back(dxy_shape);
    out_shape->push_back(wh_shape);
    out_shape->push_back(cls_shape);
    return true;
  }

  OperatorProperty* Copy() const override {
    MultiRegionTargetProp* MultiRegionTarget_sym = new MultiRegionTargetProp();
    MultiRegionTarget_sym->param_ = this->param_;
    return MultiRegionTarget_sym;
  }

  std::string TypeString() const override {
    return "_contrib_MultiRegionTarget";
  }

  //  decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  std::vector<std::pair<int, void*>> ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[mregtarget_enum::kObjPred], out_data[mregtarget_enum::kObj]}, 
            {in_data[mregtarget_enum::kDxyPred], out_data[mregtarget_enum::kDxy]},
            {in_data[mregtarget_enum::kWHPred], out_data[mregtarget_enum::kWH]},
            {in_data[mregtarget_enum::kClsPred], out_data[mregtarget_enum::kCls]}};
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
  MultiRegionTargetParam param_;
};  // class MultiRegionTargetProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_MULTIBOX_TARGET_INL_H_
