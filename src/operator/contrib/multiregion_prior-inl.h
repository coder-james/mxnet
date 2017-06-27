/*!
 * Copyright (c) 2017 by Contributors
 * \file multiregion_prior-inl.h
 * \brief generate multiregion prior boxes
 * \author Yidong Ma
*/
#ifndef MXNET_OPERATOR_CONTRIB_MULTIREGION_PRIOR_INL_H_
#define MXNET_OPERATOR_CONTRIB_MULTIREGION_PRIOR_INL_H_
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

namespace mregprior_enum {
enum MultiRegionPriorOpInputs {kData};
enum MultiRegionPriorOpOutputs {kOut};
}  // namespace mregprior_enum

struct MultiRegionPriorParam : public dmlc::Parameter<MultiRegionPriorParam> {
  nnvm::Tuple<float> anchors;
  DMLC_DECLARE_PARAMETER(MultiRegionPriorParam) {
    DMLC_DECLARE_FIELD(anchors).set_default({1.0f})
    .describe("List of anchors of generated MultiRegionPriores.");
  }
};  // struct MultiRegionPriorParam

template<typename xpu, typename DType>
class MultiRegionPriorOp : public Operator {
 public:
  explicit MultiRegionPriorOp(MultiRegionPriorParam param)
    : anchors_(param.anchors.begin(), param.anchors.end()) {
      CHECK_GT(anchors_.size(), 0);
    }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(static_cast<int>(in_data.size()), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> out;
    const int num_ac = static_cast<int>(anchors_.size());
    const int num_anchors = num_ac / 2;  // anchors per location
    int in_height = in_data[mregprior_enum::kData].size(2);
    int in_width = in_data[mregprior_enum::kData].size(3);
    Shape<2> oshape = Shape2(num_anchors * in_width * in_height, 6);
    out = out_data[mregprior_enum::kOut].get_with_shape<xpu, 2, DType>(oshape, s);
    MultiRegionPriorForward(out, anchors_, in_width, in_height, 6);
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
    Tensor<xpu, 2, DType> grad = in_grad[mregprior_enum::kData].FlatTo2D<xpu, DType>(s);
    grad = 0.f;
  }

 private:
  std::vector<float> anchors_;
};  // class MultiRegionPriorOp

template<typename xpu>
Operator *CreateOp(MultiRegionPriorParam, int dtype);

#if DMLC_USE_CXX11
class MultiRegionPriorProp: public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Inputs: [data]" << in_shape->size();
    TShape dshape = in_shape->at(mregprior_enum::kData);
    CHECK_GE(dshape.ndim(), 4) << "Input data should be 4D: batch-channel-height-width";
    int in_height = dshape[2];
    CHECK_GT(in_height, 0) << "Input height should > 0";
    int in_width = dshape[3];
    CHECK_GT(in_width, 0) << "Input width should > 0";
    TShape oshape = TShape(3);
    int num_anchors = param_.anchors.ndim() / 2;
    oshape[0] = 1;
    oshape[1] = in_height * in_width * num_anchors;
    oshape[2] = 6;
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MultiRegionPriorProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_MultiRegionPrior";
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  MultiRegionPriorParam param_;
};  // class MultiRegionPriorProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_MULTIREGION_PRIOR_INL_H_
