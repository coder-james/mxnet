/*!
 * Copyright (c) 2017 by Contributors
 * \file multiregion_regression-inl.h
 * \brief MultiRegion Regression Output.
 * \author Yidong Ma
 */
#ifndef MXNET_OPERATOR_MULTIREGION_REGRESSION_INL_H_
#define MXNET_OPERATOR_MULTIREGION_REGRESSION_INL_H_

#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../operator_common.h"

namespace mxnet {
namespace op {

namespace reg_enum {
enum MultiRegionRegressionOpInputs {kData, kLabel, kScale};
enum MultiRegionRegressionOutputs {kOut};
}  // reg_enum

struct MultiRegionRegressionParam : public dmlc::Parameter<MultiRegionRegressionParam> {
  float grad_scale;
  bool scalable;
  DMLC_DECLARE_PARAMETER(MultiRegionRegressionParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
    DMLC_DECLARE_FIELD(scalable).set_default(false)
    .describe("Scalable loss gradient.");
  };
};

template<typename xpu, typename FOp>
class MultiRegionRegressionOp : public Operator {
 public:
  explicit MultiRegionRegressionOp(MultiRegionRegressionParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3U) << "MultiRegionRegressionOp Input: [data, label, coord_grad_scale]";
    CHECK_EQ(out_data.size(), 1U) << "MultiRegionRegressionOp Output: [output]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> out = out_data[reg_enum::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> data = in_data[reg_enum::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> label = in_data[reg_enum::kLabel].FlatTo2D<xpu, real_t>(s);
    Assign(out, req[reg_enum::kOut], F<FOp>(data, label));
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
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_GE(in_grad.size(), 1U);
    CHECK_GE(req.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> out = out_data[reg_enum::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> grad = in_grad[reg_enum::kData].FlatTo2D<xpu, real_t>(s);
    if(param_.scalable){
        Tensor<xpu, 2> scale = in_data[reg_enum::kScale].FlatTo2D<xpu, real_t>(s);
        Assign(grad, req[reg_enum::kData], F<mshadow::op::mul>(scale, out));
    }else{
        Assign(grad, req[reg_enum::kData], param_.grad_scale * out);
    }
  }

 private:
  MultiRegionRegressionParam param_;
};

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateMultiRegionRegressionOp(MultiRegionRegressionParam param);

#if DMLC_USE_CXX11
class MultiRegionRegressionProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label", "coord_grad_scale"};
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
    CHECK_EQ(in_shape->size(), 3) << "Input:[data, label, coord_grad_scale]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    auto &lshape = (*in_shape)[1];
    if (lshape[0] != dshape[0] || lshape.Size() != dshape.Size()) {
      std::ostringstream os;
      os << "Shape inconsistent, Provided=" << lshape << ','
         << " inferred shape=" << dshape;
      throw ::mxnet::op::InferShapeError(os.str(), 1);
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MultiRegionRegressionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_MultiRegionRegression";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[reg_enum::kScale], out_data[reg_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[reg_enum::kOut], in_grad[reg_enum::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[reg_enum::kData], out_data[reg_enum::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 protected:
  MultiRegionRegressionParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MULTIREGION_REGRESSION_INL_H_
