/*!
 * Copyright (c) 2017 by Contributors
 * \file multiregion_regression.cc
 * \brief multiregion regression output
 * \author Yidong Ma
*/
#include "./multiregion_regression-inl.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateMultiRegionRegressionOp<cpu>(MultiRegionRegressionParam param) {
  return new MultiRegionRegressionOp<cpu, mshadow::op::minus>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *MultiRegionRegressionProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateMultiRegionRegressionOp, param_);
}

DMLC_REGISTER_PARAMETER(MultiRegionRegressionParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_MultiRegionRegression, MultiRegionRegressionProp)
.describe("Computes for multiregion loss.")
.add_argument("data", "NDArray-or-Symbol", "Input data to the function.")
.add_argument("label", "NDArray-or-Symbol", "Input label to the function.")
.add_argument("coord_grad_scale", "NDArray-or-Symbol", "GradScale to the function.")
.add_arguments(MultiRegionRegressionParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
