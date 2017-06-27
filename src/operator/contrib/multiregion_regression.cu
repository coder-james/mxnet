/*!
 * Copyright (c) 2017 by Contributors
 * \file multiregion_regression.cu
 * \brief multiregion regression output
 * \author Yidong Ma
*/
#include "./multiregion_regression-inl.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateMultiRegionRegressionOp<gpu>(MultiRegionRegressionParam param) {
  return new MultiRegionRegressionOp<gpu, mshadow::op::minus>(param);
}
}  // namespace op
}  // namespace mxnet

