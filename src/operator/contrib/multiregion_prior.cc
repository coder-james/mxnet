/*!
 * Copyright (c) 2017 by Contributors
 * \file multiregion_prior.cc
 * \brief generate multiregion prior boxes cpu implementation
 * \author Yidong Ma
*/

#include "./multiregion_prior-inl.h"

namespace mshadow {
template<typename DType>
inline void MultiRegionPriorForward(const Tensor<cpu, 2, DType> &out,
                            const std::vector<float> &anchors,
                            const int in_width, const int in_height,
			    const int channels) {
  const int num_ac = static_cast<int>(anchors.size());
  int count = 0;
  for (int r = 0; r < in_height; ++r) {
    for (int c = 0; c < in_width; ++c) {
      for (int i = 0; i < num_ac/2; ++i) {
	int offset = 2 * i;
        float sw = anchors[offset];
        float sh = anchors[offset + 1];
        out[count][0] = c;  // left x
        out[count][1] = r;  // top y
        out[count][2] = sw;  // bias w
        out[count][3] = sh;  // bias h
	out[count][4] = in_width; //feature width
	out[count][5] = in_height; //feature height
        ++count;
      }
    }
  }
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(MultiRegionPriorParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiRegionPriorOp<cpu, DType>(param);
  });
  return op;
}

Operator* MultiRegionPriorProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                       std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(MultiRegionPriorParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_MultiRegionPrior, MultiRegionPriorProp)
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_arguments(MultiRegionPriorParam::__FIELDS__())
.describe("Generate anchor boxes from data, anchor_str.");

}  // namespace op
}  // namespace mxnet
