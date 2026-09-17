#ifndef MYVLLM_OPERATORS_SOFTMAX_H_
#define MYVLLM_OPERATORS_SOFTMAX_H_

#include "layer/include/non_param_layer.h"

namespace my_vllm 
{
class SoftmaxLayer : public NonParamLayer 
{
public:
    explicit SoftmaxLayer(int dim = -1);

    InferStatus Forward(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
        std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

    static ParseParameterAttrStatus CreateInstance(
        const std::shared_ptr<RuntimeOperator>& op,
        std::shared_ptr<Layer>& softmax_layer);

private:
    int softmax_dim_ = -1;
};
}  // namespace my_vllm

#endif  // MYVLLM_OPERATORS_SOFTMAX_H_
