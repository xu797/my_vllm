#ifndef MYVLLM_OPERATORS_RELU_H_
#define MYVLLM_OPERATORS_RELU_H_

#include "layer/include/non_param_layer.h"
#include "layer/include/layer_factory.h"

namespace my_vllm
{

class ReluLayer : public NonParamLayer
{
public:
    ReluLayer() : NonParamLayer("Relu") {}
    InferStatus Forward(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
        std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

    static ParseParameterAttrStatus GetInstance(
        const std::shared_ptr<RuntimeOperator>& op,
        std::shared_ptr<Layer>& relu_layer);
};

}

#endif