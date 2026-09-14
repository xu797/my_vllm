#ifndef MYVLLM_OPERATORS_SIGMOID_H_
#define MYVLLM_OPERATORS_SIGMOID_H_

#include "layer/include/non_param_layer.h"
#include "layer/include/layer_factory.h"

namespace my_vllm
{

class SigmoidLayer : public NonParamLayer
{
public:
    SigmoidLayer() : NonParamLayer("Sigmoid") {}
    InferStatus Forward(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
        std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

    static ParseParameterAttrStatus GetInstance(
        const std::shared_ptr<RuntimeOperator>& op,
        std::shared_ptr<Layer>& sigmoid_layer);
};

}

#endif