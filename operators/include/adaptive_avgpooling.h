#ifndef MYVLLM_OPERATORS_AVGPOOLING_H_
#define MYVLLM_OPERATORS_AVGPOOLING_H_

#include "layer/include/non_param_layer.h"

namespace my_vllm
{
class AdaptiveAveragePoolingLayer : public NonParamLayer 
{
public:
    explicit AdaptiveAveragePoolingLayer(uint32_t output_h, uint32_t output_w);

    InferStatus Forward(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
        std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

    static ParseParameterAttrStatus CreateInstance(
        const std::shared_ptr<RuntimeOperator>& op,
        std::shared_ptr<Layer>& avg_layer);

private:
    uint32_t output_h_ = 0;
    uint32_t output_w_ = 0;
};

}  // namespace my_vllm
#endif  // MYVLLM_OPERATORS_AVGPOOLING_H_
