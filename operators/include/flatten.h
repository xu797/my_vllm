#ifndef MYVLLM_OPERATORS_FLATTEN_H_
#define MYVLLM_OPERATORS_FLATTEN_H_

#include "layer/include/non_param_layer.h"
#include "layer/include/layer_factory.h"

namespace my_vllm
{

class FlattenLayer : public NonParamLayer 
{
public:
    explicit FlattenLayer(int start_dim, int end_dim);

    InferStatus Forward(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
        std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

    static ParseParameterAttrStatus CreateInstance(
        const std::shared_ptr<RuntimeOperator>& op,
        std::shared_ptr<Layer>& flatten_layer);

private:
    int start_dim_ = 0;
    int end_dim_ = 0;
};

}  // namespace my_vllm
#endif  // MYVLLM_OPERATORS_FLATTEN_H_
