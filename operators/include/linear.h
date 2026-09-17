#ifndef MYVLLM_OPERATORS_LINEAR_H_
#define MYVLLM_OPERATORS_LINEAR_H_

#include "layer/include/param_layer.h"

namespace my_vllm
{
class LinearLayer : public ParamLayer 
{

public:
    explicit LinearLayer(int32_t in_features, int32_t out_features, bool use_bias);

    InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                        std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &linear_layer);
private:
    int32_t in_features_ = 0;
    int32_t out_features_ = 0;
    bool use_bias_ = false;
    
};
}

#endif