#ifndef MYVLLM_EXPRESSION_H_
#define MYVLLM_EXPRESSION_H_

#include "parse_expression.h"
#include "layer/include/non_param_layer.h"

namespace my_vllm
{
class ExpressionLayer : public NonParamLayer 
{
public:
    explicit ExpressionLayer( std::string statement);

    InferStatus Forward(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
        std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

    static ParseParameterAttrStatus GetInstance(
        const std::shared_ptr<RuntimeOperator>& op,
        std::shared_ptr<Layer>& expression_layer);

private:
    std::string statement_;
    std::unique_ptr<ExpressionParser> parser_;
};
}  // namespace my_vllm
#endif  // MYVLLM_EXPRESSION_H_

