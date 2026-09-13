#ifndef MYVLLM_RUNTIME_OPERATOR_H_
#define MYVLLM_RUNTIME_OPERATOR_H_

#include "ir.h"
#include "runtime_parameter.h"
#include "runtime_attribute.h"
#include "runtime_operand.h"

namespace my_vllm
{

class Layer;

struct RuntimeOperator 
{
    virtual ~RuntimeOperator();
    bool has_forward = false;
    std::string name;                /// 节点名称，唯一标识算子
    std::string type;                /// 算子类型：Conv2d、BatchNorm、Relu
    std::shared_ptr<Layer> layer;    /// 绑定的计算实现层（真正做forward的逻辑）
    std::vector<std::string> output_names;
    std::shared_ptr<RuntimeOperand> output_operands;
    std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands;
    std::vector<std::shared_ptr<RuntimeOperand>> input_operands_seq;
    std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators;
    std::map<std::string, RuntimeParameter*> params;
    std::map<std::string, std::shared_ptr<RuntimeAttribute>> attribute;
};

class RuntimeOperatorUtils 
{

public:
    /**
     * 如果图是第一次运行，则根据节点输入operand的形状准备好后续Layer计算中所需要的Tensor
     * 如果图是第二次以上运行，则检查输入operand的形状和operand中张量的形状是否匹配
     * @param operators 计算图中的计算节点
     */
    static void InitOperatorInput(
        const std::vector<std::shared_ptr<RuntimeOperator>>& operators);

    /**
     * 如果图是第一次运行，则根据节点输出operand的形状准备好后续Layer计算中所需要的Tensor
     * 如果图是第二次以上运行，则检查输出operand的形状和operand中张量的形状是否匹配
     * @param pnnx_operators pnnx图节点
     * @param operators KuiperInfer计算图中的计算节点
     */
    static void InitOperatorOutput(
        const std::vector<pnnx::Operator*>& pnnx_operators,
        const std::vector<std::shared_ptr<RuntimeOperator>>& operators);
};

}

#endif