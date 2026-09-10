#ifndef MYVLLM_RUNTIME_OPERATOR_H_
#define MYVLLM_RUNTIME_OPERATOR_H_

#include "runtime_parameter.h"
#include "runtime_attribute.h"

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

}

#endif