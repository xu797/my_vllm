#ifndef MYVLLM_RUNTIME_OPERAND_H_
#define MYVLLM_RUNTIME_OPERAND_H_
#include <string>
#include <vector>
#include <memory>

#include "tensor/include/tensor.h"
#include "runtime_datatype.h"


namespace my_vllm
{

struct RuntimeOperand
{
    std::string name;
    std::vector<int> shapes;
    std::vector<std::shared_ptr<Tensor<float>>> datas;
    RuntimeDataType type = RuntimeDataType::kTypeUnknown;
};

}// end of namespace my-vllm

#endif