#include "runtime_operator.h"
#include "tensor/include/tensor_util.h"

namespace my_vllm {
RuntimeOperator::~RuntimeOperator() {
  for (auto& [_, param] : this->params) {
    if (param != nullptr) {
      delete param;
      param = nullptr;
    }
  }
}

}  // namespace my_vllm
