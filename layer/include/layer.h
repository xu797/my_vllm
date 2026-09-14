#ifndef MYVLLM_LAYER_H_
#define MYVLLM_LAYER_H_

#include <glog/logging.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensor/include/tensor.h"
#include "graph/include/runtime_operator.h"
#include "graph/include/status_code.h"

namespace my_vllm
{

class Layer 
{

public:
    explicit Layer(std::string layer_name) : layer_name_(std::move(layer_name)) {}

    virtual ~Layer() = default;

    virtual InferStatus Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs);

    virtual InferStatus Forward();
    virtual const std::vector<std::shared_ptr<Tensor<float>>>& weights() const;
    virtual const std::vector<std::shared_ptr<Tensor<float>>>& bias() const;
    virtual void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights);
    virtual void set_bias(const std::vector<std::shared_ptr<Tensor<float>>>& bias);
    virtual void set_weights(const std::vector<float>& weights);
    virtual void set_bias(const std::vector<float>& bias);
    virtual const std::string& layer_name() const { return this->layer_name_; }

    void set_runtime_operator(const std::shared_ptr<RuntimeOperator>& runtime_operator);

protected:
    std::weak_ptr<RuntimeOperator> runtime_operator_;
    std::string layer_name_;  
};

}

#endif