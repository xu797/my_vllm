#ifndef MYVLLM_RUNTIME_IR_H_
#define MYVLLM_RUNTIME_IR_H_

#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>
#include <glog/logging.h>

#include "ir.h"
#include "runtime_operand.h"
#include "runtime_operator.h"
#include "layer/include/layer.h"
#include "layer/include/layer_factory.h"

namespace my_vllm 
{
// 计算图结构，由多个计算节点和节点之间的数据流图组成
class RuntimeGraph 
{

// class Layer;

public:
    RuntimeGraph(std::string param_path, std::string bin_path);
      
    void set_bin_path(const std::string &bin_path);
    void set_param_path(const std::string &param_path);

    const std::string &param_path() const;
    const std::string &bin_path() const;

    bool Init();
    const std::vector<std::shared_ptr<RuntimeOperator>> &operators() const;

    void Build(const std::string &input_name, const std::string &output_name);

    const std::vector<std::shared_ptr<RuntimeOperator>> &get_topo_queues() const;

    static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<RuntimeOperator> &op);

    std::vector<std::shared_ptr<Tensor<float>>> Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, bool debug);

private:
    static void InitGraphOperatorsInput(
        const std::vector<pnnx::Operand *> &inputs, 
        const std::shared_ptr<RuntimeOperator> &runtime_operator);


    static void InitGraphOperatorsOutput(
        const std::vector<pnnx::Operand *> &outputs,
        const std::shared_ptr<RuntimeOperator> &runtime_operator);


    static void InitGraphAttrs(
        const std::map<std::string, pnnx::Attribute> &attrs,
        const std::shared_ptr<RuntimeOperator> &runtime_operator);


    static void InitGraphParams(
        const std::map<std::string, pnnx::Parameter> &params,
        const std::shared_ptr<RuntimeOperator> &runtime_operator);

    void ReverseTopo(const std::shared_ptr<RuntimeOperator> &root_op);
    
    static void ProbeNextLayer(const std::shared_ptr<RuntimeOperator> &current_op, const std::vector<std::shared_ptr<Tensor<float>>> &layer_output_data);

private:
    enum class GraphState
    {
        NeedInit = -2,
        NeedBuild = -1,
        Complete = 0,
    };

public:
    GraphState graph_state() const;
    
private:
    GraphState graph_state_ = GraphState::NeedInit;
    std::string input_name_;  // 计算图输入节点的名称
    std::string output_name_; // 计算图输出节点的名称
    std::string param_path_;  // 计算图的结构文件
    std::string bin_path_;    // 计算图的权重文件

    std::vector<std::shared_ptr<RuntimeOperator>> operators_;
    std::map<std::string, std::shared_ptr<RuntimeOperator>> operators_maps_;
    std::vector<std::shared_ptr<RuntimeOperator>> topo_operators_;

    std::unique_ptr<pnnx::Graph> graph_; // pnnx的graph
};

} // namespace my_vllm
#endif