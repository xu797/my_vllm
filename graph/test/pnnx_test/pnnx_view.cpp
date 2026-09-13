#include <iostream>
#include <vector>
#include <string>
#include "ir.h"

void print_operand(const pnnx::Operand* operand)
{
    if (!operand) return;
    std::cout << "  Operand: name=" << operand->name
              << ", dtype=" << operand->type
              << ", shape=[";
    for (size_t i=0; i<operand->shape.size(); i++)
    {
        if(i>0) std::cout << ",";
        std::cout << operand->shape[i];
    }
    std::cout << "]\n";
}

void print_parameter(const pnnx::Parameter& param)
{
    switch(param.type)
    {
        case 1: // bool
            std::cout << param.b;
            break;
        case 2: // int scalar
            std::cout << param.i;
            break;
        case 3: // float scalar
            std::cout << param.f;
            break;
        case 4: // string
            std::cout << "\"" << param.s << "\"";
            break;
        case 5: // int array ai
            for(size_t i=0; i<param.ai.size(); i++)
            {
                if(i>0) std::cout << ",";
                std::cout << param.ai[i];
            }
            break;
        case 6: // float array af
            for(size_t i=0; i<param.af.size(); i++)
            {
                if(i>0) std::cout << ",";
                std::cout << param.af[i];
            }
            break;
        case 7: // string array as
            for(size_t i=0; i<param.as.size(); i++)
            {
                if(i>0) std::cout << ",";
                std::cout << "\"" << param.as[i] << "\"";
            }
            break;
        default:
            std::cout << "unknown(type=" << param.type << ")";
    }
}

int main()
{
    std::string param_path = "demo_net.pnnx.param";
    std::string bin_path = "demo_net.pnnx.bin";
    pnnx::Graph g;
    int ret = g.load(param_path, bin_path);
    if(ret != 0)
    {
        std::cerr << "load pnnx failed, ret=" << ret << std::endl;
        return -1;
    }
    std::cout << "===== Total Operators: " << g.ops.size() << "\n";
    for(const pnnx::Operator* op : g.ops)
    {
        std::cout << "\n--- Operator --- \n";
        std::cout << "name: " << op->name << "\n";
        std::cout << "type: " << op->type << "\n";
        std::cout << "inputs:\n";
        for(auto in : op->inputs) print_operand(in);
        std::cout << "outputs:\n";
        for(auto out : op->outputs) print_operand(out);
        std::cout << "Attributes (weights):\n";
        for(auto& [attr_name, attr] : op->attrs)
        {
            std::cout << "  attr " << attr_name << ", shape=[";
            for(size_t i=0; i<attr.shape.size();i++)
            {
                if(i>0) std::cout << ",";
                std::cout << attr.shape[i];
            }
            std::cout << "], bytes=" << attr.data.size() << "\n";
        }
        std::cout << "Parameters:\n";
        for(auto& [param_name, param] : op->params)
        {
            std::cout << "  param " << param_name << " = [";
            print_parameter(param);
            std::cout << "]\n";
        }
    }
    return 0;
}
