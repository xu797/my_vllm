#ifndef MYVLLM_LAYER_FACTORY_H_
#define MYVLLM_LAYER_FACTORY_H_

#include <map>
#include <memory>
#include <string>

#include "layer.h"
#include "runtime_operator.h"

namespace my_vllm
{

class LayerRegisterer 
{
public:
    //Creator 指向一个返回值是ParseParameterAttrStatus的函数，参数是后面括号里面的两个，也就是说Creator是函数指针
    typedef ParseParameterAttrStatus (*Creator)(
        const std::shared_ptr<RuntimeOperator> &op,
        std::shared_ptr<Layer> &layer);

    typedef std::map<std::string, Creator> CreateRegistry;

public:
    static void RegisterCreator(const std::string &layer_type, const Creator &creator);

    static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<RuntimeOperator> &op);

    static CreateRegistry &Registry();

    static std::vector<std::string> layer_types();
};

class LayerRegistererWrapper
{
public:
    LayerRegistererWrapper(const std::string &layer_type, const LayerRegisterer::Creator &creator)
    {
        LayerRegisterer::RegisterCreator(layer_type, creator);
    }
};

}  // namespace my_vllm

#endif  
