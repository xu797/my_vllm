#include "maxpooling.h"

namespace my_vllm
{

MaxPoolingLayer::MaxPoolingLayer(
    uint32_t padding_h, uint32_t padding_w,
    uint32_t pooling_size_h, uint32_t pooling_size_w, 
    uint32_t stride_h, uint32_t stride_w)
    : NonParamLayer("MaxPooling"),
      padding_h_(padding_h), padding_w_(padding_w),
      pooling_size_h_(pooling_size_h), pooling_size_w_(pooling_size_w),
      stride_h_(stride_h), stride_w_(stride_w) 
{

}

InferStatus MaxPoolingLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
    std::vector<std::shared_ptr<Tensor<float>>> &outputs) 
{
    if (inputs.empty()) 
    {
        LOG(ERROR) << "The input tensor array in the maxpooling layer is empty";
        return InferStatus::kInferFailedInputEmpty;
    }
    if (inputs.size() != outputs.size()) 
    {
        LOG(ERROR) << "The input and output tensor array size of the maxpooling layer do not match";
        return InferStatus::kInferFailedInputOutSizeMatchError;
    }

    const uint32_t batch = inputs.size();
    const uint32_t pooling_h = pooling_size_h_;
    const uint32_t pooling_w = pooling_size_w_;
    if (!stride_h_ || !stride_w_)
    {
        LOG(ERROR) << "The stride parameter is set incorrectly. It must always be greater than 0";
        return InferStatus::kInferFailedStrideParameterError;
    }

    for (uint32_t i = 0; i < batch; ++i) 
    {
        const std::shared_ptr<ftensor>& input_data = inputs.at(i);
        if (input_data == nullptr || input_data->empty())
        {
            LOG(ERROR) << "The input tensor array in the max pooling layer has an empty tensor " << i << "th";
            return InferStatus::kInferFailedInputEmpty;
        } 
        else 
        {
            uint32_t input_h = input_data->rows();
            uint32_t input_w = input_data->cols();
            uint32_t output_h = uint32_t(std::floor((int(input_h) - int(pooling_h) + 2 * padding_h_) / stride_h_ + 1));
            uint32_t output_w = uint32_t(std::floor((int(input_w) - int(pooling_w) + 2 * padding_w_) / stride_w_ + 1));
            if (!output_w || !output_h) 
            {
                LOG(ERROR) << "The output size of tensor " << i << "th" << " in the max pooling layer is less than zero";
                return InferStatus::kInferFailedOutputSizeError;
            } 
            else 
            {
                const std::shared_ptr<ftensor>& output_data = outputs.at(i);
                if (output_data != nullptr && !output_data->empty())
                {
                    if (output_data->rows() != output_h || output_data->cols() != output_w) 
                    {
                        LOG(ERROR) << "The output tensor array in the max pooling layer has an incorrectly sized tensor " << i << "th";
                        return InferStatus::kInferFailedOutputSizeError;
                    }
                }
            }
        }
    }

    for (uint32_t i = 0; i < batch; ++i) 
    {
        const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);
        CHECK(input_data == nullptr || !input_data->empty())
            << "The input tensor array in the max pooling layer has an empty tensor"
            << i << "th";

        const uint32_t input_h = input_data->rows();
        const uint32_t input_w = input_data->cols();
        const uint32_t input_padded_h = input_data->rows() + 2 * padding_h_;
        const uint32_t input_padded_w = input_data->cols() + 2 * padding_w_;

        const uint32_t input_c = input_data->channels();

        const uint32_t output_h = uint32_t(std::floor((int(input_padded_h) - int(pooling_h)) / stride_h_ + 1));
        const uint32_t output_w = uint32_t(std::floor((int(input_padded_w) - int(pooling_w)) / stride_w_ + 1));

        std::shared_ptr<Tensor<float>> output_data = outputs.at(i);
        if (output_data == nullptr || output_data->empty())
        {
            output_data = std::make_shared<Tensor<float>>(input_c, output_h, output_w);
            outputs.at(i) = output_data;
        }

        CHECK(output_data->rows() == output_h && output_data->cols() == output_w && output_data->channels() == input_c)
            << "The output tensor array in the max pooling layer has an incorrectly sized tensor "
            << i << "th";

        for (uint32_t ic = 0; ic < input_c; ++ic) 
        {
            const arma::fmat& input_channel = input_data->slice(ic);
            arma::fmat& output_channel = output_data->slice(ic);
            for (uint32_t c = 0; c < input_padded_w - pooling_w + 1; c += stride_w_)
            {
                int output_col = int(c / stride_w_);
                for (uint32_t r = 0; r < input_padded_h - pooling_h + 1; r += stride_h_) 
                {
                    int output_row = int(r / stride_h_);
                    float* output_channel_ptr = output_channel.colptr(output_col);
                    float max_value = std::numeric_limits<float>::lowest();
                    for (uint32_t w = 0; w < pooling_w; ++w) 
                    {
                        uint32_t global_c = c + w;  // 填充图全局列坐标
                        for (uint32_t h = 0; h < pooling_h; ++h) 
                        {
                            uint32_t global_r = r + h; // 填充图全局行坐标
                            float current_value = std::numeric_limits<float>::lowest();

                            // 判断是否落在原图有效区域
                            if (global_r >= padding_h_ && global_r < padding_h_ + input_h &&
                                global_c >= padding_w_ && global_c < padding_w_ + input_w)
                            {
                                uint32_t mat_c = global_c - padding_w_;
                                uint32_t mat_r = global_r - padding_h_;
                                const float* col_ptr = input_channel.colptr(mat_c);
                                current_value = *(col_ptr + mat_r);
                            }
                            else
                            {
                                current_value = std::numeric_limits<float>::lowest();
                            }

                            max_value = std::max(max_value, current_value);
                        }
                    }
                }
            }
        }
    }
    return InferStatus::kInferSuccess;
}


ParseParameterAttrStatus MaxPoolingLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator> &op,
    std::shared_ptr<Layer> &maxpooling_layer) 
{
    CHECK(op != nullptr) << "MaxPooling operator is nullptr";
    maxpooling_layer = std::make_shared<MaxPoolingLayer>();
    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

// 使用工具类注册算子 这里相当于定义一个全局变量
LayerRegistererWrapper kMaxPoolingGetInstance("nn.MaxPool2d", MaxPoolingLayer::GetInstance);

}