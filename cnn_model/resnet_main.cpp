#include <vector>
#include <opencv2/opencv.hpp>
#include "expression/include/expression.h"
#include "graph/include/runtime_ir.h"
#include "operators/include/softmax.h"
#include <cassert>
#include <iostream>
#include <string>

namespace my_vllm {
using sftensor = std::shared_ptr<ftensor>;
}

my_vllm::sftensor PreProcessImage(const cv::Mat &image)
{
    using namespace my_vllm;
    assert(!image.empty());
    // 调整输入大小
    cv::Mat resize_image;
    cv::resize(image, resize_image, cv::Size(224, 224));
    cv::Mat rgb_image;
    cv::cvtColor(resize_image, rgb_image, cv::COLOR_BGR2RGB);
    rgb_image.convertTo(rgb_image, CV_32FC3);
    std::vector<cv::Mat> split_images;
    cv::split(rgb_image, split_images);
    uint32_t input_w = 224;
    uint32_t input_h = 224;
    uint32_t input_c = 3;
    sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);
    uint32_t index = 0;
    for (const auto &split_image : split_images) 
    {
        assert(split_image.total() == input_w * input_h);
        const cv::Mat &split_image_t = split_image.t();
        memcpy(input->slice(index).memptr(), split_image_t.data,
              sizeof(float) * split_image.total());
        index += 1;
    }
    float mean_r = 0.485f;
    float mean_g = 0.456f;
    float mean_b = 0.406f;
    float var_r = 0.229f;
    float var_g = 0.224f;
    float var_b = 0.225f;
    assert(input->channels() == 3);
    input->data() = input->data() / 255.f;
    input->slice(0) = (input->slice(0) - mean_r) / var_r;
    input->slice(1) = (input->slice(1) - mean_g) / var_g;
    input->slice(2) = (input->slice(2) - mean_b) / var_b;
    return input;
}

int main(){
    using namespace my_vllm;
    const std::string param_path = "/home/xu/work/my_vllm/model/model_file/resnet18_batch1.pnnx.param";
    const std::string weight_path = "/home/xu/work/my_vllm/model/model_file/resnet18_batch1.pnnx.bin";
    RuntimeGraph graph(param_path, weight_path);

    graph.Build("pnnx_input_0", "pnnx_output_0");
    const uint32_t batch_size = 1;
    std::vector<sftensor> inputs;
    const std::string path("/home/xu/work/my_vllm/model/model_file/car.jpg");
    for (uint32_t i = 0; i < batch_size; ++i) 
    {
        cv::Mat image = cv::imread(path);
        if(image.empty()){
            std::cerr << "image read failed: " << path << std::endl;
            return -1;
        }
        // 图像预处理
        sftensor input = PreProcessImage(image);
        // input->Show();
        inputs.push_back(input);
    }
    auto outputs = graph.Forward(inputs, true);
    assert(outputs.size() == batch_size); // 替换ASSERT_EQ

    SoftmaxLayer softmax_layer(0);
    std::vector<sftensor> outputs_softmax(batch_size);
    softmax_layer.Forward(outputs, outputs_softmax);
    assert(outputs_softmax.size() == batch_size);

    for (int i = 0; i < outputs_softmax.size(); ++i) {
        const sftensor &output_tensor = outputs_softmax.at(i);
        assert(output_tensor->size() == 1 * 1000);
        // 找到类别概率最大的种类
        float max_prob = -1;
        int max_index = -1;
        for (int j = 0; j < output_tensor->size(); ++j) {
            float prob = output_tensor->index(j);
            if (max_prob <= prob) {
                max_prob = prob;
                max_index = j;
            }
        }
        printf("class with max prob is %f index %d\n", max_prob, max_index);
    }
    return 0;
}




