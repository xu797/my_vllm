#ifndef MYVLLM_MODEL_RESNET_UTILS_H_
#define MYVLLM_MODEL_RESNET_UTILS_H_

#include <opencv2/opencv.hpp>

struct Detection {
  cv::Rect box;
  float conf = 0.f;
  int class_id = -1;
};

float Letterbox(
    const cv::Mat &image,
    cv::Mat &out_image,
    const cv::Size &new_shape = cv::Size(640, 640),
    int stride = 32,
    const cv::Scalar &color = cv::Scalar(114, 114, 114),
    bool fixed_shape = false,
    bool scale_up = false);

void ScaleCoords(const cv::Size &img_shape, cv::Rect &coords, const cv::Size &img_origin_shape);

#endif //MYVLLM_MODEL_RESNET_UTILS_H_
