#ifndef MYVLLM_TENSOR_H_
#define MYVLLM_TENSOR_H_

#include<vector>
#include<armadillo>
#include <glog/logging.h>

namespace my_vllm
{

template<typename T>
class Tensor;

template<>
class Tensor<float>
{
private:
    arma::fcube data_;
    std::vector<uint32_t> raw_shapes_;
public:
    Tensor() = default;
    ~Tensor(){};
    Tensor(const Tensor& tensor);  // 拷贝构造 Tensor<float> a; Tensor<float> b(a); a,b数据一样，但是相互独立。指向不同内存空间。
    Tensor(Tensor&& tensor) noexcept;                // 移动构造
    Tensor<float>& operator=(Tensor&& tensor) noexcept; // 移动赋值
    Tensor<float>& operator=(const Tensor& tensor);  // 拷贝赋值
    //explict禁止隐式转换
    explicit Tensor(uint32_t size);
    explicit Tensor(uint32_t rows, uint32_t cols);
    explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
    explicit Tensor(const std::vector<uint32_t> &shapes);

    uint32_t rows() const;
    uint32_t cols() const;
    uint32_t channels() const;
    uint32_t size() const;

    /*
    使用values数组里面的数据去初始化tensor
    @param: values,用来初始化的数组
    */
    void Fill(const std::vector<float>& values, bool row_major = true);
    void Fill(float value);
    /*
    改变张量形状
    @param: shapes
    */
    void Reshape(const std::vector<uint32_t>& shapes, bool row_major = true); 
    
    /**
   * 返回Tensor内的所有数据
   * @param row_major 是否是行主序列的
   * @return Tensor内的所有数据
   */
    std::vector<float> values(bool row_major = true);
    bool empty() const;

    /*  
    * 返回数据的原始指针
    * @return 返回数据的原始指针
    */
    float* raw_ptr();

    /**
     * 返回数据的原始指针
     * @param offset 数据指针的偏移量
     * @return 返回数据的原始指针
     */
    float* raw_ptr(uint32_t offset);

    void set_data(const arma::fcube& data); //设置张量中的具体数据
    float index(uint32_t offset) const; //返回张量中offset位置的元素
    float& index(uint32_t offset);  //返回张量中offset位置的元素

    std::vector<uint32_t> shapes() const;
    const std::vector<uint32_t>& raw_shapes() const;

    arma::fcube& data();
    const arma::fcube& data() const;
    arma::fmat& slice(uint32_t channel);
    const arma::fmat& slice(uint32_t channel) const;

    float at(uint32_t channel, uint32_t row, uint32_t col) const;
    float& at(uint32_t channel, uint32_t row, uint32_t col);

    void Padding(const std::vector<uint32_t>& pads, float padding_value);
    void Ones();
    void Rand();
    void Show();

    void Flatten(bool row_major = false);
    void Transform(const std::function<float(float)>& filter);
    float* matrix_raw_ptr(uint32_t index);
};

using ftensor = Tensor<float>;
using sftensor = std::shared_ptr<Tensor<float>>;
    
}

#endif