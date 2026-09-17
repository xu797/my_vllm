#include "tensor.h"
#include<assert.h>
#include<memory.h>


namespace my_vllm
{

Tensor<float>::Tensor(const Tensor& tensor)
{
    if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
}

Tensor<float>::Tensor(Tensor<float>&& tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = tensor.raw_shapes_;
  }
}

Tensor<float>& Tensor<float>::operator=(Tensor<float>&& tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

Tensor<float>::Tensor(uint32_t size)
{
    this->data_ = arma::fcube(1, size, 1);
    this->raw_shapes_ = std::vector<uint32_t>{size};
}

Tensor<float>::Tensor(uint32_t rows, uint32_t cols)
{
    this->data_ = arma::fcube(rows, cols, 1);
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
}

Tensor<float>::Tensor(u_int32_t channels, uint32_t rows, u_int32_t cols)
{
    this->data_ = arma::fcube(rows, cols, channels);
    if(channels == 1 && rows == 1)
    {
        this->raw_shapes_ = std::vector<uint32_t>{cols};
    }
    else if(channels == 1)
    {
        this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
    }
    else
    {
        this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
    }
}

Tensor<float>::Tensor(const std::vector<uint32_t> &shapes)
{
    CHECK(!shapes.empty() && shapes.size() <= 3);

    uint32_t remaining = 3 - shapes.size();
    std::vector<uint32_t> shapes_(3, 1);
    std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

    uint32_t channels = shapes_.at(0);
    uint32_t rows = shapes_.at(1);
    uint32_t cols = shapes_.at(2);

    data_ = arma::fcube(rows, cols, channels);
    if (channels == 1 && rows == 1) 
    {
        this->raw_shapes_ = std::vector<uint32_t>{cols};
    } 
    else if (channels == 1)
    {
        this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
    } 
    else 
    {
        this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
    }
}

uint32_t Tensor<float>::rows() const
{
    // CHECK(!this->data_.empty());
    assert(!this->data_.empty() && "tensor data is empty");
    return this->data_.n_rows;
}

uint32_t Tensor<float>::cols() const
{
    // CHECK(!this->data_.empty());
    assert(!this->data_.empty() && "tensor data is empty");
    return this->data_.n_cols;
}

uint32_t Tensor<float>::channels() const
{
    // CHECK(!this->data_.empty());
    assert(!this->data_.empty() && "tensor data is empty");
    return this->data_.n_slices;
}

uint32_t Tensor<float>::size() const
{
    // CHECK(!this->data_.empty());
    assert(!this->data_.empty() && "tensor data is empty");
    return this->data_.size();
}

const std::vector<uint32_t>& Tensor<float>::raw_shapes() const
{
    assert(!this->raw_shapes_.empty() && "raw_shapes_ vetcor is empty");
    return this->raw_shapes_;
}

bool Tensor<float>::empty() const
{
    return this->data_.empty();
}

void Tensor<float>::Fill(const std::vector<float>& values, bool row_major)
{
    assert(!this->data_.empty() && "tensor data is empty");
    const uint32_t total_elements = values.size();
    // assert((total_elements != this->data_.size()) && "size is not same");
    assert((total_elements == this->data_.size()) && "size is not same");

    if(row_major)
    {
        const uint32_t rows = this->rows();
        const uint32_t cols = this->cols();
        const uint32_t palne = rows * cols;
        const uint32_t channels = this->channels();

        for(uint32_t i = 0; i < channels; ++i)
        {
            auto &channel_data = this->data_.slice(i);
            const arma::fmat &channel_data_t = arma::fmat(values.data() + i * palne, cols, rows);
            channel_data = channel_data_t.t();
        }
    }
    else
    {
        std::copy(values.begin(), values.end(), this->data_.memptr());
    }

}

void Tensor<float>::Fill(float value) 
{
    this->data_.fill(value);
}

void Tensor<float>::Reshape(const std::vector<uint32_t>& shapes, bool row_major)
{
    assert(!this->data_.empty() && "tensor data is empty");
    assert(!shapes.empty() && "shapes is empty");
    const uint32_t origin_size = this->data_.size();
    const uint32_t current_size = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<>());
    assert((origin_size == current_size) && "size is not same");

    std::vector<float> values;
    if(row_major)
    {
        values = this->values(true);
    }

    if(shapes.size() == 3)
    {
        this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
        this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
    }
    else if(shapes.size() ==2)
    {
        this->data_.reshape(shapes.at(0), shapes.at(1), 1);
        this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
    }
    else
    {
        this->data_.reshape(1, shapes.at(0), 1);
        this->raw_shapes_ = {shapes.at(0)};
    }

    if(row_major)
    {
        this->Fill(values, true);
    }

}

std::vector<float> Tensor<float>::values(bool row_major)
{
    assert(!this->data_.empty() && "tensor data is empty");
    std::vector<float> values(this->data_.size());
    if(row_major)
    {
        const uint32_t rows = this->rows();
        const uint32_t cols = this->cols();
        const uint32_t palne = rows * cols;
        const uint32_t channels = this->channels();
        uint32_t index = 0;
        for(uint32_t i = 0; i < channels; ++i)
        {
            const arma::fmat &channel_data = this->data_.slice(i).t();
            std::copy(channel_data.begin(), channel_data.end(), values.begin() + index);
            index += palne;
        }
    }
    else
    {
        std::copy(this->data_.mem, this->data_.mem + this->data_.size(), values.begin());
    }

    return values;
}

float* Tensor<float>::raw_ptr()
{
    assert(!this->data_.empty() && "tensor data is empty");
    return this->data_.memptr();
}

float* Tensor<float>::raw_ptr(uint32_t offset)
{
    const uint32_t size = this->size();
    assert(offset < size && "offset > size");
    return this->data_.memptr() + offset;
}

void Tensor<float>::set_data(const arma::fcube& data)
{
    assert(this->data_.n_cols == data.n_cols && "n_cols is not same");
    assert(this->data_.n_rows == data.n_rows && "n_rows is not same");
    assert(this->data_.n_slices == data.n_slices && "n_slices is not same");
    this->data_ = data;
}

float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const 
{
    return this->data_.at(row, col, channel);
}

float& Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col)
{
    return this->data_.at(row, col, channel);
}
float Tensor<float>::index(uint32_t offset) const
{
    assert(offset < size() && "offset > size");
    return this->data_.at(offset);
}

float& Tensor<float>::index(uint32_t offset)
{
    assert(offset < size() && "offset > size");
    return this->data_.at(offset);
}

std::vector<uint32_t> Tensor<float>::shapes() const
{
    return {this->channels(), this->rows(), this->cols()};
}
   
arma::fcube& Tensor<float>::data()
{
    return this->data_;
}

const arma::fcube& Tensor<float>::data() const
{
    return this->data_;
}
arma::fmat& Tensor<float>::slice(uint32_t channel)
{
    return this->data_.slice(channel);
}

const arma::fmat& Tensor<float>::slice(uint32_t channel) const
{
    return this->data_.slice(channel);
}

void Tensor<float>::Ones()
{
    // this->data_.fill(1.0f);
    this->Fill(1.0f);
}

void Tensor<float>::Rand()
{
    this->data_.randn();
}

void Tensor<float>::Show()
{
    uint32_t channels = this->channels();
    for(uint32_t i = 0; i < channels; ++i)
    {     
        std::cout << "---------- Channel [" << i << "] ----------\n";
        // 获取第i通道矩阵 arma::fmat
        const arma::fmat& mat = this->data_.slice(i);
        // arma重载了<<，直接输出矩阵
        std::cout << mat << std::endl;
    }
}

void Tensor<float>::Flatten(bool row_major)
{
    assert(!this->data_.empty() && "tensor data is empty");
    uint32_t total = this->size();
    // 展平为一维：shape = {total}
    this->Reshape({total}, row_major);
}

void Tensor<float>::Transform(const std::function<float(float)>& filter)
{
    assert(!this->data_.empty() && "tensor data is empty");
    const float* ptr = this->data_.memptr();
    uint32_t total_elem = this->size();
    for(uint32_t i = 0; i < total_elem; ++i)
    {
        filter(ptr[i]);
    }
}

void Tensor<float>::Padding(const std::vector<uint32_t>& pads, float padding_value)
{
    assert(!this->data_.empty() && "tensor data is empty");
    assert(pads.size() == 4 && "pads must be [top,bottom,left,right]");

    uint32_t top    = pads[0];
    uint32_t bottom = pads[1];
    uint32_t left   = pads[2];
    uint32_t right  = pads[3];

    uint32_t old_rows = this->rows();
    uint32_t old_cols = this->cols();
    uint32_t chs      = this->channels();

    // 计算padding之后新的行列
    uint32_t new_rows = old_rows + top + bottom;
    uint32_t new_cols = old_cols + left + right;

    // 创建新fcube，全部填充padding_value
    arma::fcube new_cube(new_rows, new_cols, chs);
    new_cube.fill(padding_value);

    // 每个通道分别拷贝原矩阵到新矩阵的中间位置
    for(uint32_t c = 0; c < chs; ++c)
    {
        const arma::fmat& old_mat = this->data_.slice(c);
        arma::fmat& new_mat = new_cube.slice(c);
        // 拷贝：从new_mat的(top,left)开始，复制old_mat整块
        new_mat.submat(top, left, top + old_rows - 1, left + old_cols - 1) = old_mat;
    }

    // 替换内部数据
    this->data_ = std::move(new_cube);

    // 更新raw_shapes_
    auto& rs = this->raw_shapes_;
    if(rs.size() == 3)
    {
        // [ch,rows,cols]
        rs[1] = new_rows;
        rs[2] = new_cols;
    }
    else if(rs.size() == 2)
    {
        // [rows,cols]
        rs[0] = new_rows;
        rs[1] = new_cols;
    }
    else
    {
        // 一维张量做padding一般无意义，也可以assert拦截
        assert(false && "padding not support 1‑D tensor");
    }
}

float* Tensor<float>::matrix_raw_ptr(uint32_t index)
{   
    uint32_t plane = this->cols() * this->rows();
    return this->raw_ptr(plane * index);
}

} //end of namespace my_vllm