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
    assert((total_elements != this->data_.size()) && "size is not same");

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

void Tensor<float>::Reshape(const std::vector<uint32_t>& shapes, bool row_major)
{
    assert(!this->data_.empty() && "tensor data is empty");
    assert(!shapes.empty() && "shapes is empty");
    const uint32_t origin_size = this->data_.size();
    const uint32_t current_size = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<>());
    assert((origin_size != current_size) && "size is not same");

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
    assert(size < offset && "offset > size");
    return this->data_.memptr() + offset;
}

}