/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tiny_dnn/core/kernels/maxpool_grad_op.h"
#include "tiny_dnn/core/kernels/maxpool_op.h"

#include "tiny_dnn/util/util.h"

#ifdef DNN_USE_IMAGE_API
#include "tiny_dnn/util/image.h"
#endif

namespace tiny_dnn {

/**
 * applies max-pooling operaton to the spatial data
 **/
class max_pooling_layer : public layer {
 public:
  /**
   * @param in_width     [in] width of input image
   * @param in_height    [in] height of input image
   * @param in_channels  [in] the number of input image channels(depth)
   * @param pooling_size [in] factor by which to downscale
   **/
  max_pooling_layer(size_t in_width,
                    size_t in_height,
                    size_t in_channels,
                    size_t pooling_size,
                    core::backend_t backend_type = core::default_engine())
    : max_pooling_layer(in_width,
                        in_height,
                        in_channels,
                        pooling_size,
                        (in_height == 1 ? 1 : pooling_size),
                        backend_type) {}

  max_pooling_layer(const shape3d &in_shape,
                    size_t pooling_size,
                    size_t stride,
                    core::backend_t backend_type = core::default_engine())
    : max_pooling_layer(in_shape.width_,
                        in_shape.height_,
                        in_shape.depth_,
                        pooling_size,
                        stride,
                        backend_type) {}

  max_pooling_layer(size_t in_width,
                    size_t in_height,
                    size_t in_channels,
                    size_t pooling_size,
                    size_t stride,
                    core::backend_t backend_type = core::default_engine())
    : max_pooling_layer(in_width,
                        in_height,
                        in_channels,
                        pooling_size,
                        (in_height == 1 ? 1 : pooling_size),
                        stride,
                        stride,
                        padding::valid,
                        backend_type) {}

  /**
   * @param in_width     [in] width of input image
   * @param in_height    [in] height of input image
   * @param in_channels  [in] the number of input image channels(depth)
   * @param pooling_size [in] factor by which to downscale
   * @param stride       [in] interval at which to apply the filters to the
   *input
  **/
  max_pooling_layer(size_t in_width,
                    size_t in_height,
                    size_t in_channels,
                    size_t pooling_size_x,
                    size_t pooling_size_y,
                    size_t stride_x,
                    size_t stride_y,
                    padding pad_type             = padding::valid,
                    core::backend_t backend_type = core::default_engine())
    : layer({vector_type::data}, {vector_type::data}) {


    size_t pad_along_height = ((conv_out_length(in_height, pooling_size_y, stride_y, pad_type) - 1) * stride_y + pooling_size_y - in_height);

    size_t pad_along_width = ((conv_out_length(in_width, pooling_size_x, stride_x, pad_type) - 1) * stride_x + pooling_size_x - in_width);
    set_maxpool_params(
      shape3d(in_width, in_height, in_channels),
      shape3d(in_width+pad_along_width, in_height+pad_along_height, in_channels),
      shape3d(conv_out_length(in_width, pooling_size_x, stride_x, pad_type),
              conv_out_length(in_height, pooling_size_y, stride_y, pad_type),
              in_channels),
      pooling_size_x, pooling_size_y, stride_x, stride_y, pad_type);

      

    init_connection();
    init_backend(backend_type);
    layer::set_backend_type(backend_type);
  }

  // move constructor
  max_pooling_layer(max_pooling_layer &&other)  // NOLINT
    : layer(std::move(other)), params_(std::move(other.params_)) {
    init_connection();
    init_backend(std::move(layer::engine()));
  }

  size_t fan_in_size() const override { return params_.out2in[0].size(); }

  size_t fan_out_size() const override { return 1; }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {

    /* pad_along_width = ((params_.out.width_ - 1) * w_stride + w_width - in.width_); */
    /* conv_params.pad_top = pad_along_height / 2; */
    /* params_.pad_bottom = pad_along_height - params_.pad_top; */
    /* params_.pad_left = pad_along_width / 2; */
    /* params_.pad_right = pad_along_width - params_.pad_left; */

    // forward convolutional op context
    fwd_ctx_.set_in_out(in_data, out_data);
    fwd_ctx_.setParallelize(layer::parallelize());
    fwd_ctx_.setEngine(layer::engine());

    // hackey padding job
    core::conv_params conv_params;
    /* std::cout << "params_.out.height_" << params_.out.height_ << std::endl; */
    size_t pad_along_height = ((params_.out.height_ -1) * params_.stride_y + params_.pool_size_y - params_.in.height_);

    size_t pad_along_width = ((params_.out.width_ - 1) * params_.stride_x + params_.pool_size_x - params_.in.width_);

    conv_params.pad_top = pad_along_height / 2;
    conv_params.pad_bottom = pad_along_height - conv_params.pad_top;
    conv_params.pad_left = pad_along_width / 2;
    conv_params.pad_right = pad_along_width - conv_params.pad_left;
    std::cout << "pad_top" << conv_params.pad_top << std::endl;
    std::cout << "pad_bottom" << conv_params.pad_bottom << std::endl;
    std::cout << "pad_left" << conv_params.pad_left << std::endl;
    std::cout << "pad_right" << conv_params.pad_right << std::endl;

    conv_params.in_padded =
      shape3d(in_length(params_.in.width_, pad_along_width, params_.pad_type),
              in_length(params_.in.height_, pad_along_height, params_.pad_type), params_.in.depth_);
    // set in params to padded size for correct index calculation
    conv_params.in =
      shape3d(params_.in.width_, params_.in.height_, params_.in.depth_);

    std::cout << conv_params.in_padded.height_ << std::endl;
    std::cout << conv_params.in_padded.width_ << std::endl;

    core::Conv2dPaddingTF padding_op = core::Conv2dPaddingTF(conv_params);



    // apply padding to the input tensor
    std::cout << "mp padding" << std::endl;
    padding_op.copy_and_pad_input(*in_data[0], cws_.prev_out_padded_);

    fwd_in_data_.resize(in_data.size());
    std::copy(in_data.begin(), in_data.end(), fwd_in_data_.begin());
    fwd_in_data_[0] = &cws_.prev_out_padded_;



    fwd_ctx_.set_in_out(fwd_in_data_, out_data);


    /* const vec_t &in          = fwd_ctx_.input(0)[0]; */
    /* std::cout << "paddedData" << std::endl; */
    /* int i = 0; */
    /* for (auto a : in){ */
    /*   std::cout << i << " " << a << std::endl; */
    /*   i++; */
    /* } */
    /* // init padding buffer */
    /* if (params_.pad_type == padding::same) { */
    /*   std::cout << params_.in_padded.size() << std::endl; */
    /*   cws_.prev_delta_padded_.resize( */
    /*     1, vec_t(params_.in_padded.size(), float_t(0))); */
    /* } */
    // launch convolutional kernel
    kernel_fwd_->compute(fwd_ctx_);
  }

  size_t in_length(size_t in_length,
                   size_t dim_padding,
                   padding pad_type) const {
    return pad_type == padding::same ? (in_length + dim_padding)
                                     : in_length;
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    // backward convolutional op context
    bwd_ctx_.set_in_out(in_data, out_data, out_grad, in_grad);
    bwd_ctx_.setParallelize(layer::parallelize());
    bwd_ctx_.setEngine(layer::engine());

    // launch convolutional kernel
    kernel_back_->compute(bwd_ctx_);
  }

  std::vector<index3d<size_t>> in_shape() const override {
    return {params_.in};
  }

  std::vector<index3d<size_t>> out_shape() const override {
    return {params_.out};
  }

  std::string layer_type() const override { return std::string("max-pool"); }

  std::string kernel_file() const override {
    return std::string("../tiny_cnn/core/kernels/cl_kernels/pooling.cl");
  }

  std::pair<size_t, size_t> pool_size() const {
    return std::make_pair(params_.pool_size_x, params_.pool_size_y);
  }

  void set_sample_count(size_t sample_count) override {
    layer::set_sample_count(sample_count);
    params_.out2inmax.resize(sample_count,
                             std::vector<size_t>(params_.out.size()));
  }

  friend struct serialization_buddy;

 private:
  /* The Max Poling operation params */
  core::maxpool_params params_;

  /* forward op context */
  core::OpKernelContext fwd_ctx_;

  /* backward op context */
  core::OpKernelContext bwd_ctx_;

  /* Forward and backward ops */
  std::shared_ptr<core::OpKernel> kernel_fwd_;
  std::shared_ptr<core::OpKernel> kernel_back_;

  std::vector<tensor_t *> fwd_in_data_;

  /* Buffer to store padded data */
  struct conv_layer_worker_specific_storage {
    tensor_t prev_out_padded_;
    tensor_t prev_delta_padded_;
  } cws_;

  void connect_kernel(size_t pooling_size_x,
                      size_t pooling_size_y,
                      size_t outx,
                      size_t outy,
                      size_t c) {
    size_t dxmax =
      std::min(pooling_size_x, params_.in_padded.width_ - outx * params_.stride_x);

    size_t dymax =
      std::min(pooling_size_y, params_.in_padded.height_ - outy * params_.stride_y);
      /* std::cout << "pooling_size_y" << pooling_size_y << std::endl; */
      /* std::cout << "pooling_size_x" << pooling_size_x << std::endl; */
      /* std::cout << "outx" << outx << std::endl; */
      /* std::cout << "outy" << outy << std::endl; */
      /* std::cout << "params_.in.height" << params_.in.height_ << std::endl; */
      /* std::cout << "dxmax" << dxmax << std::endl; */
      /* std::cout << "dymax" << dymax << std::endl; */
      /* std::cout << "in size" << params_.in_padded.height_ << " " << params_.in_padded.width_ << std::endl; */

    /* std::cout << "outx" << outx << "outy" << outy << std::endl; */
    for (size_t dy = 0; dy < dymax; dy++) {
      for (size_t dx = 0; dx < dxmax; dx++) {
        size_t in_index = params_.in_padded.get_index(outx * params_.stride_x + dx,
                                               outy * params_.stride_y + dy, c);
        /* std::cout << " in_index " << in_index << std::endl; */
        size_t out_index = params_.out.get_index(outx, outy, c);

        if (in_index >= params_.in2out.size()) {
          throw nn_error("index overflow");
        }
        if (out_index >= params_.out2in.size()) {
          throw nn_error("index overflow");
        }
        params_.in2out[in_index] = out_index;
        params_.out2in[out_index].push_back(in_index);
      }
    }
  }

  void init_connection() {
    params_.in2out.resize(params_.in_padded.size());
    params_.out2in.resize(params_.out.size());

    for (size_t c = 0; c < params_.in.depth_; ++c) {
      for (size_t y = 0; y < params_.out.height_; ++y) {
        for (size_t x = 0; x < params_.out.width_; ++x) {
          connect_kernel(params_.pool_size_x, params_.pool_size_y, x, y, c);
        }
      }
    }
  }

  void init_backend(core::backend_t backend_type) {
    core::OpKernelConstruction ctx =
      core::OpKernelConstruction(layer::device(), &params_);

    if (backend_type == core::backend_t::internal ||
        backend_type == core::backend_t::nnpack ||
        backend_type == core::backend_t::avx) {
      kernel_fwd_.reset(new MaxPoolOp(ctx));
      kernel_back_.reset(new MaxPoolGradOp(ctx));
      return;
    } else {
      throw nn_error("Not supported engine: " + to_string(backend_type));
    }
  }

  void set_maxpool_params(const shape3d &in,
                          const shape3d &in_padded,
                          const shape3d &out,
                          size_t pooling_size_x,
                          size_t pooling_size_y,
                          size_t stride_x,
                          size_t stride_y,
                          padding pad_type) {
    params_.in          = in;
    params_.in_padded   = in_padded;
    params_.out         = out;
    params_.pool_size_x = pooling_size_x;
    params_.pool_size_y = pooling_size_y;
    params_.stride_x    = stride_x;
    params_.stride_y    = stride_y;
    params_.pad_type    = pad_type;
  }
};

}  // namespace tiny_dnn
