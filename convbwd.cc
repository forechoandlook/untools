#include <iostream>
#include <dnnl.hpp>

using namespace dnnl;

extern "C" void conv_bwd(float* src_data, float* weights_data, float* dst_data,
                         float* grad_input, float* grad_weight, float* grad_bias, // 输出的梯度指针
                         int batch_size, int in_channels, int out_channels,
                         int height, int width, int kernel_h, int kernel_w,
                         int out_h, int out_w, // 卷积参数
                         int stride_h, int stride_w, int pad_h, int pad_w) { // 支持自定义 stride 和 padding

    engine eng(engine::kind::cpu, 0);
    stream eng_stream(eng);

    // 使用传入的 stride 和 padding 参数
    memory::dims strides = {stride_h, stride_w};
    memory::dims padding_l = {pad_h, pad_w};
    memory::dims padding_r = {pad_h, pad_w};  // 对称填充

    // 创建内存描述符
    auto src_md = memory::desc({batch_size, in_channels, height, width}, memory::data_type::f32, memory::format_tag::nchw);
    auto weights_md = memory::desc({out_channels, in_channels, kernel_h, kernel_w}, memory::data_type::f32, memory::format_tag::oihw);
    auto dst_md = memory::desc({batch_size, out_channels, out_h, out_w}, memory::data_type::f32, memory::format_tag::nchw);
    // printf all info 
    std::cout << "batch_size: " << batch_size << " in_channels: " << in_channels << "  out_channels: " << out_channels << "  height: " << height << "  width: " << width << "  kernel_h: " << kernel_h << "\n";
    std::cout << "  kernel_w: " << kernel_w << "  out_h: " << out_h << "  out_w: " << out_w << "  stride_h: " << stride_h << "  stride_w: " << stride_w << "  pad_h: " << pad_h << "  pad_w: " << pad_w << std::endl;
    // todo add some check later 
    // 前向传播描述符
    auto conv_fwd_desc = convolution_forward::primitive_desc(
        eng, prop_kind::forward_training, algorithm::convolution_direct,
        src_md, weights_md, dst_md, strides, padding_l, padding_r);

    // 反向传播数据描述符
    auto conv_bwd_data_pd = convolution_backward_data::primitive_desc(
        eng, algorithm::convolution_direct,
        src_md, weights_md, dst_md, strides, padding_l, padding_r,
        conv_fwd_desc);

    // 反向传播权重描述符
    auto conv_bwd_weights_pd = convolution_backward_weights::primitive_desc(
        eng, algorithm::convolution_direct,
        src_md, weights_md, dst_md, strides, padding_l, padding_r,
        conv_fwd_desc);

    // 创建内存对象
    auto src_memory = memory(src_md, eng, src_data);
    auto weights_memory = memory(weights_md, eng, weights_data);
    auto dst_memory = memory(dst_md, eng, dst_data);

    // 创建梯度内存对象
    auto grad_input_memory = memory(src_md, eng, grad_input);
    auto grad_weight_memory = memory(weights_md, eng, grad_weight);
    auto grad_bias_memory = memory({{out_channels}, memory::data_type::f32, memory::format_tag::x}, eng, grad_bias);

    // 反向传播数据 (计算 grad_input)
    auto conv_bwd_data_prim = convolution_backward_data(conv_bwd_data_pd);
    conv_bwd_data_prim.execute(eng_stream,
        {{DNNL_ARG_DIFF_DST, dst_memory}, {DNNL_ARG_WEIGHTS, weights_memory}, {DNNL_ARG_DIFF_SRC, grad_input_memory}});

    // 反向传播权重 (计算 grad_weight 和 grad_bias)
    auto conv_bwd_weights_prim = convolution_backward_weights(conv_bwd_weights_pd);
    conv_bwd_weights_prim.execute(eng_stream,
        {{DNNL_ARG_DIFF_DST, dst_memory}, {DNNL_ARG_SRC, src_memory}, {DNNL_ARG_DIFF_WEIGHTS, grad_weight_memory}, {DNNL_ARG_DIFF_BIAS, grad_bias_memory}});

    eng_stream.wait();
}