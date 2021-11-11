template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(512, 2) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_vfuse_lb_idx_0(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(512, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_vfuse_idx_0(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_0(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_1(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_2(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_3(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_4(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_5(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 768;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_6(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_0 = 896;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(1024, 1) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_0(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(1024, 1) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_1(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_2(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_3(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_4(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(1024, 1) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_5(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 768;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
template <typename dt00, typename scalar_t2828, typename accscalar_t2929>
 __global__ __launch_bounds__(1024, 1) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_6(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, const int ns3030, const accscalar_t2929 rheight3131, const accscalar_t2929 rwidth3232, const bool align_corners3333, const PackedTensorAccessor<scalar_t2828, 4> idata3434, PackedTensorAccessor<scalar_t2828, 4> odata3535)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_0 = 896;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    int64_t _i_n_d_e_x1616 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x1616; _i_n_d_e_x1616 < (n11); _i_n_d_e_x1616 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x1616) {
        int64_t w_out1717 = index % width_col1414;
        int idx1818 = index / width_col1414;
        int64_t h_out1919 = idx1818 % height_col1313;
        int64_t channel_in2020 = idx1818 / height_col1313;
        int64_t channel_out2121 = channel_in2020 * kernel_height55 * kernel_width66;
        int64_t h_in2222 = h_out1919 * stride_height99 - pad_height77;
        int64_t w_in2323 = w_out1717 * stride_width1010 - pad_width88;
        dt00 *block_col2424 = data_col1515 + (channel_out2121 * height_col1313 + h_out1919) * width_col1414 + w_out1717;
        const dt00 *block_im2525 = data_im22 + (channel_in2020 * height33 + h_in2222) * width44 + w_in2323;
        for (int64_t i = 0; i < kernel_height55; ++i) {
            for (int64_t j = 0; j < kernel_width66; ++j) {
                int64_t h2626 = h_in2222 + i * dilation_height1111;
                int64_t w2727 = w_in2323 + j * dilation_width1212;
                * block_col2424 = (h2626 >= 0 && w2727 >= 0 && h2626 < height33 && w2727 < width44) ? block_im2525[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
                block_col2424 += height_col1313 * width_col1414;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    const int batchsize3636 = idata3434.size(0);
    const int channels3737 = idata3434.size(1);
    const int height13838 = idata3434.size(2);
    const int width13939 = idata3434.size(3);
    const int height24040 = odata3535.size(2);
    const int width24141 = odata3535.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3030); index += blockDim_x_1 * gridDim.x) {
        const int w24242 = index % width24141;
        const int h24343 = index / width24141;
        if (height13838 == height24040 && width13939 == width24141) {
            const int h15454 = h24343;
            const int w15555 = w24242;
            for (int n = 0; n < batchsize3636; n++) {
                for (int c = 0; c < channels3737; ++c) {
                    const scalar_t2828 val5656 = idata3434[n][c][h15454][w15555];
                    odata3535[n][c][h24343][w24242] = val5656;
                }
            }
            return;
        }
        const accscalar_t2929 h1r4444 = area_pixel_compute_source_index<accscalar_t2929>(rheight3131, h24343, align_corners3333, false);
        const int h14545 = h1r4444;
        const int h1p4646 = (h14545 < height13838 - 1) ? 1 : 0;
        const accscalar_t2929 h1lambda4747 = h1r4444 - h14545;
        const accscalar_t2929 h0lambda4848 = static_cast<accscalar_t2929>(1) - h1lambda4747;
        const accscalar_t2929 w1r4949 = area_pixel_compute_source_index<accscalar_t2929>(rwidth3232, w24242, align_corners3333, false);
        const int w15050 = w1r4949;
        const int w1p5151 = (w15050 < width13939 - 1) ? 1 : 0;
        const accscalar_t2929 w1lambda5252 = w1r4949 - w15050;
        const accscalar_t2929 w0lambda5353 = static_cast<accscalar_t2929>(1) - w1lambda5252;
        for (int n = 0; n < batchsize3636; n++) {
            for (int c = 0; c < channels3737; ++c) {
                const accscalar_t2929 val5757 = h0lambda4848 * (w0lambda5353 * idata3434[n][c][h14545][w15050] + w1lambda5252 * idata3434[n][c][h14545][w15050 + w1p5151]) + h1lambda4747 * (w0lambda5353 * idata3434[n][c][h14545 + h1p4646][w15050] + w1lambda5252 * idata3434[n][c][h14545 + h1p4646][w15050 + w1p5151]);
                odata3535[n][c][h24343][w24242] = static_cast<scalar_t2828>(val5757);
            }
        }
    }
}
}
