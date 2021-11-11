template <typename dt33, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 2) void im2col_kernel_MaxPoolForward_fused_kernel_vfuse_lb_idx_0(const int64_t n34, const dt33 *data_im35, const int64_t height36, const int64_t width37, const int64_t kernel_height38, const int64_t kernel_width39, const int64_t pad_height40, const int64_t pad_width41, const int64_t stride_height42, const int64_t stride_width43, const int64_t dilation_height44, const int64_t dilation_width45, const int64_t height_col46, const int64_t width_col47, dt33 *data_col48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (n34); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int64_t w_out50 = index % width_col47;
        int idx51 = index / width_col47;
        int64_t h_out52 = idx51 % height_col46;
        int64_t channel_in53 = idx51 / height_col46;
        int64_t channel_out54 = channel_in53 * kernel_height38 * kernel_width39;
        int64_t h_in55 = h_out52 * stride_height42 - pad_height40;
        int64_t w_in56 = w_out50 * stride_width43 - pad_width41;
        dt33 *block_col57 = data_col48 + (channel_out54 * height_col46 + h_out52) * width_col47 + w_out50;
        const dt33 *block_im58 = data_im35 + (channel_in53 * height36 + h_in55) * width37 + w_in56;
        for (int64_t i = 0; i < kernel_height38; ++i) {
            for (int64_t j = 0; j < kernel_width39; ++j) {
                int64_t h59 = h_in55 + i * dilation_height44;
                int64_t w60 = w_in56 + j * dilation_width45;
                * block_col57 = (h59 >= 0 && w60 >= 0 && h59 < height36 && w60 < width37) ? block_im58[i * dilation_height44 * width37 + j * dilation_width45] : ScalarConvert<int, dt33>::to(0);
                block_col57 += height_col46 * width_col47;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (nthreads2); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int pw21 = index % pooled_width9;
        int ph22 = (index / pooled_width9) % pooled_height8;
        int c23 = (index / pooled_width9 / pooled_height8) % channels5;
        int n24 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart25 = ph22 * stride_h12 - pad_h14;
        int wstart26 = pw21 * stride_w13 - pad_w15;
        int hend27 = min(hstart25 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend28 = min(wstart26 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart25 < 0)
            hstart25 += dilation_h16;
        while (wstart26 < 0)
            wstart26 += dilation_w17;
        accscalar_t1 maxval29 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx30 = hstart25 * width7 + wstart26;
        const scalar_t0 *btm_data31 = bottom_data3 + (n24 * channels5 + c23) * height6 * width7;
        for (int h = hstart25; h < hend27; h += dilation_h16) {
            for (int w = wstart26; w < wend28; w += dilation_w17) {
                scalar_t0 val32 = btm_data31[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val32) > maxval29) || THCNumerics<scalar_t0>::isnan(val32)) {
                    maxidx30 = h * width7 + w;
                    maxval29 = ScalarConvert<scalar_t0, accscalar_t1>::to(val32);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval29);
        top_mask19[index] = maxidx30;
    }
}
}
template <typename dt33, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 0) void im2col_kernel_MaxPoolForward_fused_kernel_vfuse_idx_0(const int64_t n34, const dt33 *data_im35, const int64_t height36, const int64_t width37, const int64_t kernel_height38, const int64_t kernel_width39, const int64_t pad_height40, const int64_t pad_width41, const int64_t stride_height42, const int64_t stride_width43, const int64_t dilation_height44, const int64_t dilation_width45, const int64_t height_col46, const int64_t width_col47, dt33 *data_col48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (n34); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int64_t w_out50 = index % width_col47;
        int idx51 = index / width_col47;
        int64_t h_out52 = idx51 % height_col46;
        int64_t channel_in53 = idx51 / height_col46;
        int64_t channel_out54 = channel_in53 * kernel_height38 * kernel_width39;
        int64_t h_in55 = h_out52 * stride_height42 - pad_height40;
        int64_t w_in56 = w_out50 * stride_width43 - pad_width41;
        dt33 *block_col57 = data_col48 + (channel_out54 * height_col46 + h_out52) * width_col47 + w_out50;
        const dt33 *block_im58 = data_im35 + (channel_in53 * height36 + h_in55) * width37 + w_in56;
        for (int64_t i = 0; i < kernel_height38; ++i) {
            for (int64_t j = 0; j < kernel_width39; ++j) {
                int64_t h59 = h_in55 + i * dilation_height44;
                int64_t w60 = w_in56 + j * dilation_width45;
                * block_col57 = (h59 >= 0 && w60 >= 0 && h59 < height36 && w60 < width37) ? block_im58[i * dilation_height44 * width37 + j * dilation_width45] : ScalarConvert<int, dt33>::to(0);
                block_col57 += height_col46 * width_col47;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (nthreads2); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int pw21 = index % pooled_width9;
        int ph22 = (index / pooled_width9) % pooled_height8;
        int c23 = (index / pooled_width9 / pooled_height8) % channels5;
        int n24 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart25 = ph22 * stride_h12 - pad_h14;
        int wstart26 = pw21 * stride_w13 - pad_w15;
        int hend27 = min(hstart25 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend28 = min(wstart26 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart25 < 0)
            hstart25 += dilation_h16;
        while (wstart26 < 0)
            wstart26 += dilation_w17;
        accscalar_t1 maxval29 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx30 = hstart25 * width7 + wstart26;
        const scalar_t0 *btm_data31 = bottom_data3 + (n24 * channels5 + c23) * height6 * width7;
        for (int h = hstart25; h < hend27; h += dilation_h16) {
            for (int w = wstart26; w < wend28; w += dilation_w17) {
                scalar_t0 val32 = btm_data31[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val32) > maxval29) || THCNumerics<scalar_t0>::isnan(val32)) {
                    maxidx30 = h * width7 + w;
                    maxval29 = ScalarConvert<scalar_t0, accscalar_t1>::to(val32);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval29);
        top_mask19[index] = maxidx30;
    }
}
}
template <typename dt33, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_idx_0(const int64_t n34, const dt33 *data_im35, const int64_t height36, const int64_t width37, const int64_t kernel_height38, const int64_t kernel_width39, const int64_t pad_height40, const int64_t pad_width41, const int64_t stride_height42, const int64_t stride_width43, const int64_t dilation_height44, const int64_t dilation_width45, const int64_t height_col46, const int64_t width_col47, dt33 *data_col48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (n34); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int64_t w_out50 = index % width_col47;
        int idx51 = index / width_col47;
        int64_t h_out52 = idx51 % height_col46;
        int64_t channel_in53 = idx51 / height_col46;
        int64_t channel_out54 = channel_in53 * kernel_height38 * kernel_width39;
        int64_t h_in55 = h_out52 * stride_height42 - pad_height40;
        int64_t w_in56 = w_out50 * stride_width43 - pad_width41;
        dt33 *block_col57 = data_col48 + (channel_out54 * height_col46 + h_out52) * width_col47 + w_out50;
        const dt33 *block_im58 = data_im35 + (channel_in53 * height36 + h_in55) * width37 + w_in56;
        for (int64_t i = 0; i < kernel_height38; ++i) {
            for (int64_t j = 0; j < kernel_width39; ++j) {
                int64_t h59 = h_in55 + i * dilation_height44;
                int64_t w60 = w_in56 + j * dilation_width45;
                * block_col57 = (h59 >= 0 && w60 >= 0 && h59 < height36 && w60 < width37) ? block_im58[i * dilation_height44 * width37 + j * dilation_width45] : ScalarConvert<int, dt33>::to(0);
                block_col57 += height_col46 * width_col47;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (nthreads2); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int pw21 = index % pooled_width9;
        int ph22 = (index / pooled_width9) % pooled_height8;
        int c23 = (index / pooled_width9 / pooled_height8) % channels5;
        int n24 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart25 = ph22 * stride_h12 - pad_h14;
        int wstart26 = pw21 * stride_w13 - pad_w15;
        int hend27 = min(hstart25 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend28 = min(wstart26 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart25 < 0)
            hstart25 += dilation_h16;
        while (wstart26 < 0)
            wstart26 += dilation_w17;
        accscalar_t1 maxval29 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx30 = hstart25 * width7 + wstart26;
        const scalar_t0 *btm_data31 = bottom_data3 + (n24 * channels5 + c23) * height6 * width7;
        for (int h = hstart25; h < hend27; h += dilation_h16) {
            for (int w = wstart26; w < wend28; w += dilation_w17) {
                scalar_t0 val32 = btm_data31[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val32) > maxval29) || THCNumerics<scalar_t0>::isnan(val32)) {
                    maxidx30 = h * width7 + w;
                    maxval29 = ScalarConvert<scalar_t0, accscalar_t1>::to(val32);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval29);
        top_mask19[index] = maxidx30;
    }
}
}
template <typename dt33, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_idx_1(const int64_t n34, const dt33 *data_im35, const int64_t height36, const int64_t width37, const int64_t kernel_height38, const int64_t kernel_width39, const int64_t pad_height40, const int64_t pad_width41, const int64_t stride_height42, const int64_t stride_width43, const int64_t dilation_height44, const int64_t dilation_width45, const int64_t height_col46, const int64_t width_col47, dt33 *data_col48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (n34); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int64_t w_out50 = index % width_col47;
        int idx51 = index / width_col47;
        int64_t h_out52 = idx51 % height_col46;
        int64_t channel_in53 = idx51 / height_col46;
        int64_t channel_out54 = channel_in53 * kernel_height38 * kernel_width39;
        int64_t h_in55 = h_out52 * stride_height42 - pad_height40;
        int64_t w_in56 = w_out50 * stride_width43 - pad_width41;
        dt33 *block_col57 = data_col48 + (channel_out54 * height_col46 + h_out52) * width_col47 + w_out50;
        const dt33 *block_im58 = data_im35 + (channel_in53 * height36 + h_in55) * width37 + w_in56;
        for (int64_t i = 0; i < kernel_height38; ++i) {
            for (int64_t j = 0; j < kernel_width39; ++j) {
                int64_t h59 = h_in55 + i * dilation_height44;
                int64_t w60 = w_in56 + j * dilation_width45;
                * block_col57 = (h59 >= 0 && w60 >= 0 && h59 < height36 && w60 < width37) ? block_im58[i * dilation_height44 * width37 + j * dilation_width45] : ScalarConvert<int, dt33>::to(0);
                block_col57 += height_col46 * width_col47;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (nthreads2); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int pw21 = index % pooled_width9;
        int ph22 = (index / pooled_width9) % pooled_height8;
        int c23 = (index / pooled_width9 / pooled_height8) % channels5;
        int n24 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart25 = ph22 * stride_h12 - pad_h14;
        int wstart26 = pw21 * stride_w13 - pad_w15;
        int hend27 = min(hstart25 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend28 = min(wstart26 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart25 < 0)
            hstart25 += dilation_h16;
        while (wstart26 < 0)
            wstart26 += dilation_w17;
        accscalar_t1 maxval29 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx30 = hstart25 * width7 + wstart26;
        const scalar_t0 *btm_data31 = bottom_data3 + (n24 * channels5 + c23) * height6 * width7;
        for (int h = hstart25; h < hend27; h += dilation_h16) {
            for (int w = wstart26; w < wend28; w += dilation_w17) {
                scalar_t0 val32 = btm_data31[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val32) > maxval29) || THCNumerics<scalar_t0>::isnan(val32)) {
                    maxidx30 = h * width7 + w;
                    maxval29 = ScalarConvert<scalar_t0, accscalar_t1>::to(val32);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval29);
        top_mask19[index] = maxidx30;
    }
}
}
template <typename dt33, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_idx_2(const int64_t n34, const dt33 *data_im35, const int64_t height36, const int64_t width37, const int64_t kernel_height38, const int64_t kernel_width39, const int64_t pad_height40, const int64_t pad_width41, const int64_t stride_height42, const int64_t stride_width43, const int64_t dilation_height44, const int64_t dilation_width45, const int64_t height_col46, const int64_t width_col47, dt33 *data_col48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (n34); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int64_t w_out50 = index % width_col47;
        int idx51 = index / width_col47;
        int64_t h_out52 = idx51 % height_col46;
        int64_t channel_in53 = idx51 / height_col46;
        int64_t channel_out54 = channel_in53 * kernel_height38 * kernel_width39;
        int64_t h_in55 = h_out52 * stride_height42 - pad_height40;
        int64_t w_in56 = w_out50 * stride_width43 - pad_width41;
        dt33 *block_col57 = data_col48 + (channel_out54 * height_col46 + h_out52) * width_col47 + w_out50;
        const dt33 *block_im58 = data_im35 + (channel_in53 * height36 + h_in55) * width37 + w_in56;
        for (int64_t i = 0; i < kernel_height38; ++i) {
            for (int64_t j = 0; j < kernel_width39; ++j) {
                int64_t h59 = h_in55 + i * dilation_height44;
                int64_t w60 = w_in56 + j * dilation_width45;
                * block_col57 = (h59 >= 0 && w60 >= 0 && h59 < height36 && w60 < width37) ? block_im58[i * dilation_height44 * width37 + j * dilation_width45] : ScalarConvert<int, dt33>::to(0);
                block_col57 += height_col46 * width_col47;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (nthreads2); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int pw21 = index % pooled_width9;
        int ph22 = (index / pooled_width9) % pooled_height8;
        int c23 = (index / pooled_width9 / pooled_height8) % channels5;
        int n24 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart25 = ph22 * stride_h12 - pad_h14;
        int wstart26 = pw21 * stride_w13 - pad_w15;
        int hend27 = min(hstart25 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend28 = min(wstart26 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart25 < 0)
            hstart25 += dilation_h16;
        while (wstart26 < 0)
            wstart26 += dilation_w17;
        accscalar_t1 maxval29 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx30 = hstart25 * width7 + wstart26;
        const scalar_t0 *btm_data31 = bottom_data3 + (n24 * channels5 + c23) * height6 * width7;
        for (int h = hstart25; h < hend27; h += dilation_h16) {
            for (int w = wstart26; w < wend28; w += dilation_w17) {
                scalar_t0 val32 = btm_data31[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val32) > maxval29) || THCNumerics<scalar_t0>::isnan(val32)) {
                    maxidx30 = h * width7 + w;
                    maxval29 = ScalarConvert<scalar_t0, accscalar_t1>::to(val32);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval29);
        top_mask19[index] = maxidx30;
    }
}
}
template <typename dt33, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_idx_3(const int64_t n34, const dt33 *data_im35, const int64_t height36, const int64_t width37, const int64_t kernel_height38, const int64_t kernel_width39, const int64_t pad_height40, const int64_t pad_width41, const int64_t stride_height42, const int64_t stride_width43, const int64_t dilation_height44, const int64_t dilation_width45, const int64_t height_col46, const int64_t width_col47, dt33 *data_col48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (n34); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int64_t w_out50 = index % width_col47;
        int idx51 = index / width_col47;
        int64_t h_out52 = idx51 % height_col46;
        int64_t channel_in53 = idx51 / height_col46;
        int64_t channel_out54 = channel_in53 * kernel_height38 * kernel_width39;
        int64_t h_in55 = h_out52 * stride_height42 - pad_height40;
        int64_t w_in56 = w_out50 * stride_width43 - pad_width41;
        dt33 *block_col57 = data_col48 + (channel_out54 * height_col46 + h_out52) * width_col47 + w_out50;
        const dt33 *block_im58 = data_im35 + (channel_in53 * height36 + h_in55) * width37 + w_in56;
        for (int64_t i = 0; i < kernel_height38; ++i) {
            for (int64_t j = 0; j < kernel_width39; ++j) {
                int64_t h59 = h_in55 + i * dilation_height44;
                int64_t w60 = w_in56 + j * dilation_width45;
                * block_col57 = (h59 >= 0 && w60 >= 0 && h59 < height36 && w60 < width37) ? block_im58[i * dilation_height44 * width37 + j * dilation_width45] : ScalarConvert<int, dt33>::to(0);
                block_col57 += height_col46 * width_col47;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (nthreads2); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int pw21 = index % pooled_width9;
        int ph22 = (index / pooled_width9) % pooled_height8;
        int c23 = (index / pooled_width9 / pooled_height8) % channels5;
        int n24 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart25 = ph22 * stride_h12 - pad_h14;
        int wstart26 = pw21 * stride_w13 - pad_w15;
        int hend27 = min(hstart25 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend28 = min(wstart26 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart25 < 0)
            hstart25 += dilation_h16;
        while (wstart26 < 0)
            wstart26 += dilation_w17;
        accscalar_t1 maxval29 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx30 = hstart25 * width7 + wstart26;
        const scalar_t0 *btm_data31 = bottom_data3 + (n24 * channels5 + c23) * height6 * width7;
        for (int h = hstart25; h < hend27; h += dilation_h16) {
            for (int w = wstart26; w < wend28; w += dilation_w17) {
                scalar_t0 val32 = btm_data31[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val32) > maxval29) || THCNumerics<scalar_t0>::isnan(val32)) {
                    maxidx30 = h * width7 + w;
                    maxval29 = ScalarConvert<scalar_t0, accscalar_t1>::to(val32);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval29);
        top_mask19[index] = maxidx30;
    }
}
}
template <typename dt33, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_idx_4(const int64_t n34, const dt33 *data_im35, const int64_t height36, const int64_t width37, const int64_t kernel_height38, const int64_t kernel_width39, const int64_t pad_height40, const int64_t pad_width41, const int64_t stride_height42, const int64_t stride_width43, const int64_t dilation_height44, const int64_t dilation_width45, const int64_t height_col46, const int64_t width_col47, dt33 *data_col48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (n34); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int64_t w_out50 = index % width_col47;
        int idx51 = index / width_col47;
        int64_t h_out52 = idx51 % height_col46;
        int64_t channel_in53 = idx51 / height_col46;
        int64_t channel_out54 = channel_in53 * kernel_height38 * kernel_width39;
        int64_t h_in55 = h_out52 * stride_height42 - pad_height40;
        int64_t w_in56 = w_out50 * stride_width43 - pad_width41;
        dt33 *block_col57 = data_col48 + (channel_out54 * height_col46 + h_out52) * width_col47 + w_out50;
        const dt33 *block_im58 = data_im35 + (channel_in53 * height36 + h_in55) * width37 + w_in56;
        for (int64_t i = 0; i < kernel_height38; ++i) {
            for (int64_t j = 0; j < kernel_width39; ++j) {
                int64_t h59 = h_in55 + i * dilation_height44;
                int64_t w60 = w_in56 + j * dilation_width45;
                * block_col57 = (h59 >= 0 && w60 >= 0 && h59 < height36 && w60 < width37) ? block_im58[i * dilation_height44 * width37 + j * dilation_width45] : ScalarConvert<int, dt33>::to(0);
                block_col57 += height_col46 * width_col47;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (nthreads2); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int pw21 = index % pooled_width9;
        int ph22 = (index / pooled_width9) % pooled_height8;
        int c23 = (index / pooled_width9 / pooled_height8) % channels5;
        int n24 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart25 = ph22 * stride_h12 - pad_h14;
        int wstart26 = pw21 * stride_w13 - pad_w15;
        int hend27 = min(hstart25 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend28 = min(wstart26 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart25 < 0)
            hstart25 += dilation_h16;
        while (wstart26 < 0)
            wstart26 += dilation_w17;
        accscalar_t1 maxval29 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx30 = hstart25 * width7 + wstart26;
        const scalar_t0 *btm_data31 = bottom_data3 + (n24 * channels5 + c23) * height6 * width7;
        for (int h = hstart25; h < hend27; h += dilation_h16) {
            for (int w = wstart26; w < wend28; w += dilation_w17) {
                scalar_t0 val32 = btm_data31[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val32) > maxval29) || THCNumerics<scalar_t0>::isnan(val32)) {
                    maxidx30 = h * width7 + w;
                    maxval29 = ScalarConvert<scalar_t0, accscalar_t1>::to(val32);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval29);
        top_mask19[index] = maxidx30;
    }
}
}
template <typename dt33, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_lb_idx_0(const int64_t n34, const dt33 *data_im35, const int64_t height36, const int64_t width37, const int64_t kernel_height38, const int64_t kernel_width39, const int64_t pad_height40, const int64_t pad_width41, const int64_t stride_height42, const int64_t stride_width43, const int64_t dilation_height44, const int64_t dilation_width45, const int64_t height_col46, const int64_t width_col47, dt33 *data_col48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (n34); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int64_t w_out50 = index % width_col47;
        int idx51 = index / width_col47;
        int64_t h_out52 = idx51 % height_col46;
        int64_t channel_in53 = idx51 / height_col46;
        int64_t channel_out54 = channel_in53 * kernel_height38 * kernel_width39;
        int64_t h_in55 = h_out52 * stride_height42 - pad_height40;
        int64_t w_in56 = w_out50 * stride_width43 - pad_width41;
        dt33 *block_col57 = data_col48 + (channel_out54 * height_col46 + h_out52) * width_col47 + w_out50;
        const dt33 *block_im58 = data_im35 + (channel_in53 * height36 + h_in55) * width37 + w_in56;
        for (int64_t i = 0; i < kernel_height38; ++i) {
            for (int64_t j = 0; j < kernel_width39; ++j) {
                int64_t h59 = h_in55 + i * dilation_height44;
                int64_t w60 = w_in56 + j * dilation_width45;
                * block_col57 = (h59 >= 0 && w60 >= 0 && h59 < height36 && w60 < width37) ? block_im58[i * dilation_height44 * width37 + j * dilation_width45] : ScalarConvert<int, dt33>::to(0);
                block_col57 += height_col46 * width_col47;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (nthreads2); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int pw21 = index % pooled_width9;
        int ph22 = (index / pooled_width9) % pooled_height8;
        int c23 = (index / pooled_width9 / pooled_height8) % channels5;
        int n24 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart25 = ph22 * stride_h12 - pad_h14;
        int wstart26 = pw21 * stride_w13 - pad_w15;
        int hend27 = min(hstart25 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend28 = min(wstart26 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart25 < 0)
            hstart25 += dilation_h16;
        while (wstart26 < 0)
            wstart26 += dilation_w17;
        accscalar_t1 maxval29 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx30 = hstart25 * width7 + wstart26;
        const scalar_t0 *btm_data31 = bottom_data3 + (n24 * channels5 + c23) * height6 * width7;
        for (int h = hstart25; h < hend27; h += dilation_h16) {
            for (int w = wstart26; w < wend28; w += dilation_w17) {
                scalar_t0 val32 = btm_data31[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val32) > maxval29) || THCNumerics<scalar_t0>::isnan(val32)) {
                    maxidx30 = h * width7 + w;
                    maxval29 = ScalarConvert<scalar_t0, accscalar_t1>::to(val32);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval29);
        top_mask19[index] = maxidx30;
    }
}
}
template <typename dt33, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_lb_idx_1(const int64_t n34, const dt33 *data_im35, const int64_t height36, const int64_t width37, const int64_t kernel_height38, const int64_t kernel_width39, const int64_t pad_height40, const int64_t pad_width41, const int64_t stride_height42, const int64_t stride_width43, const int64_t dilation_height44, const int64_t dilation_width45, const int64_t height_col46, const int64_t width_col47, dt33 *data_col48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (n34); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int64_t w_out50 = index % width_col47;
        int idx51 = index / width_col47;
        int64_t h_out52 = idx51 % height_col46;
        int64_t channel_in53 = idx51 / height_col46;
        int64_t channel_out54 = channel_in53 * kernel_height38 * kernel_width39;
        int64_t h_in55 = h_out52 * stride_height42 - pad_height40;
        int64_t w_in56 = w_out50 * stride_width43 - pad_width41;
        dt33 *block_col57 = data_col48 + (channel_out54 * height_col46 + h_out52) * width_col47 + w_out50;
        const dt33 *block_im58 = data_im35 + (channel_in53 * height36 + h_in55) * width37 + w_in56;
        for (int64_t i = 0; i < kernel_height38; ++i) {
            for (int64_t j = 0; j < kernel_width39; ++j) {
                int64_t h59 = h_in55 + i * dilation_height44;
                int64_t w60 = w_in56 + j * dilation_width45;
                * block_col57 = (h59 >= 0 && w60 >= 0 && h59 < height36 && w60 < width37) ? block_im58[i * dilation_height44 * width37 + j * dilation_width45] : ScalarConvert<int, dt33>::to(0);
                block_col57 += height_col46 * width_col47;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (nthreads2); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int pw21 = index % pooled_width9;
        int ph22 = (index / pooled_width9) % pooled_height8;
        int c23 = (index / pooled_width9 / pooled_height8) % channels5;
        int n24 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart25 = ph22 * stride_h12 - pad_h14;
        int wstart26 = pw21 * stride_w13 - pad_w15;
        int hend27 = min(hstart25 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend28 = min(wstart26 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart25 < 0)
            hstart25 += dilation_h16;
        while (wstart26 < 0)
            wstart26 += dilation_w17;
        accscalar_t1 maxval29 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx30 = hstart25 * width7 + wstart26;
        const scalar_t0 *btm_data31 = bottom_data3 + (n24 * channels5 + c23) * height6 * width7;
        for (int h = hstart25; h < hend27; h += dilation_h16) {
            for (int w = wstart26; w < wend28; w += dilation_w17) {
                scalar_t0 val32 = btm_data31[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val32) > maxval29) || THCNumerics<scalar_t0>::isnan(val32)) {
                    maxidx30 = h * width7 + w;
                    maxval29 = ScalarConvert<scalar_t0, accscalar_t1>::to(val32);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval29);
        top_mask19[index] = maxidx30;
    }
}
}
template <typename dt33, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_lb_idx_2(const int64_t n34, const dt33 *data_im35, const int64_t height36, const int64_t width37, const int64_t kernel_height38, const int64_t kernel_width39, const int64_t pad_height40, const int64_t pad_width41, const int64_t stride_height42, const int64_t stride_width43, const int64_t dilation_height44, const int64_t dilation_width45, const int64_t height_col46, const int64_t width_col47, dt33 *data_col48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (n34); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int64_t w_out50 = index % width_col47;
        int idx51 = index / width_col47;
        int64_t h_out52 = idx51 % height_col46;
        int64_t channel_in53 = idx51 / height_col46;
        int64_t channel_out54 = channel_in53 * kernel_height38 * kernel_width39;
        int64_t h_in55 = h_out52 * stride_height42 - pad_height40;
        int64_t w_in56 = w_out50 * stride_width43 - pad_width41;
        dt33 *block_col57 = data_col48 + (channel_out54 * height_col46 + h_out52) * width_col47 + w_out50;
        const dt33 *block_im58 = data_im35 + (channel_in53 * height36 + h_in55) * width37 + w_in56;
        for (int64_t i = 0; i < kernel_height38; ++i) {
            for (int64_t j = 0; j < kernel_width39; ++j) {
                int64_t h59 = h_in55 + i * dilation_height44;
                int64_t w60 = w_in56 + j * dilation_width45;
                * block_col57 = (h59 >= 0 && w60 >= 0 && h59 < height36 && w60 < width37) ? block_im58[i * dilation_height44 * width37 + j * dilation_width45] : ScalarConvert<int, dt33>::to(0);
                block_col57 += height_col46 * width_col47;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (nthreads2); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int pw21 = index % pooled_width9;
        int ph22 = (index / pooled_width9) % pooled_height8;
        int c23 = (index / pooled_width9 / pooled_height8) % channels5;
        int n24 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart25 = ph22 * stride_h12 - pad_h14;
        int wstart26 = pw21 * stride_w13 - pad_w15;
        int hend27 = min(hstart25 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend28 = min(wstart26 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart25 < 0)
            hstart25 += dilation_h16;
        while (wstart26 < 0)
            wstart26 += dilation_w17;
        accscalar_t1 maxval29 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx30 = hstart25 * width7 + wstart26;
        const scalar_t0 *btm_data31 = bottom_data3 + (n24 * channels5 + c23) * height6 * width7;
        for (int h = hstart25; h < hend27; h += dilation_h16) {
            for (int w = wstart26; w < wend28; w += dilation_w17) {
                scalar_t0 val32 = btm_data31[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val32) > maxval29) || THCNumerics<scalar_t0>::isnan(val32)) {
                    maxidx30 = h * width7 + w;
                    maxval29 = ScalarConvert<scalar_t0, accscalar_t1>::to(val32);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval29);
        top_mask19[index] = maxidx30;
    }
}
}
template <typename dt33, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_lb_idx_3(const int64_t n34, const dt33 *data_im35, const int64_t height36, const int64_t width37, const int64_t kernel_height38, const int64_t kernel_width39, const int64_t pad_height40, const int64_t pad_width41, const int64_t stride_height42, const int64_t stride_width43, const int64_t dilation_height44, const int64_t dilation_width45, const int64_t height_col46, const int64_t width_col47, dt33 *data_col48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (n34); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int64_t w_out50 = index % width_col47;
        int idx51 = index / width_col47;
        int64_t h_out52 = idx51 % height_col46;
        int64_t channel_in53 = idx51 / height_col46;
        int64_t channel_out54 = channel_in53 * kernel_height38 * kernel_width39;
        int64_t h_in55 = h_out52 * stride_height42 - pad_height40;
        int64_t w_in56 = w_out50 * stride_width43 - pad_width41;
        dt33 *block_col57 = data_col48 + (channel_out54 * height_col46 + h_out52) * width_col47 + w_out50;
        const dt33 *block_im58 = data_im35 + (channel_in53 * height36 + h_in55) * width37 + w_in56;
        for (int64_t i = 0; i < kernel_height38; ++i) {
            for (int64_t j = 0; j < kernel_width39; ++j) {
                int64_t h59 = h_in55 + i * dilation_height44;
                int64_t w60 = w_in56 + j * dilation_width45;
                * block_col57 = (h59 >= 0 && w60 >= 0 && h59 < height36 && w60 < width37) ? block_im58[i * dilation_height44 * width37 + j * dilation_width45] : ScalarConvert<int, dt33>::to(0);
                block_col57 += height_col46 * width_col47;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (nthreads2); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int pw21 = index % pooled_width9;
        int ph22 = (index / pooled_width9) % pooled_height8;
        int c23 = (index / pooled_width9 / pooled_height8) % channels5;
        int n24 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart25 = ph22 * stride_h12 - pad_h14;
        int wstart26 = pw21 * stride_w13 - pad_w15;
        int hend27 = min(hstart25 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend28 = min(wstart26 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart25 < 0)
            hstart25 += dilation_h16;
        while (wstart26 < 0)
            wstart26 += dilation_w17;
        accscalar_t1 maxval29 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx30 = hstart25 * width7 + wstart26;
        const scalar_t0 *btm_data31 = bottom_data3 + (n24 * channels5 + c23) * height6 * width7;
        for (int h = hstart25; h < hend27; h += dilation_h16) {
            for (int w = wstart26; w < wend28; w += dilation_w17) {
                scalar_t0 val32 = btm_data31[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val32) > maxval29) || THCNumerics<scalar_t0>::isnan(val32)) {
                    maxidx30 = h * width7 + w;
                    maxval29 = ScalarConvert<scalar_t0, accscalar_t1>::to(val32);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval29);
        top_mask19[index] = maxidx30;
    }
}
}
template <typename dt33, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_lb_idx_4(const int64_t n34, const dt33 *data_im35, const int64_t height36, const int64_t width37, const int64_t kernel_height38, const int64_t kernel_width39, const int64_t pad_height40, const int64_t pad_width41, const int64_t stride_height42, const int64_t stride_width43, const int64_t dilation_height44, const int64_t dilation_width45, const int64_t height_col46, const int64_t width_col47, dt33 *data_col48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (n34); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int64_t w_out50 = index % width_col47;
        int idx51 = index / width_col47;
        int64_t h_out52 = idx51 % height_col46;
        int64_t channel_in53 = idx51 / height_col46;
        int64_t channel_out54 = channel_in53 * kernel_height38 * kernel_width39;
        int64_t h_in55 = h_out52 * stride_height42 - pad_height40;
        int64_t w_in56 = w_out50 * stride_width43 - pad_width41;
        dt33 *block_col57 = data_col48 + (channel_out54 * height_col46 + h_out52) * width_col47 + w_out50;
        const dt33 *block_im58 = data_im35 + (channel_in53 * height36 + h_in55) * width37 + w_in56;
        for (int64_t i = 0; i < kernel_height38; ++i) {
            for (int64_t j = 0; j < kernel_width39; ++j) {
                int64_t h59 = h_in55 + i * dilation_height44;
                int64_t w60 = w_in56 + j * dilation_width45;
                * block_col57 = (h59 >= 0 && w60 >= 0 && h59 < height36 && w60 < width37) ? block_im58[i * dilation_height44 * width37 + j * dilation_width45] : ScalarConvert<int, dt33>::to(0);
                block_col57 += height_col46 * width_col47;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (nthreads2); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int pw21 = index % pooled_width9;
        int ph22 = (index / pooled_width9) % pooled_height8;
        int c23 = (index / pooled_width9 / pooled_height8) % channels5;
        int n24 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart25 = ph22 * stride_h12 - pad_h14;
        int wstart26 = pw21 * stride_w13 - pad_w15;
        int hend27 = min(hstart25 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend28 = min(wstart26 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart25 < 0)
            hstart25 += dilation_h16;
        while (wstart26 < 0)
            wstart26 += dilation_w17;
        accscalar_t1 maxval29 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx30 = hstart25 * width7 + wstart26;
        const scalar_t0 *btm_data31 = bottom_data3 + (n24 * channels5 + c23) * height6 * width7;
        for (int h = hstart25; h < hend27; h += dilation_h16) {
            for (int w = wstart26; w < wend28; w += dilation_w17) {
                scalar_t0 val32 = btm_data31[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val32) > maxval29) || THCNumerics<scalar_t0>::isnan(val32)) {
                    maxidx30 = h * width7 + w;
                    maxval29 = ScalarConvert<scalar_t0, accscalar_t1>::to(val32);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval29);
        top_mask19[index] = maxidx30;
    }
}
}
