template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(512, 2) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_lb_idx_0(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 32;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 32 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(512, 0) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_idx_0(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 32;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 32 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_0(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 56;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 56;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 56 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_1(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 48;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 48;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 48 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_2(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 40;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 40;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 40 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_3(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_4(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 24;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 24;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 24 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_5(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 16;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 16;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 16 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_6(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 8;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 8;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 8 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_0(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 56;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 56;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 56 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_1(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 48;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 48;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 48 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_2(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 40;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 40;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 40 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_3(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_4(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 24;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 24;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 24 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 1) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_5(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 16;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 16;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 16 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 1) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_6(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    int64_t _i_n_d_e_x45 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x45; _i_n_d_e_x45 < (n30); _i_n_d_e_x45 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x45) {
        int64_t w_out46 = index % width_col43;
        int idx47 = index / width_col43;
        int64_t h_out48 = idx47 % height_col42;
        int64_t channel_in49 = idx47 / height_col42;
        int64_t channel_out50 = channel_in49 * kernel_height34 * kernel_width35;
        int64_t h_in51 = h_out48 * stride_height38 - pad_height36;
        int64_t w_in52 = w_out46 * stride_width39 - pad_width37;
        dt29 *block_col53 = data_col44 + (channel_out50 * height_col42 + h_out48) * width_col43 + w_out46;
        const dt29 *block_im54 = data_im31 + (channel_in49 * height32 + h_in51) * width33 + w_in52;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h55 = h_in51 + i * dilation_height40;
                int64_t w56 = w_in52 + j * dilation_width41;
                * block_col53 = (h55 >= 0 && w56 >= 0 && h55 < height32 && w56 < width33) ? block_im54[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col53 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 8;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 8;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 8 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
