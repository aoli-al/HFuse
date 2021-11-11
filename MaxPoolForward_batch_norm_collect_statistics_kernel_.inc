template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(512, 2) void MaxPoolForward_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_lb_idx_0(const int nthreads31, const scalar_t29 *bottom_data32, const int num33, const int channels34, const int height35, const int width36, const int pooled_height37, const int pooled_width38, const int kernel_h39, const int kernel_w40, const int stride_h41, const int stride_w42, const int pad_h43, const int pad_w44, const int dilation_h45, const int dilation_w46, scalar_t29 *top_data47, int64_t *top_mask48, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (nthreads31); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int pw50 = index % pooled_width38;
        int ph51 = (index / pooled_width38) % pooled_height37;
        int c52 = (index / pooled_width38 / pooled_height37) % channels34;
        int n53 = index / pooled_width38 / pooled_height37 / channels34;
        int hstart54 = ph51 * stride_h41 - pad_h43;
        int wstart55 = pw50 * stride_w42 - pad_w44;
        int hend56 = min(hstart54 + (kernel_h39 - 1) * dilation_h45 + 1, height35);
        int wend57 = min(wstart55 + (kernel_w40 - 1) * dilation_w46 + 1, width36);
        while (hstart54 < 0)
            hstart54 += dilation_h45;
        while (wstart55 < 0)
            wstart55 += dilation_w46;
        accscalar_t30 maxval58 = at::numeric_limits<accscalar_t30>::lower_bound();
        int maxidx59 = hstart54 * width36 + wstart55;
        const scalar_t29 *btm_data60 = bottom_data32 + (n53 * channels34 + c52) * height35 * width36;
        for (int h = hstart54; h < hend56; h += dilation_h45) {
            for (int w = wstart55; w < wend57; w += dilation_w46) {
                scalar_t29 val61 = btm_data60[h * width36 + w];
                if ((ScalarConvert<scalar_t29, accscalar_t30>::to(val61) > maxval58) || THCNumerics<scalar_t29>::isnan(val61)) {
                    maxidx59 = h * width36 + w;
                    maxval58 = ScalarConvert<scalar_t29, accscalar_t30>::to(val61);
                }
            }
        }
        top_data47[index] = ScalarConvert<scalar_t29, accscalar_t30>::to(maxval58);
        top_mask48[index] = maxidx59;
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(512, 0) void MaxPoolForward_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_idx_0(const int nthreads31, const scalar_t29 *bottom_data32, const int num33, const int channels34, const int height35, const int width36, const int pooled_height37, const int pooled_width38, const int kernel_h39, const int kernel_w40, const int stride_h41, const int stride_w42, const int pad_h43, const int pad_w44, const int dilation_h45, const int dilation_w46, scalar_t29 *top_data47, int64_t *top_mask48, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (nthreads31); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int pw50 = index % pooled_width38;
        int ph51 = (index / pooled_width38) % pooled_height37;
        int c52 = (index / pooled_width38 / pooled_height37) % channels34;
        int n53 = index / pooled_width38 / pooled_height37 / channels34;
        int hstart54 = ph51 * stride_h41 - pad_h43;
        int wstart55 = pw50 * stride_w42 - pad_w44;
        int hend56 = min(hstart54 + (kernel_h39 - 1) * dilation_h45 + 1, height35);
        int wend57 = min(wstart55 + (kernel_w40 - 1) * dilation_w46 + 1, width36);
        while (hstart54 < 0)
            hstart54 += dilation_h45;
        while (wstart55 < 0)
            wstart55 += dilation_w46;
        accscalar_t30 maxval58 = at::numeric_limits<accscalar_t30>::lower_bound();
        int maxidx59 = hstart54 * width36 + wstart55;
        const scalar_t29 *btm_data60 = bottom_data32 + (n53 * channels34 + c52) * height35 * width36;
        for (int h = hstart54; h < hend56; h += dilation_h45) {
            for (int w = wstart55; w < wend57; w += dilation_w46) {
                scalar_t29 val61 = btm_data60[h * width36 + w];
                if ((ScalarConvert<scalar_t29, accscalar_t30>::to(val61) > maxval58) || THCNumerics<scalar_t29>::isnan(val61)) {
                    maxidx59 = h * width36 + w;
                    maxval58 = ScalarConvert<scalar_t29, accscalar_t30>::to(val61);
                }
            }
        }
        top_data47[index] = ScalarConvert<scalar_t29, accscalar_t30>::to(maxval58);
        top_mask48[index] = maxidx59;
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(768, 0) void MaxPoolForward_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_0(const int nthreads31, const scalar_t29 *bottom_data32, const int num33, const int channels34, const int height35, const int width36, const int pooled_height37, const int pooled_width38, const int kernel_h39, const int kernel_w40, const int stride_h41, const int stride_w42, const int pad_h43, const int pad_w44, const int dilation_h45, const int dilation_w46, scalar_t29 *top_data47, int64_t *top_mask48, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (nthreads31); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int pw50 = index % pooled_width38;
        int ph51 = (index / pooled_width38) % pooled_height37;
        int c52 = (index / pooled_width38 / pooled_height37) % channels34;
        int n53 = index / pooled_width38 / pooled_height37 / channels34;
        int hstart54 = ph51 * stride_h41 - pad_h43;
        int wstart55 = pw50 * stride_w42 - pad_w44;
        int hend56 = min(hstart54 + (kernel_h39 - 1) * dilation_h45 + 1, height35);
        int wend57 = min(wstart55 + (kernel_w40 - 1) * dilation_w46 + 1, width36);
        while (hstart54 < 0)
            hstart54 += dilation_h45;
        while (wstart55 < 0)
            wstart55 += dilation_w46;
        accscalar_t30 maxval58 = at::numeric_limits<accscalar_t30>::lower_bound();
        int maxidx59 = hstart54 * width36 + wstart55;
        const scalar_t29 *btm_data60 = bottom_data32 + (n53 * channels34 + c52) * height35 * width36;
        for (int h = hstart54; h < hend56; h += dilation_h45) {
            for (int w = wstart55; w < wend57; w += dilation_w46) {
                scalar_t29 val61 = btm_data60[h * width36 + w];
                if ((ScalarConvert<scalar_t29, accscalar_t30>::to(val61) > maxval58) || THCNumerics<scalar_t29>::isnan(val61)) {
                    maxidx59 = h * width36 + w;
                    maxval58 = ScalarConvert<scalar_t29, accscalar_t30>::to(val61);
                }
            }
        }
        top_data47[index] = ScalarConvert<scalar_t29, accscalar_t30>::to(maxval58);
        top_mask48[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 40;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 40;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 40 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640;
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(768, 0) void MaxPoolForward_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_1(const int nthreads31, const scalar_t29 *bottom_data32, const int num33, const int channels34, const int height35, const int width36, const int pooled_height37, const int pooled_width38, const int kernel_h39, const int kernel_w40, const int stride_h41, const int stride_w42, const int pad_h43, const int pad_w44, const int dilation_h45, const int dilation_w46, scalar_t29 *top_data47, int64_t *top_mask48, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (nthreads31); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int pw50 = index % pooled_width38;
        int ph51 = (index / pooled_width38) % pooled_height37;
        int c52 = (index / pooled_width38 / pooled_height37) % channels34;
        int n53 = index / pooled_width38 / pooled_height37 / channels34;
        int hstart54 = ph51 * stride_h41 - pad_h43;
        int wstart55 = pw50 * stride_w42 - pad_w44;
        int hend56 = min(hstart54 + (kernel_h39 - 1) * dilation_h45 + 1, height35);
        int wend57 = min(wstart55 + (kernel_w40 - 1) * dilation_w46 + 1, width36);
        while (hstart54 < 0)
            hstart54 += dilation_h45;
        while (wstart55 < 0)
            wstart55 += dilation_w46;
        accscalar_t30 maxval58 = at::numeric_limits<accscalar_t30>::lower_bound();
        int maxidx59 = hstart54 * width36 + wstart55;
        const scalar_t29 *btm_data60 = bottom_data32 + (n53 * channels34 + c52) * height35 * width36;
        for (int h = hstart54; h < hend56; h += dilation_h45) {
            for (int w = wstart55; w < wend57; w += dilation_w46) {
                scalar_t29 val61 = btm_data60[h * width36 + w];
                if ((ScalarConvert<scalar_t29, accscalar_t30>::to(val61) > maxval58) || THCNumerics<scalar_t29>::isnan(val61)) {
                    maxidx59 = h * width36 + w;
                    maxval58 = ScalarConvert<scalar_t29, accscalar_t30>::to(val61);
                }
            }
        }
        top_data47[index] = ScalarConvert<scalar_t29, accscalar_t30>::to(maxval58);
        top_mask48[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 32;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 32 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512;
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(768, 0) void MaxPoolForward_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_2(const int nthreads31, const scalar_t29 *bottom_data32, const int num33, const int channels34, const int height35, const int width36, const int pooled_height37, const int pooled_width38, const int kernel_h39, const int kernel_w40, const int stride_h41, const int stride_w42, const int pad_h43, const int pad_w44, const int dilation_h45, const int dilation_w46, scalar_t29 *top_data47, int64_t *top_mask48, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (nthreads31); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int pw50 = index % pooled_width38;
        int ph51 = (index / pooled_width38) % pooled_height37;
        int c52 = (index / pooled_width38 / pooled_height37) % channels34;
        int n53 = index / pooled_width38 / pooled_height37 / channels34;
        int hstart54 = ph51 * stride_h41 - pad_h43;
        int wstart55 = pw50 * stride_w42 - pad_w44;
        int hend56 = min(hstart54 + (kernel_h39 - 1) * dilation_h45 + 1, height35);
        int wend57 = min(wstart55 + (kernel_w40 - 1) * dilation_w46 + 1, width36);
        while (hstart54 < 0)
            hstart54 += dilation_h45;
        while (wstart55 < 0)
            wstart55 += dilation_w46;
        accscalar_t30 maxval58 = at::numeric_limits<accscalar_t30>::lower_bound();
        int maxidx59 = hstart54 * width36 + wstart55;
        const scalar_t29 *btm_data60 = bottom_data32 + (n53 * channels34 + c52) * height35 * width36;
        for (int h = hstart54; h < hend56; h += dilation_h45) {
            for (int w = wstart55; w < wend57; w += dilation_w46) {
                scalar_t29 val61 = btm_data60[h * width36 + w];
                if ((ScalarConvert<scalar_t29, accscalar_t30>::to(val61) > maxval58) || THCNumerics<scalar_t29>::isnan(val61)) {
                    maxidx59 = h * width36 + w;
                    maxval58 = ScalarConvert<scalar_t29, accscalar_t30>::to(val61);
                }
            }
        }
        top_data47[index] = ScalarConvert<scalar_t29, accscalar_t30>::to(maxval58);
        top_mask48[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 24;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 24;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 24 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384;
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(768, 0) void MaxPoolForward_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_3(const int nthreads31, const scalar_t29 *bottom_data32, const int num33, const int channels34, const int height35, const int width36, const int pooled_height37, const int pooled_width38, const int kernel_h39, const int kernel_w40, const int stride_h41, const int stride_w42, const int pad_h43, const int pad_w44, const int dilation_h45, const int dilation_w46, scalar_t29 *top_data47, int64_t *top_mask48, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (nthreads31); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int pw50 = index % pooled_width38;
        int ph51 = (index / pooled_width38) % pooled_height37;
        int c52 = (index / pooled_width38 / pooled_height37) % channels34;
        int n53 = index / pooled_width38 / pooled_height37 / channels34;
        int hstart54 = ph51 * stride_h41 - pad_h43;
        int wstart55 = pw50 * stride_w42 - pad_w44;
        int hend56 = min(hstart54 + (kernel_h39 - 1) * dilation_h45 + 1, height35);
        int wend57 = min(wstart55 + (kernel_w40 - 1) * dilation_w46 + 1, width36);
        while (hstart54 < 0)
            hstart54 += dilation_h45;
        while (wstart55 < 0)
            wstart55 += dilation_w46;
        accscalar_t30 maxval58 = at::numeric_limits<accscalar_t30>::lower_bound();
        int maxidx59 = hstart54 * width36 + wstart55;
        const scalar_t29 *btm_data60 = bottom_data32 + (n53 * channels34 + c52) * height35 * width36;
        for (int h = hstart54; h < hend56; h += dilation_h45) {
            for (int w = wstart55; w < wend57; w += dilation_w46) {
                scalar_t29 val61 = btm_data60[h * width36 + w];
                if ((ScalarConvert<scalar_t29, accscalar_t30>::to(val61) > maxval58) || THCNumerics<scalar_t29>::isnan(val61)) {
                    maxidx59 = h * width36 + w;
                    maxval58 = ScalarConvert<scalar_t29, accscalar_t30>::to(val61);
                }
            }
        }
        top_data47[index] = ScalarConvert<scalar_t29, accscalar_t30>::to(maxval58);
        top_mask48[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 16;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 16;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 16 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256;
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(768, 0) void MaxPoolForward_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_4(const int nthreads31, const scalar_t29 *bottom_data32, const int num33, const int channels34, const int height35, const int width36, const int pooled_height37, const int pooled_width38, const int kernel_h39, const int kernel_w40, const int stride_h41, const int stride_w42, const int pad_h43, const int pad_w44, const int dilation_h45, const int dilation_w46, scalar_t29 *top_data47, int64_t *top_mask48, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (nthreads31); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int pw50 = index % pooled_width38;
        int ph51 = (index / pooled_width38) % pooled_height37;
        int c52 = (index / pooled_width38 / pooled_height37) % channels34;
        int n53 = index / pooled_width38 / pooled_height37 / channels34;
        int hstart54 = ph51 * stride_h41 - pad_h43;
        int wstart55 = pw50 * stride_w42 - pad_w44;
        int hend56 = min(hstart54 + (kernel_h39 - 1) * dilation_h45 + 1, height35);
        int wend57 = min(wstart55 + (kernel_w40 - 1) * dilation_w46 + 1, width36);
        while (hstart54 < 0)
            hstart54 += dilation_h45;
        while (wstart55 < 0)
            wstart55 += dilation_w46;
        accscalar_t30 maxval58 = at::numeric_limits<accscalar_t30>::lower_bound();
        int maxidx59 = hstart54 * width36 + wstart55;
        const scalar_t29 *btm_data60 = bottom_data32 + (n53 * channels34 + c52) * height35 * width36;
        for (int h = hstart54; h < hend56; h += dilation_h45) {
            for (int w = wstart55; w < wend57; w += dilation_w46) {
                scalar_t29 val61 = btm_data60[h * width36 + w];
                if ((ScalarConvert<scalar_t29, accscalar_t30>::to(val61) > maxval58) || THCNumerics<scalar_t29>::isnan(val61)) {
                    maxidx59 = h * width36 + w;
                    maxval58 = ScalarConvert<scalar_t29, accscalar_t30>::to(val61);
                }
            }
        }
        top_data47[index] = ScalarConvert<scalar_t29, accscalar_t30>::to(maxval58);
        top_mask48[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 8;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 8;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 8 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128;
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(768, 2) void MaxPoolForward_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_0(const int nthreads31, const scalar_t29 *bottom_data32, const int num33, const int channels34, const int height35, const int width36, const int pooled_height37, const int pooled_width38, const int kernel_h39, const int kernel_w40, const int stride_h41, const int stride_w42, const int pad_h43, const int pad_w44, const int dilation_h45, const int dilation_w46, scalar_t29 *top_data47, int64_t *top_mask48, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (nthreads31); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int pw50 = index % pooled_width38;
        int ph51 = (index / pooled_width38) % pooled_height37;
        int c52 = (index / pooled_width38 / pooled_height37) % channels34;
        int n53 = index / pooled_width38 / pooled_height37 / channels34;
        int hstart54 = ph51 * stride_h41 - pad_h43;
        int wstart55 = pw50 * stride_w42 - pad_w44;
        int hend56 = min(hstart54 + (kernel_h39 - 1) * dilation_h45 + 1, height35);
        int wend57 = min(wstart55 + (kernel_w40 - 1) * dilation_w46 + 1, width36);
        while (hstart54 < 0)
            hstart54 += dilation_h45;
        while (wstart55 < 0)
            wstart55 += dilation_w46;
        accscalar_t30 maxval58 = at::numeric_limits<accscalar_t30>::lower_bound();
        int maxidx59 = hstart54 * width36 + wstart55;
        const scalar_t29 *btm_data60 = bottom_data32 + (n53 * channels34 + c52) * height35 * width36;
        for (int h = hstart54; h < hend56; h += dilation_h45) {
            for (int w = wstart55; w < wend57; w += dilation_w46) {
                scalar_t29 val61 = btm_data60[h * width36 + w];
                if ((ScalarConvert<scalar_t29, accscalar_t30>::to(val61) > maxval58) || THCNumerics<scalar_t29>::isnan(val61)) {
                    maxidx59 = h * width36 + w;
                    maxval58 = ScalarConvert<scalar_t29, accscalar_t30>::to(val61);
                }
            }
        }
        top_data47[index] = ScalarConvert<scalar_t29, accscalar_t30>::to(maxval58);
        top_mask48[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 40;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 40;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 40 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640;
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(768, 2) void MaxPoolForward_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_1(const int nthreads31, const scalar_t29 *bottom_data32, const int num33, const int channels34, const int height35, const int width36, const int pooled_height37, const int pooled_width38, const int kernel_h39, const int kernel_w40, const int stride_h41, const int stride_w42, const int pad_h43, const int pad_w44, const int dilation_h45, const int dilation_w46, scalar_t29 *top_data47, int64_t *top_mask48, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (nthreads31); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int pw50 = index % pooled_width38;
        int ph51 = (index / pooled_width38) % pooled_height37;
        int c52 = (index / pooled_width38 / pooled_height37) % channels34;
        int n53 = index / pooled_width38 / pooled_height37 / channels34;
        int hstart54 = ph51 * stride_h41 - pad_h43;
        int wstart55 = pw50 * stride_w42 - pad_w44;
        int hend56 = min(hstart54 + (kernel_h39 - 1) * dilation_h45 + 1, height35);
        int wend57 = min(wstart55 + (kernel_w40 - 1) * dilation_w46 + 1, width36);
        while (hstart54 < 0)
            hstart54 += dilation_h45;
        while (wstart55 < 0)
            wstart55 += dilation_w46;
        accscalar_t30 maxval58 = at::numeric_limits<accscalar_t30>::lower_bound();
        int maxidx59 = hstart54 * width36 + wstart55;
        const scalar_t29 *btm_data60 = bottom_data32 + (n53 * channels34 + c52) * height35 * width36;
        for (int h = hstart54; h < hend56; h += dilation_h45) {
            for (int w = wstart55; w < wend57; w += dilation_w46) {
                scalar_t29 val61 = btm_data60[h * width36 + w];
                if ((ScalarConvert<scalar_t29, accscalar_t30>::to(val61) > maxval58) || THCNumerics<scalar_t29>::isnan(val61)) {
                    maxidx59 = h * width36 + w;
                    maxval58 = ScalarConvert<scalar_t29, accscalar_t30>::to(val61);
                }
            }
        }
        top_data47[index] = ScalarConvert<scalar_t29, accscalar_t30>::to(maxval58);
        top_mask48[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 32;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 32 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512;
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(768, 2) void MaxPoolForward_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_2(const int nthreads31, const scalar_t29 *bottom_data32, const int num33, const int channels34, const int height35, const int width36, const int pooled_height37, const int pooled_width38, const int kernel_h39, const int kernel_w40, const int stride_h41, const int stride_w42, const int pad_h43, const int pad_w44, const int dilation_h45, const int dilation_w46, scalar_t29 *top_data47, int64_t *top_mask48, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (nthreads31); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int pw50 = index % pooled_width38;
        int ph51 = (index / pooled_width38) % pooled_height37;
        int c52 = (index / pooled_width38 / pooled_height37) % channels34;
        int n53 = index / pooled_width38 / pooled_height37 / channels34;
        int hstart54 = ph51 * stride_h41 - pad_h43;
        int wstart55 = pw50 * stride_w42 - pad_w44;
        int hend56 = min(hstart54 + (kernel_h39 - 1) * dilation_h45 + 1, height35);
        int wend57 = min(wstart55 + (kernel_w40 - 1) * dilation_w46 + 1, width36);
        while (hstart54 < 0)
            hstart54 += dilation_h45;
        while (wstart55 < 0)
            wstart55 += dilation_w46;
        accscalar_t30 maxval58 = at::numeric_limits<accscalar_t30>::lower_bound();
        int maxidx59 = hstart54 * width36 + wstart55;
        const scalar_t29 *btm_data60 = bottom_data32 + (n53 * channels34 + c52) * height35 * width36;
        for (int h = hstart54; h < hend56; h += dilation_h45) {
            for (int w = wstart55; w < wend57; w += dilation_w46) {
                scalar_t29 val61 = btm_data60[h * width36 + w];
                if ((ScalarConvert<scalar_t29, accscalar_t30>::to(val61) > maxval58) || THCNumerics<scalar_t29>::isnan(val61)) {
                    maxidx59 = h * width36 + w;
                    maxval58 = ScalarConvert<scalar_t29, accscalar_t30>::to(val61);
                }
            }
        }
        top_data47[index] = ScalarConvert<scalar_t29, accscalar_t30>::to(maxval58);
        top_mask48[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 24;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 24;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 24 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384;
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(768, 2) void MaxPoolForward_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_3(const int nthreads31, const scalar_t29 *bottom_data32, const int num33, const int channels34, const int height35, const int width36, const int pooled_height37, const int pooled_width38, const int kernel_h39, const int kernel_w40, const int stride_h41, const int stride_w42, const int pad_h43, const int pad_w44, const int dilation_h45, const int dilation_w46, scalar_t29 *top_data47, int64_t *top_mask48, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (nthreads31); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int pw50 = index % pooled_width38;
        int ph51 = (index / pooled_width38) % pooled_height37;
        int c52 = (index / pooled_width38 / pooled_height37) % channels34;
        int n53 = index / pooled_width38 / pooled_height37 / channels34;
        int hstart54 = ph51 * stride_h41 - pad_h43;
        int wstart55 = pw50 * stride_w42 - pad_w44;
        int hend56 = min(hstart54 + (kernel_h39 - 1) * dilation_h45 + 1, height35);
        int wend57 = min(wstart55 + (kernel_w40 - 1) * dilation_w46 + 1, width36);
        while (hstart54 < 0)
            hstart54 += dilation_h45;
        while (wstart55 < 0)
            wstart55 += dilation_w46;
        accscalar_t30 maxval58 = at::numeric_limits<accscalar_t30>::lower_bound();
        int maxidx59 = hstart54 * width36 + wstart55;
        const scalar_t29 *btm_data60 = bottom_data32 + (n53 * channels34 + c52) * height35 * width36;
        for (int h = hstart54; h < hend56; h += dilation_h45) {
            for (int w = wstart55; w < wend57; w += dilation_w46) {
                scalar_t29 val61 = btm_data60[h * width36 + w];
                if ((ScalarConvert<scalar_t29, accscalar_t30>::to(val61) > maxval58) || THCNumerics<scalar_t29>::isnan(val61)) {
                    maxidx59 = h * width36 + w;
                    maxval58 = ScalarConvert<scalar_t29, accscalar_t30>::to(val61);
                }
            }
        }
        top_data47[index] = ScalarConvert<scalar_t29, accscalar_t30>::to(maxval58);
        top_mask48[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 16;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 16;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 16 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256;
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(768, 2) void MaxPoolForward_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_4(const int nthreads31, const scalar_t29 *bottom_data32, const int num33, const int channels34, const int height35, const int width36, const int pooled_height37, const int pooled_width38, const int kernel_h39, const int kernel_w40, const int stride_h41, const int stride_w42, const int pad_h43, const int pad_w44, const int dilation_h45, const int dilation_w46, scalar_t29 *top_data47, int64_t *top_mask48, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    int64_t _i_n_d_e_x49 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x49; _i_n_d_e_x49 < (nthreads31); _i_n_d_e_x49 += blockDim_x_1 * gridDim.x , index = _i_n_d_e_x49) {
        int pw50 = index % pooled_width38;
        int ph51 = (index / pooled_width38) % pooled_height37;
        int c52 = (index / pooled_width38 / pooled_height37) % channels34;
        int n53 = index / pooled_width38 / pooled_height37 / channels34;
        int hstart54 = ph51 * stride_h41 - pad_h43;
        int wstart55 = pw50 * stride_w42 - pad_w44;
        int hend56 = min(hstart54 + (kernel_h39 - 1) * dilation_h45 + 1, height35);
        int wend57 = min(wstart55 + (kernel_w40 - 1) * dilation_w46 + 1, width36);
        while (hstart54 < 0)
            hstart54 += dilation_h45;
        while (wstart55 < 0)
            wstart55 += dilation_w46;
        accscalar_t30 maxval58 = at::numeric_limits<accscalar_t30>::lower_bound();
        int maxidx59 = hstart54 * width36 + wstart55;
        const scalar_t29 *btm_data60 = bottom_data32 + (n53 * channels34 + c52) * height35 * width36;
        for (int h = hstart54; h < hend56; h += dilation_h45) {
            for (int w = wstart55; w < wend57; w += dilation_w46) {
                scalar_t29 val61 = btm_data60[h * width36 + w];
                if ((ScalarConvert<scalar_t29, accscalar_t30>::to(val61) > maxval58) || THCNumerics<scalar_t29>::isnan(val61)) {
                    maxidx59 = h * width36 + w;
                    maxval58 = ScalarConvert<scalar_t29, accscalar_t30>::to(val61);
                }
            }
        }
        top_data47[index] = ScalarConvert<scalar_t29, accscalar_t30>::to(maxval58);
        top_mask48[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 8;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 8;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 8 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128;
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
