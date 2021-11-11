template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(512, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_lb_idx_0(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(512, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_idx_0(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_0(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_1(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_2(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_3(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_4(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_5(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_6(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_0(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_1(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_2(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_3(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_4(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 1) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_5(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
template <typename scalar_t2929, typename accscalar_t3030, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 1) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_6(const int ns3131, const accscalar_t3030 rheight3232, const accscalar_t3030 rwidth3333, const bool align_corners3434, const PackedTensorAccessor<scalar_t2929, 4> idata3535, PackedTensorAccessor<scalar_t2929, 4> odata3636, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    const int batchsize3737 = idata3535.size(0);
    const int channels3838 = idata3535.size(1);
    const int height13939 = idata3535.size(2);
    const int width14040 = idata3535.size(3);
    const int height24141 = odata3636.size(2);
    const int width24242 = odata3636.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns3131); index += blockDim_x_1 * gridDim.x) {
        const int w24343 = index % width24242;
        const int h24444 = index / width24242;
        if (height13939 == height24141 && width14040 == width24242) {
            const int h15555 = h24444;
            const int w15656 = w24343;
            for (int n = 0; n < batchsize3737; n++) {
                for (int c = 0; c < channels3838; ++c) {
                    const scalar_t2929 val5757 = idata3535[n][c][h15555][w15656];
                    odata3636[n][c][h24444][w24343] = val5757;
                }
            }
            return;
        }
        const accscalar_t3030 h1r4545 = area_pixel_compute_source_index<accscalar_t3030>(rheight3232, h24444, align_corners3434, false);
        const int h14646 = h1r4545;
        const int h1p4747 = (h14646 < height13939 - 1) ? 1 : 0;
        const accscalar_t3030 h1lambda4848 = h1r4545 - h14646;
        const accscalar_t3030 h0lambda4949 = static_cast<accscalar_t3030>(1) - h1lambda4848;
        const accscalar_t3030 w1r5050 = area_pixel_compute_source_index<accscalar_t3030>(rwidth3333, w24343, align_corners3434, false);
        const int w15151 = w1r5050;
        const int w1p5252 = (w15151 < width14040 - 1) ? 1 : 0;
        const accscalar_t3030 w1lambda5353 = w1r5050 - w15151;
        const accscalar_t3030 w0lambda5454 = static_cast<accscalar_t3030>(1) - w1lambda5353;
        for (int n = 0; n < batchsize3737; n++) {
            for (int c = 0; c < channels3838; ++c) {
                const accscalar_t3030 val5858 = h0lambda4949 * (w0lambda5454 * idata3535[n][c][h14646][w15151] + w1lambda5353 * idata3535[n][c][h14646][w15151 + w1p5252]) + h1lambda4848 * (w0lambda5454 * idata3535[n][c][h14646 + h1p4747][w15151] + w1lambda5353 * idata3535[n][c][h14646 + h1p4747][w15151 + w1p5252]);
                odata3636[n][c][h24444][w24343] = static_cast<scalar_t2929>(val5858);
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
    static int shared_n1212[160] __attribute__((shared));
    int plane1313 = blockIdx.x;
    int N1414 = input55.size(0) * input55.size(2);
    int tid1515 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t33 *shared_avg_var1616 = (stat_accscalar_t33 *)&shared_n1212[WARP_SIZE];
    stat_accscalar_t33 avg1717 = 0;
    stat_accscalar_t33 var_n1818 = 0;
    int n1919 = 0;
    for (int batch = threadIdx_y_0; batch < input55.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input55.size(2); x += blockDim_x_0) {
            stat_accscalar_t33 v2020 = input55[batch][plane1313][x];
            stat_accscalar_t33 d12121 = v2020 - avg1717;
            n1919++;
            avg1717 += d12121 / n1919;
            var_n1818 += d12121 * (v2020 - avg1717);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2222 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2323 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2424 = 1. / fmaxf(1., n1919 + o_n2323);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2222) * (avg1717 - o_avg2222) * n1919 * o_n2323 * factor2424;
        avg1717 = (n1919 * avg1717 + o_n2323 * o_avg2222) * factor2424;
        n1919 += o_n2323;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid1515 % WARP_SIZE == 0) {
        shared_n1212[tid1515 / WARP_SIZE] = n1919;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2] = avg1717;
        shared_avg_var1616[tid1515 / WARP_SIZE * 2 + 1] = var_n1818;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid1515 < WARP_SIZE) {
        n1919 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n1212[tid1515] : 0);
        avg1717 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515] : stat_accscalar_t33(0));
        var_n1818 = (tid1515 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var1616[2 * tid1515 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2525 = WARP_SHFL_XOR(avg1717, 1 << i, WARP_SIZE);
        int o_n2626 = WARP_SHFL_XOR(n1919, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2727 = 1. / fmaxf(1., n1919 + o_n2626);
        var_n1818 += WARP_SHFL_XOR(var_n1818, 1 << i, WARP_SIZE) + (avg1717 - o_avg2525) * (avg1717 - o_avg2525) * n1919 * o_n2626 * factor2727;
        avg1717 = (n1919 * avg1717 + o_n2626 * o_avg2525) * factor2727;
        n1919 += o_n2626;
    }
    if (tid1515 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1313] = avg1717;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1313] = VarTransform00<stat_accscalar_t33>({})(var_n1818 / N1414, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1313] + momentum77 * avg1717);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2828 = var_n1818 / (N1414 - 1);
            running_var99[plane1313] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1313] + momentum77 * unbiasedVar2828);
        }
    }
}
}
