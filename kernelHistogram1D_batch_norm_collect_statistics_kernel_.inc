template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(512, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_lb_idx_0(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,512;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,512;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 32;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 32 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(512, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_idx_0(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,512;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,512;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 32;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 32 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_0(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,128;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,128;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 56;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 56;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 56 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_1(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,256;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,256;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 48;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 48;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 48 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_2(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,384;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,384;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 40;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 40;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 40 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_3(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,512;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,512;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_4(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,640;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,640;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 24;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 24;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 24 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_5(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,768;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,768;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 16;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 16;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 16 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_6(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,896;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,896;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 8;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 8;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 8 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_0(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,128;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,128;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 56;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 56;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 56 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_1(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,256;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,256;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 48;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 48;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 48 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_2(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,384;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,384;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 40;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 40;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 40 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_3(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,512;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,512;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_4(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,640;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,640;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 24;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 24;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 24 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_5(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,768;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,768;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 16;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 16;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 16 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
template <typename output_t2935, typename input_t3036, typename IndexType3137, int ADims3238, int PDims3339, int BDims3440, at::native::CUDAHistogramMemoryType MemoryType3541 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3642, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_6(TensorInfo<output_t2935, IndexType3137> a3743, TensorInfo<output_t2935, IndexType3137> p3844, TensorInfo<input_t3036, IndexType3137> b3945, int nbins4046, input_t3036 minvalue4147, input_t3036 maxvalue4248, IndexType3137 totalElements4349, Op3642 getOp4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    unsigned int blockDim_x_151 = 512;
    unsigned int threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153 = 1;
    unsigned int threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155 = 1;
    unsigned int threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    extern unsigned char my_smem4557[] __attribute__((shared));
    output_t2935 *smem4658 = nullptr;
    smem4658 = reinterpret_cast<output_t2935 *>(my_smem4557);
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        smem4658[i] = 0;
    }
    asm ("bar.sync 2,896;");
    ;
    for (IndexType3137 linearIndex = blockIdx.x * blockDim_x_151 + threadIdx_x_152; linearIndex < totalElements4349; linearIndex += gridDim.x * blockDim_x_151) {
        const IndexType3137 bOffset4759 = IndexToOffset<input_t3036, IndexType3137, BDims3440>::get(linearIndex, b3945);
        const input_t3036 bVal4860 = b3945.data[bOffset4759];
        if (bVal4860 >= minvalue4147 && bVal4860 <= maxvalue4248) {
            const IndexType3137 bin4961 = getBin<input_t3036, IndexType3137>(bVal4860, minvalue4147, maxvalue4248, nbins4046);
            atomicAdd(& smem4658[bin4961], getOp4450(linearIndex));
        }
    }
    asm ("bar.sync 2,896;");
    ;
    for (IndexType3137 i = threadIdx_x_152; i < a3743.sizes[0]; i += blockDim_x_151) {
        const IndexType3137 aOffset5062 = IndexToOffset<output_t2935, IndexType3137, ADims3238>::get(i, a3743);
        atomicAdd(& a3743.data[aOffset5062], smem4658[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 8;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 8;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 8 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    unsigned int blockDim_x_012 = 32;
    unsigned int threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014 = 16;
    unsigned int threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016 = 1;
    unsigned int threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319 = blockIdx.x;
    int N1420 = input55.size(0) * input55.size(2);
    int tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723 = 0;
    stat_accscalar_t33 var_n1824 = 0;
    int n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}
