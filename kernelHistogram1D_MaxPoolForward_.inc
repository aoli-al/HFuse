template <typename output_t33, typename input_t34, typename IndexType35, int ADims36, int PDims37, int BDims38, at::native::CUDAHistogramMemoryType MemoryType39 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op40, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_vfuse_lb_idx_0(TensorInfo<output_t33, IndexType35> a41, TensorInfo<output_t33, IndexType35> p42, TensorInfo<input_t34, IndexType35> b43, int nbins44, input_t34 minvalue45, input_t34 maxvalue46, IndexType35 totalElements47, Op40 getOp48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem49[] __attribute__((shared));
    output_t33 *smem50 = nullptr;
    smem50 = reinterpret_cast<output_t33 *>(my_smem49);
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        smem50[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType35 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements47; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType35 bOffset51 = IndexToOffset<input_t34, IndexType35, BDims38>::get(linearIndex, b43);
        const input_t34 bVal52 = b43.data[bOffset51];
        if (bVal52 >= minvalue45 && bVal52 <= maxvalue46) {
            const IndexType35 bin53 = getBin<input_t34, IndexType35>(bVal52, minvalue45, maxvalue46, nbins44);
            atomicAdd(& smem50[bin53], getOp48(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        const IndexType35 aOffset54 = IndexToOffset<output_t33, IndexType35, ADims36>::get(i, a41);
        atomicAdd(& a41.data[aOffset54], smem50[i]);
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
template <typename output_t33, typename input_t34, typename IndexType35, int ADims36, int PDims37, int BDims38, at::native::CUDAHistogramMemoryType MemoryType39 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op40, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_vfuse_idx_0(TensorInfo<output_t33, IndexType35> a41, TensorInfo<output_t33, IndexType35> p42, TensorInfo<input_t34, IndexType35> b43, int nbins44, input_t34 minvalue45, input_t34 maxvalue46, IndexType35 totalElements47, Op40 getOp48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem49[] __attribute__((shared));
    output_t33 *smem50 = nullptr;
    smem50 = reinterpret_cast<output_t33 *>(my_smem49);
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        smem50[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType35 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements47; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType35 bOffset51 = IndexToOffset<input_t34, IndexType35, BDims38>::get(linearIndex, b43);
        const input_t34 bVal52 = b43.data[bOffset51];
        if (bVal52 >= minvalue45 && bVal52 <= maxvalue46) {
            const IndexType35 bin53 = getBin<input_t34, IndexType35>(bVal52, minvalue45, maxvalue46, nbins44);
            atomicAdd(& smem50[bin53], getOp48(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        const IndexType35 aOffset54 = IndexToOffset<output_t33, IndexType35, ADims36>::get(i, a41);
        atomicAdd(& a41.data[aOffset54], smem50[i]);
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
template <typename output_t33, typename input_t34, typename IndexType35, int ADims36, int PDims37, int BDims38, at::native::CUDAHistogramMemoryType MemoryType39 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op40, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_idx_0(TensorInfo<output_t33, IndexType35> a41, TensorInfo<output_t33, IndexType35> p42, TensorInfo<input_t34, IndexType35> b43, int nbins44, input_t34 minvalue45, input_t34 maxvalue46, IndexType35 totalElements47, Op40 getOp48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    extern unsigned char my_smem49[] __attribute__((shared));
    output_t33 *smem50 = nullptr;
    smem50 = reinterpret_cast<output_t33 *>(my_smem49);
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        smem50[i] = 0;
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType35 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements47; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType35 bOffset51 = IndexToOffset<input_t34, IndexType35, BDims38>::get(linearIndex, b43);
        const input_t34 bVal52 = b43.data[bOffset51];
        if (bVal52 >= minvalue45 && bVal52 <= maxvalue46) {
            const IndexType35 bin53 = getBin<input_t34, IndexType35>(bVal52, minvalue45, maxvalue46, nbins44);
            atomicAdd(& smem50[bin53], getOp48(linearIndex));
        }
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        const IndexType35 aOffset54 = IndexToOffset<output_t33, IndexType35, ADims36>::get(i, a41);
        atomicAdd(& a41.data[aOffset54], smem50[i]);
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
template <typename output_t33, typename input_t34, typename IndexType35, int ADims36, int PDims37, int BDims38, at::native::CUDAHistogramMemoryType MemoryType39 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op40, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_idx_1(TensorInfo<output_t33, IndexType35> a41, TensorInfo<output_t33, IndexType35> p42, TensorInfo<input_t34, IndexType35> b43, int nbins44, input_t34 minvalue45, input_t34 maxvalue46, IndexType35 totalElements47, Op40 getOp48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    extern unsigned char my_smem49[] __attribute__((shared));
    output_t33 *smem50 = nullptr;
    smem50 = reinterpret_cast<output_t33 *>(my_smem49);
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        smem50[i] = 0;
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType35 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements47; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType35 bOffset51 = IndexToOffset<input_t34, IndexType35, BDims38>::get(linearIndex, b43);
        const input_t34 bVal52 = b43.data[bOffset51];
        if (bVal52 >= minvalue45 && bVal52 <= maxvalue46) {
            const IndexType35 bin53 = getBin<input_t34, IndexType35>(bVal52, minvalue45, maxvalue46, nbins44);
            atomicAdd(& smem50[bin53], getOp48(linearIndex));
        }
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        const IndexType35 aOffset54 = IndexToOffset<output_t33, IndexType35, ADims36>::get(i, a41);
        atomicAdd(& a41.data[aOffset54], smem50[i]);
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
template <typename output_t33, typename input_t34, typename IndexType35, int ADims36, int PDims37, int BDims38, at::native::CUDAHistogramMemoryType MemoryType39 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op40, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_idx_2(TensorInfo<output_t33, IndexType35> a41, TensorInfo<output_t33, IndexType35> p42, TensorInfo<input_t34, IndexType35> b43, int nbins44, input_t34 minvalue45, input_t34 maxvalue46, IndexType35 totalElements47, Op40 getOp48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    extern unsigned char my_smem49[] __attribute__((shared));
    output_t33 *smem50 = nullptr;
    smem50 = reinterpret_cast<output_t33 *>(my_smem49);
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        smem50[i] = 0;
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType35 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements47; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType35 bOffset51 = IndexToOffset<input_t34, IndexType35, BDims38>::get(linearIndex, b43);
        const input_t34 bVal52 = b43.data[bOffset51];
        if (bVal52 >= minvalue45 && bVal52 <= maxvalue46) {
            const IndexType35 bin53 = getBin<input_t34, IndexType35>(bVal52, minvalue45, maxvalue46, nbins44);
            atomicAdd(& smem50[bin53], getOp48(linearIndex));
        }
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        const IndexType35 aOffset54 = IndexToOffset<output_t33, IndexType35, ADims36>::get(i, a41);
        atomicAdd(& a41.data[aOffset54], smem50[i]);
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
template <typename output_t33, typename input_t34, typename IndexType35, int ADims36, int PDims37, int BDims38, at::native::CUDAHistogramMemoryType MemoryType39 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op40, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_idx_3(TensorInfo<output_t33, IndexType35> a41, TensorInfo<output_t33, IndexType35> p42, TensorInfo<input_t34, IndexType35> b43, int nbins44, input_t34 minvalue45, input_t34 maxvalue46, IndexType35 totalElements47, Op40 getOp48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem49[] __attribute__((shared));
    output_t33 *smem50 = nullptr;
    smem50 = reinterpret_cast<output_t33 *>(my_smem49);
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        smem50[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType35 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements47; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType35 bOffset51 = IndexToOffset<input_t34, IndexType35, BDims38>::get(linearIndex, b43);
        const input_t34 bVal52 = b43.data[bOffset51];
        if (bVal52 >= minvalue45 && bVal52 <= maxvalue46) {
            const IndexType35 bin53 = getBin<input_t34, IndexType35>(bVal52, minvalue45, maxvalue46, nbins44);
            atomicAdd(& smem50[bin53], getOp48(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        const IndexType35 aOffset54 = IndexToOffset<output_t33, IndexType35, ADims36>::get(i, a41);
        atomicAdd(& a41.data[aOffset54], smem50[i]);
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
template <typename output_t33, typename input_t34, typename IndexType35, int ADims36, int PDims37, int BDims38, at::native::CUDAHistogramMemoryType MemoryType39 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op40, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_idx_4(TensorInfo<output_t33, IndexType35> a41, TensorInfo<output_t33, IndexType35> p42, TensorInfo<input_t34, IndexType35> b43, int nbins44, input_t34 minvalue45, input_t34 maxvalue46, IndexType35 totalElements47, Op40 getOp48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    extern unsigned char my_smem49[] __attribute__((shared));
    output_t33 *smem50 = nullptr;
    smem50 = reinterpret_cast<output_t33 *>(my_smem49);
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        smem50[i] = 0;
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType35 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements47; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType35 bOffset51 = IndexToOffset<input_t34, IndexType35, BDims38>::get(linearIndex, b43);
        const input_t34 bVal52 = b43.data[bOffset51];
        if (bVal52 >= minvalue45 && bVal52 <= maxvalue46) {
            const IndexType35 bin53 = getBin<input_t34, IndexType35>(bVal52, minvalue45, maxvalue46, nbins44);
            atomicAdd(& smem50[bin53], getOp48(linearIndex));
        }
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        const IndexType35 aOffset54 = IndexToOffset<output_t33, IndexType35, ADims36>::get(i, a41);
        atomicAdd(& a41.data[aOffset54], smem50[i]);
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
template <typename output_t33, typename input_t34, typename IndexType35, int ADims36, int PDims37, int BDims38, at::native::CUDAHistogramMemoryType MemoryType39 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op40, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_idx_0(TensorInfo<output_t33, IndexType35> a41, TensorInfo<output_t33, IndexType35> p42, TensorInfo<input_t34, IndexType35> b43, int nbins44, input_t34 minvalue45, input_t34 maxvalue46, IndexType35 totalElements47, Op40 getOp48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    extern unsigned char my_smem49[] __attribute__((shared));
    output_t33 *smem50 = nullptr;
    smem50 = reinterpret_cast<output_t33 *>(my_smem49);
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        smem50[i] = 0;
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType35 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements47; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType35 bOffset51 = IndexToOffset<input_t34, IndexType35, BDims38>::get(linearIndex, b43);
        const input_t34 bVal52 = b43.data[bOffset51];
        if (bVal52 >= minvalue45 && bVal52 <= maxvalue46) {
            const IndexType35 bin53 = getBin<input_t34, IndexType35>(bVal52, minvalue45, maxvalue46, nbins44);
            atomicAdd(& smem50[bin53], getOp48(linearIndex));
        }
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        const IndexType35 aOffset54 = IndexToOffset<output_t33, IndexType35, ADims36>::get(i, a41);
        atomicAdd(& a41.data[aOffset54], smem50[i]);
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
template <typename output_t33, typename input_t34, typename IndexType35, int ADims36, int PDims37, int BDims38, at::native::CUDAHistogramMemoryType MemoryType39 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op40, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_idx_1(TensorInfo<output_t33, IndexType35> a41, TensorInfo<output_t33, IndexType35> p42, TensorInfo<input_t34, IndexType35> b43, int nbins44, input_t34 minvalue45, input_t34 maxvalue46, IndexType35 totalElements47, Op40 getOp48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    extern unsigned char my_smem49[] __attribute__((shared));
    output_t33 *smem50 = nullptr;
    smem50 = reinterpret_cast<output_t33 *>(my_smem49);
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        smem50[i] = 0;
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType35 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements47; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType35 bOffset51 = IndexToOffset<input_t34, IndexType35, BDims38>::get(linearIndex, b43);
        const input_t34 bVal52 = b43.data[bOffset51];
        if (bVal52 >= minvalue45 && bVal52 <= maxvalue46) {
            const IndexType35 bin53 = getBin<input_t34, IndexType35>(bVal52, minvalue45, maxvalue46, nbins44);
            atomicAdd(& smem50[bin53], getOp48(linearIndex));
        }
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        const IndexType35 aOffset54 = IndexToOffset<output_t33, IndexType35, ADims36>::get(i, a41);
        atomicAdd(& a41.data[aOffset54], smem50[i]);
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
template <typename output_t33, typename input_t34, typename IndexType35, int ADims36, int PDims37, int BDims38, at::native::CUDAHistogramMemoryType MemoryType39 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op40, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_idx_2(TensorInfo<output_t33, IndexType35> a41, TensorInfo<output_t33, IndexType35> p42, TensorInfo<input_t34, IndexType35> b43, int nbins44, input_t34 minvalue45, input_t34 maxvalue46, IndexType35 totalElements47, Op40 getOp48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    extern unsigned char my_smem49[] __attribute__((shared));
    output_t33 *smem50 = nullptr;
    smem50 = reinterpret_cast<output_t33 *>(my_smem49);
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        smem50[i] = 0;
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType35 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements47; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType35 bOffset51 = IndexToOffset<input_t34, IndexType35, BDims38>::get(linearIndex, b43);
        const input_t34 bVal52 = b43.data[bOffset51];
        if (bVal52 >= minvalue45 && bVal52 <= maxvalue46) {
            const IndexType35 bin53 = getBin<input_t34, IndexType35>(bVal52, minvalue45, maxvalue46, nbins44);
            atomicAdd(& smem50[bin53], getOp48(linearIndex));
        }
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        const IndexType35 aOffset54 = IndexToOffset<output_t33, IndexType35, ADims36>::get(i, a41);
        atomicAdd(& a41.data[aOffset54], smem50[i]);
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
template <typename output_t33, typename input_t34, typename IndexType35, int ADims36, int PDims37, int BDims38, at::native::CUDAHistogramMemoryType MemoryType39 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op40, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_idx_3(TensorInfo<output_t33, IndexType35> a41, TensorInfo<output_t33, IndexType35> p42, TensorInfo<input_t34, IndexType35> b43, int nbins44, input_t34 minvalue45, input_t34 maxvalue46, IndexType35 totalElements47, Op40 getOp48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem49[] __attribute__((shared));
    output_t33 *smem50 = nullptr;
    smem50 = reinterpret_cast<output_t33 *>(my_smem49);
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        smem50[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType35 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements47; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType35 bOffset51 = IndexToOffset<input_t34, IndexType35, BDims38>::get(linearIndex, b43);
        const input_t34 bVal52 = b43.data[bOffset51];
        if (bVal52 >= minvalue45 && bVal52 <= maxvalue46) {
            const IndexType35 bin53 = getBin<input_t34, IndexType35>(bVal52, minvalue45, maxvalue46, nbins44);
            atomicAdd(& smem50[bin53], getOp48(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        const IndexType35 aOffset54 = IndexToOffset<output_t33, IndexType35, ADims36>::get(i, a41);
        atomicAdd(& a41.data[aOffset54], smem50[i]);
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
template <typename output_t33, typename input_t34, typename IndexType35, int ADims36, int PDims37, int BDims38, at::native::CUDAHistogramMemoryType MemoryType39 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op40, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_idx_4(TensorInfo<output_t33, IndexType35> a41, TensorInfo<output_t33, IndexType35> p42, TensorInfo<input_t34, IndexType35> b43, int nbins44, input_t34 minvalue45, input_t34 maxvalue46, IndexType35 totalElements47, Op40 getOp48, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    extern unsigned char my_smem49[] __attribute__((shared));
    output_t33 *smem50 = nullptr;
    smem50 = reinterpret_cast<output_t33 *>(my_smem49);
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        smem50[i] = 0;
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType35 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements47; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType35 bOffset51 = IndexToOffset<input_t34, IndexType35, BDims38>::get(linearIndex, b43);
        const input_t34 bVal52 = b43.data[bOffset51];
        if (bVal52 >= minvalue45 && bVal52 <= maxvalue46) {
            const IndexType35 bin53 = getBin<input_t34, IndexType35>(bVal52, minvalue45, maxvalue46, nbins44);
            atomicAdd(& smem50[bin53], getOp48(linearIndex));
        }
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType35 i = threadIdx_x_1; i < a41.sizes[0]; i += blockDim_x_1) {
        const IndexType35 aOffset54 = IndexToOffset<output_t33, IndexType35, ADims36>::get(i, a41);
        atomicAdd(& a41.data[aOffset54], smem50[i]);
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
