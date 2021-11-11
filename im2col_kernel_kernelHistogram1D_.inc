template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(512, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_vfuse_lb_idx_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(512, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_vfuse_idx_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_1(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_2(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_3(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_4(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_5(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 768;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_6(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_0 = 896;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_1(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_2(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_3(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_4(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(1024, 1) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_5(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 768;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
template <typename dt0, typename output_t28, typename input_t29, typename IndexType30, int ADims31, int PDims32, int BDims33, at::native::CUDAHistogramMemoryType MemoryType34 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op35>
 __global__ __launch_bounds__(1024, 1) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_6(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t28, IndexType30> a36, TensorInfo<output_t28, IndexType30> p37, TensorInfo<input_t29, IndexType30> b38, int nbins39, input_t29 minvalue40, input_t29 maxvalue41, IndexType30 totalElements42, Op35 getOp43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_0 = 896;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    int64_t _i_n_d_e_x16 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x16; _i_n_d_e_x16 < (n1); _i_n_d_e_x16 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x16) {
        int64_t w_out17 = index % width_col14;
        int idx18 = index / width_col14;
        int64_t h_out19 = idx18 % height_col13;
        int64_t channel_in20 = idx18 / height_col13;
        int64_t channel_out21 = channel_in20 * kernel_height5 * kernel_width6;
        int64_t h_in22 = h_out19 * stride_height9 - pad_height7;
        int64_t w_in23 = w_out17 * stride_width10 - pad_width8;
        dt0 *block_col24 = data_col15 + (channel_out21 * height_col13 + h_out19) * width_col14 + w_out17;
        const dt0 *block_im25 = data_im2 + (channel_in20 * height3 + h_in22) * width4 + w_in23;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h26 = h_in22 + i * dilation_height11;
                int64_t w27 = w_in23 + j * dilation_width12;
                * block_col24 = (h26 >= 0 && w27 >= 0 && h26 < height3 && w27 < width4) ? block_im25[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col24 += height_col13 * width_col14;
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
    extern unsigned char my_smem44[] __attribute__((shared));
    output_t28 *smem45 = nullptr;
    smem45 = reinterpret_cast<output_t28 *>(my_smem44);
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        smem45[i] = 0;
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType30 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements42; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType30 bOffset46 = IndexToOffset<input_t29, IndexType30, BDims33>::get(linearIndex, b38);
        const input_t29 bVal47 = b38.data[bOffset46];
        if (bVal47 >= minvalue40 && bVal47 <= maxvalue41) {
            const IndexType30 bin48 = getBin<input_t29, IndexType30>(bVal47, minvalue40, maxvalue41, nbins39);
            atomicAdd(& smem45[bin48], getOp43(linearIndex));
        }
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType30 i = threadIdx_x_1; i < a36.sizes[0]; i += blockDim_x_1) {
        const IndexType30 aOffset49 = IndexToOffset<output_t28, IndexType30, ADims31>::get(i, a36);
        atomicAdd(& a36.data[aOffset49], smem45[i]);
    }
}
}
